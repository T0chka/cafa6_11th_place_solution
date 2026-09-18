import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.csr import pack_csr_from_pairs
from src.core.ontology import (
    build_ontology_graph,
    closure_csr_by_parents,
    parse_obo_snapshot,
    prune_orphans,
)

ROOT = Path("artifacts/evaluation/reference/raw/32861969/release_kaggle_final_May2026")
OUT = Path("artifacts/evaluation/prepared")
OUT.mkdir(parents=True, exist_ok=True)

ASPECT = {"P": "BPO", "C": "CCO", "F": "MFO"}
NS = {
    "BPO": "biological_process",
    "CCO": "cellular_component",
    "MFO": "molecular_function",
}

obo = parse_obo_snapshot(
    str(ROOT / "go-basic.obo"),
    ["biological_process", "cellular_component", "molecular_function"],
)
toi = set(pd.read_csv(
    ROOT / "May2026_groundtruth_terms_of_interest.txt",
    header=None,
)[0].astype(str))
ia_df = pd.read_csv(ROOT / "IA.tsv", sep="\t", header=None, names=["term", "ia"])
ia_map = dict(zip(ia_df["term"].astype(str), ia_df["ia"].astype(float)))

def read_terms(path):
    df = pd.read_csv(path, sep="\t")
    df["aspect"] = df["aspect"].map(ASPECT)
    return df

def build_csr(df, ids, term_to_idx):
    id_to_row = {x: i for i, x in enumerate(ids)}
    rows = df["EntryID"].map(id_to_row)
    cols = df["term"].map(term_to_idx)
    if rows.isna().any():
        raise ValueError(f"{int(rows.isna().sum())} unknown EntryIDs")
    if cols.isna().any():
        bad = df.loc[cols.isna(), "term"].drop_duplicates().tolist()[:20]
        raise ValueError(f"unknown GO terms: {bad}")
    return pack_csr_from_pairs(
        rows.to_numpy(dtype=np.int64),
        cols.to_numpy(dtype=np.int32),
        len(ids),
    )

known_all = read_terms(ROOT / "May2026_groundtruth_PK_known.tsv")
prepared = {}

for regime in ["NK", "LK", "PK"]:
    gt_all = read_terms(ROOT / f"May2026_groundtruth_{regime}.tsv")
    prepared[regime] = {}

    for aspect in ["BPO", "CCO", "MFO"]:
        graph = prune_orphans(build_ontology_graph(
            obo.edges_by_ns[NS[aspect]], obo.alt_to_canon
        ))
        terms = graph.term_ids.astype(object)
        term_to_idx = {str(t): i for i, t in enumerate(terms)}
        toi_mask = np.array([str(t) in toi for t in terms], dtype=bool)
        ia = np.array([ia_map.get(str(t), 0.0) for t in terms], dtype=np.float32)
        ia[~toi_mask] = 0.0

        gt = gt_all.loc[
            gt_all["aspect"] == aspect, ["EntryID", "term"]
        ].copy()
        ids = np.sort(gt["EntryID"].unique().astype(object))
        gt_seed_indptr, gt_seed_indices = build_csr(gt, ids, term_to_idx)
        gt_indptr, gt_indices = closure_csr_by_parents(
            gt_seed_indptr, gt_seed_indices,
            graph.parents_indptr, graph.parents_indices,
        )

        if regime == "PK":
            known = known_all.loc[
                (known_all["aspect"] == aspect) &
                (known_all["EntryID"].isin(ids)),
                ["EntryID", "term"],
            ].copy()
            known_seed_indptr, known_seed_indices = build_csr(
                known, ids, term_to_idx
            )
            known_indptr, known_indices = closure_csr_by_parents(
                known_seed_indptr, known_seed_indices,
                graph.parents_indptr, graph.parents_indices,
            )
        else:
            known_indptr = np.zeros(len(ids) + 1, dtype=np.int64)
            known_indices = np.empty(0, dtype=np.int32)

        prepared[regime][aspect] = {
            "ontology_term_ids": terms,
            "graph": {
                "parents_indptr": graph.parents_indptr,
                "parents_indices": graph.parents_indices,
                "children_indptr": graph.children_indptr,
                "children_indices": graph.children_indices,
            },
            "ia": ia,
            "toi_mask": toi_mask,
            "gt": {
                "gt_ids": ids,
                "gt_indptr": gt_indptr,
                "gt_indices": gt_indices,
            },
            "known_terms": {
                "known_ids": ids,
                "known_indptr": known_indptr,
                "known_indices": known_indices,
            },
        }

        print(
            regime, aspect, "proteins=", len(ids),
            "gt_raw=", len(gt_seed_indices),
            "gt=", len(gt_indices),
            "known=", len(known_indices),
        )

with open(OUT / "ground_truth.pkl", "wb") as f:
    pickle.dump(prepared, f, protocol=pickle.HIGHEST_PROTOCOL)

print("saved:", OUT / "ground_truth.pkl")
