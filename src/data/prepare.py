"""
Prepare the canonical dataset representation consumed by embeddings and models.

Inputs from DatasetSpec:
- competition train FASTA;
- competition test FASTA;
- original competition train terms;
- GO ontology OBO;
- information-accretion file;
- when use_uniprot=True, UniProt-derived files produced by src/data/uniprot.py:
  train_terms_uni_updated.tsv, uniprot_terms_test_prots.parquet,
  uniprot_not_qualifier.parquet, uniprot_nonexp_ohe.parquet and optionally
  extra_train_sequences.fasta.

Outputs in DatasetSpec.prepared_dir:
1) train_index.parquet
   Columns: EntryID, seq_key, length, sequence.
   EntryIDs are defined by the effective training-terms file. Sequences are
   resolved in this order: train FASTA, optional extra-train FASTA, test FASTA.
   seq_key is SHA1(sequence).

2) test_index.parquet
   Columns: EntryID, seq_key, length, sequence for every test FASTA record.

3) ground_truth.pkl
   Dict keyed by BPO/CCO/MFO. Each aspect contains a canonical GO term axis,
   parent/child CSR graph, IA weights, propagated training ground truth,
   original known competition terms, NOT constraints, UniProt experimental and
   non-experimental test terms, and non-experimental evidence-code bit masks.

All GO annotations are canonicalized against the dataset OBO snapshot.
Positive annotations are propagated upward through is_a and part_of parents.
NOT annotations and evidence-code flags are not propagated.

Existing train/test index and ground_truth outputs are reused. Delete an output
to rebuild it.
"""

import hashlib
import pickle
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.csr import pack_row_aligned_csr, pack_row_aligned_csr_with_data
from src.core.io import load_terms_df
from src.core.ontology import (
    OntologyGraph,
    build_ontology_graph,
    canonize_term,
    closure_csr_by_parents,
    parse_obo_snapshot,
    prune_orphans,
    read_ia_tsv,
)
from src.data.dataset import DatasetSpec
from src.data.gaf import normalize_gaf
from src.data.uniprot import ASPECT_TO_NS, update_uniprot_annotations


def _parse_entry_id_from_header(header: str) -> str:
    parts = header.strip().split("|")
    if len(parts) >= 2 and parts[1]:
        return parts[1]
    first = header.strip().split()[0]
    if first:
        return first
    raise ValueError(f"Unexpected FASTA header format: {header!r}")


def _iter_fasta(path: Path) -> Iterator[tuple[str, str]]:
    header = None
    seq = []
    with path.open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield _parse_entry_id_from_header(header), "".join(seq)
                header = line[1:].strip()
                seq = []
            else:
                seq.append(line)
        if header is not None:
            yield _parse_entry_id_from_header(header), "".join(seq)


def _seq_key(sequence: str) -> str:
    return hashlib.sha1(sequence.encode("utf-8")).hexdigest()


def _effective_train_terms(spec: DatasetSpec) -> Path:
    return spec.updated_train_terms if spec.use_uniprot else spec.train_terms


def build_indices(spec: DatasetSpec) -> None:
    terms_path = _effective_train_terms(spec)
    terms_df = load_terms_df(terms_path)
    need_ids = set(terms_df["EntryID"].unique().tolist())

    train_rows = []
    test_rows = []
    seq_dict = {}
    found_ids = set()

    train_fastas = [spec.train_fasta]
    if spec.use_uniprot and spec.extra_train_fasta.exists():
        train_fastas.append(spec.extra_train_fasta)
    train_fastas.append(spec.test_fasta)

    for fasta_path in train_fastas:
        for entry_id, seq in _iter_fasta(fasta_path):
            entry_id = str(entry_id)
            if entry_id not in need_ids or entry_id in found_ids:
                continue
            key = _seq_key(seq)
            train_rows.append((entry_id, key, len(seq), seq))
            old = seq_dict.get(key)
            if old is None:
                seq_dict[key] = seq
            elif old != seq:
                raise ValueError(f"Inconsistent sequence for seq_key={key}")
            found_ids.add(entry_id)

    missing = sorted(need_ids - found_ids)
    if missing:
        raise ValueError(
            f"{len(missing)} training EntryIDs have no sequence in train/extra/test FASTA. "
            f"Sample: {', '.join(missing[:20])}"
        )

    for entry_id, seq in _iter_fasta(spec.test_fasta):
        entry_id = str(entry_id)
        key = _seq_key(seq)
        test_rows.append((entry_id, key, len(seq), seq))
        old = seq_dict.get(key)
        if old is None:
            seq_dict[key] = seq
        elif old != seq:
            raise ValueError(f"Inconsistent sequence for seq_key={key}")

    train_df = pd.DataFrame(
        train_rows, columns=["EntryID", "seq_key", "length", "sequence"]
    )
    test_df = pd.DataFrame(
        test_rows, columns=["EntryID", "seq_key", "length", "sequence"]
    )

    for name, df in (("train", train_df), ("test", test_df)):
        dup = df.groupby("EntryID", sort=False)["seq_key"].nunique()
        bad = dup[dup > 1]
        if not bad.empty:
            raise ValueError(f"{len(bad)} EntryIDs in {name} map to multiple sequences.")

    train_df = train_df.drop_duplicates("EntryID", keep="first")
    test_df = test_df.drop_duplicates("EntryID", keep="first")
    spec.prepared_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(spec.train_index, index=False)
    test_df.to_parquet(spec.test_index, index=False)

    print(
        f"[prepare] train index: {len(train_df):,} rows, "
        f"{train_df['seq_key'].nunique():,} unique sequences -> {spec.train_index}"
    )
    print(
        f"[prepare] test index: {len(test_df):,} rows, "
        f"{test_df['seq_key'].nunique():,} unique sequences -> {spec.test_index}"
    )


def _build_entry_csr(
    entry_ids: np.ndarray,
    term_ids: np.ndarray,
    graph: OntologyGraph,
    alt_to_canon: dict[str, str],
    propagate_up: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    term_to_idx = {t: int(i) for i, t in enumerate(graph.term_ids.tolist())}
    row_ids = np.unique(np.asarray(entry_ids, dtype=object))
    term_ids = np.asarray(term_ids, dtype=object)
    col_index = np.fromiter(
        (term_to_idx.get(canonize_term(str(t), alt_to_canon), -1) for t in term_ids),
        dtype=np.int32,
        count=int(term_ids.size),
    )

    n_missing = int((col_index < 0).sum())
    if n_missing:
        raise ValueError(f"{n_missing} terms are missing in the OBO ontology.")

    indptr, indices = pack_row_aligned_csr(
        row_ids_sorted=row_ids,
        entry_ids=np.asarray(entry_ids, dtype=object),
        col_index=col_index,
    )
    if propagate_up:
        indptr, indices = closure_csr_by_parents(
            seed_indptr=indptr,
            seed_indices=indices,
            parents_indptr=graph.parents_indptr,
            parents_indices=graph.parents_indices,
        )
    return row_ids, indptr, indices


def _build_entry_csr_with_data(
    entry_ids: np.ndarray,
    term_ids: np.ndarray,
    data: np.ndarray,
    graph: OntologyGraph,
    alt_to_canon: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    term_to_idx = {t: int(i) for i, t in enumerate(graph.term_ids.tolist())}
    entry_ids = np.asarray(entry_ids, dtype=object)
    term_ids = np.asarray(term_ids, dtype=object)
    data = np.asarray(data)
    row_ids = np.unique(entry_ids)

    col_index = np.fromiter(
        (term_to_idx.get(canonize_term(str(t), alt_to_canon), -1) for t in term_ids),
        dtype=np.int32,
        count=int(term_ids.size),
    )
    n_missing = int((col_index < 0).sum())
    if n_missing:
        raise ValueError(f"{n_missing} terms are missing in the OBO ontology.")

    indptr, indices, out_data = pack_row_aligned_csr_with_data(
        row_ids_sorted=row_ids,
        entry_ids=entry_ids,
        col_index=col_index,
        data=data,
    )
    return row_ids, indptr, indices, out_data


def _empty_csr() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.empty(0, dtype=object),
        np.zeros(1, dtype=np.int64),
        np.empty(0, dtype=np.int32),
    )


def build_ground_truth(spec: DatasetSpec) -> None:
    gt_terms_df = load_terms_df(_effective_train_terms(spec))
    known_terms_df = load_terms_df(spec.train_terms)

    obo = parse_obo_snapshot(str(spec.ontology_obo), list(ASPECT_TO_NS.values()))
    alt_to_canon = obo.alt_to_canon

    ia_map = read_ia_tsv(str(spec.ia_file))
    ia_map_canon = {}
    for term_id, weight in ia_map.items():
        canon = canonize_term(term_id, alt_to_canon)
        if canon in ia_map_canon:
            raise ValueError(f"IA canonicalization collision: {canon}")
        ia_map_canon[canon] = float(weight)

    not_df = (
        pd.read_parquet(spec.not_terms_uniprot)
        if spec.use_uniprot and spec.not_terms_uniprot.exists()
        else None
    )
    test_terms_df = (
        pd.read_parquet(spec.test_terms_uniprot)
        if spec.use_uniprot and spec.test_terms_uniprot.exists()
        else None
    )
    nonexp_df = (
        pd.read_parquet(spec.nonexp_codes_uniprot)
        if spec.use_uniprot and spec.nonexp_codes_uniprot.exists()
        else None
    )

    nonexp_cols = []
    nonexp_code_names = np.empty(0, dtype=object)
    if nonexp_df is not None:
        nonexp_cols = sorted(c for c in nonexp_df.columns if c.startswith("nonexp_"))
        if not nonexp_cols:
            raise ValueError("Non-experimental evidence parquet has no nonexp_* columns.")
        if len(nonexp_cols) > 16:
            raise ValueError("More than 16 non-experimental codes cannot fit uint16.")
        nonexp_code_names = np.asarray(
            [col.removeprefix("nonexp_") for col in nonexp_cols],
            dtype=object,
        )

    prepared = {}
    for aspect, namespace in ASPECT_TO_NS.items():
        graph = build_ontology_graph(
            edges=obo.edges_by_ns[namespace],
            alt_to_canon=alt_to_canon,
        )
        graph = prune_orphans(graph)
        term_ids = graph.term_ids.astype(object, copy=False)

        ia = np.fromiter(
            (float(ia_map_canon.get(str(term), 0.0)) for term in term_ids),
            dtype=np.float32,
            count=int(term_ids.size),
        )

        block = gt_terms_df.loc[
            gt_terms_df["aspect"] == aspect, ["EntryID", "term"]
        ]
        gt_ids, gt_indptr, gt_indices = _build_entry_csr(
            block["EntryID"].to_numpy(copy=False),
            block["term"].to_numpy(copy=False),
            graph, alt_to_canon, True,
        )

        block = known_terms_df.loc[
            known_terms_df["aspect"] == aspect, ["EntryID", "term"]
        ]
        known_ids, known_indptr, known_indices = _build_entry_csr(
            block["EntryID"].to_numpy(copy=False),
            block["term"].to_numpy(copy=False),
            graph, alt_to_canon, True,
        )

        if not_df is not None:
            block = not_df.loc[not_df["aspect"] == aspect, ["EntryID", "term"]]
            not_ids, not_indptr, not_indices = _build_entry_csr(
                block["EntryID"].to_numpy(copy=False),
                block["term"].to_numpy(copy=False),
                graph, alt_to_canon, False,
            )
        else:
            not_ids, not_indptr, not_indices = _empty_csr()

        if test_terms_df is not None:
            block = test_terms_df.loc[
                test_terms_df["aspect"] == aspect,
                ["EntryID", "term", "is_exp_confirmed"],
            ]
            exp_mask = block["is_exp_confirmed"].to_numpy(dtype=bool, copy=False)

            exp = block.loc[exp_mask, ["EntryID", "term"]]
            test_exp_ids, test_exp_indptr, test_exp_indices = _build_entry_csr(
                exp["EntryID"].to_numpy(copy=False),
                exp["term"].to_numpy(copy=False),
                graph, alt_to_canon, True,
            )

            nonexp = block.loc[~exp_mask, ["EntryID", "term"]]
            test_nonexp_ids, test_nonexp_indptr, test_nonexp_indices = _build_entry_csr(
                nonexp["EntryID"].to_numpy(copy=False),
                nonexp["term"].to_numpy(copy=False),
                graph, alt_to_canon, True,
            )
        else:
            test_exp_ids, test_exp_indptr, test_exp_indices = _empty_csr()
            test_nonexp_ids, test_nonexp_indptr, test_nonexp_indices = _empty_csr()

        if nonexp_df is not None:
            block = nonexp_df.loc[
                nonexp_df["aspect"] == aspect,
                ["EntryID", "term"] + nonexp_cols,
            ]
            grouped = block.groupby(
                ["EntryID", "term"], sort=False, observed=True
            )[nonexp_cols].max()

            mask = np.zeros(len(grouped), dtype=np.uint16)
            for bit, col in enumerate(nonexp_cols):
                values = grouped[col].to_numpy(dtype=np.uint16, copy=False)
                mask |= values << bit

            entry_ids = grouped.index.get_level_values(0).to_numpy(dtype=object, copy=False)
            terms = grouped.index.get_level_values(1).to_numpy(dtype=object, copy=False)
            nonexp_ids, nonexp_indptr, nonexp_indices, nonexp_data = (
                _build_entry_csr_with_data(
                    entry_ids, terms, mask, graph, alt_to_canon
                )
            )
        else:
            nonexp_ids = np.empty(0, dtype=object)
            nonexp_indptr = np.zeros(1, dtype=np.int64)
            nonexp_indices = np.empty(0, dtype=np.int32)
            nonexp_data = np.empty(0, dtype=np.uint16)

        prepared[aspect] = {
            "ontology_term_ids": term_ids,
            "graph": {
                "parents_indptr": graph.parents_indptr.astype(np.int64, copy=False),
                "parents_indices": graph.parents_indices.astype(np.int32, copy=False),
                "children_indptr": graph.children_indptr.astype(np.int64, copy=False),
                "children_indices": graph.children_indices.astype(np.int32, copy=False),
            },
            "ia": ia,
            "gt": {
                "gt_ids": gt_ids,
                "gt_indptr": gt_indptr.astype(np.int64, copy=False),
                "gt_indices": gt_indices.astype(np.int32, copy=False),
            },
            "known_terms": {
                "known_ids": known_ids,
                "known_indptr": known_indptr.astype(np.int64, copy=False),
                "known_indices": known_indices.astype(np.int32, copy=False),
            },
            "not_terms": {
                "not_ids": not_ids,
                "not_indptr": not_indptr.astype(np.int64, copy=False),
                "not_indices": not_indices.astype(np.int32, copy=False),
            },
            "test_exp": {
                "test_exp_ids": test_exp_ids,
                "test_exp_indptr": test_exp_indptr.astype(np.int64, copy=False),
                "test_exp_indices": test_exp_indices.astype(np.int32, copy=False),
            },
            "test_nonexp": {
                "test_nonexp_ids": test_nonexp_ids,
                "test_nonexp_indptr": test_nonexp_indptr.astype(np.int64, copy=False),
                "test_nonexp_indices": test_nonexp_indices.astype(np.int32, copy=False),
            },
            "nonexp_codes": {
                "nonexp_ids": nonexp_ids,
                "nonexp_indptr": nonexp_indptr.astype(np.int64, copy=False),
                "nonexp_indices": nonexp_indices.astype(np.int32, copy=False),
                "nonexp_data": nonexp_data.astype(np.uint16, copy=False),
                "nonexp_code_names": nonexp_code_names,
            },
        }

    with spec.ground_truth.open("wb") as handle:
        pickle.dump(prepared, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[prepare] wrote: {spec.ground_truth}")
    for aspect, data in prepared.items():
        print(
            f"[prepare] {aspect}: terms={len(data['ontology_term_ids'])} | "
            f"gt={len(data['gt']['gt_ids'])} | "
            f"known={len(data['known_terms']['known_ids'])} | "
            f"not={len(data['not_terms']['not_ids'])} | "
            f"test_exp={len(data['test_exp']['test_exp_ids'])} | "
            f"test_nonexp={len(data['test_nonexp']['test_nonexp_ids'])} | "
            f"nonexp_codes={len(data['nonexp_codes']['nonexp_ids'])}"
        )


def prepare_dataset(spec: DatasetSpec) -> None:
    print("\n=== Preparing dataset ===")
    spec.prepared_dir.mkdir(parents=True, exist_ok=True)
    spec.cache_dir.mkdir(parents=True, exist_ok=True)

    if spec.use_uniprot:
        if not spec.uniprot_gaf_parquet.exists():
            if spec.uniprot_gaf_gz is None or not spec.uniprot_gaf_gz.exists():
                raise FileNotFoundError(
                    "UniProt snapshot is missing. Provide either the normalized "
                    "Parquet or the raw .gaf.gz file declared by DatasetSpec."
                )
            normalize_gaf(spec.uniprot_gaf_gz, spec.uniprot_gaf_parquet)
        update_uniprot_annotations(spec)

    if not spec.train_index.exists() or not spec.test_index.exists():
        build_indices(spec)
    else:
        print("[prepare] train/test indices already exist")

    if not spec.ground_truth.exists():
        build_ground_truth(spec)
    else:
        print("[prepare] ground_truth.pkl already exists")
