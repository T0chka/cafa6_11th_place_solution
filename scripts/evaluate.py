import pickle
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.core.csr import pack_csr_from_pairs_with_data
from src.core.propagation import drop_terms_csr, propagate_max_up_csr, topk_from_csr
from src.core.scoring import score_weighted_f

ROOT = Path("artifacts/evaluation")
PRED = ROOT / "predictions"
PREPARED = ROOT / "prepared/ground_truth.pkl"
RESULTS = ROOT / "results"
METHODS = [
    "hmlp_esm2", "mlp_t5_esm1b", "pyb_t5",
    "blast_knn", "naive_prior", "nonexp", "ltr",
]
REGIMES = ["NK", "LK", "PK"]
ASPECTS = ["BPO", "CCO", "MFO"]
NS_TO_ASPECT = {
    "biological_process": "BPO",
    "cellular_component": "CCO",
    "molecular_function": "MFO",
}

t0 = time.perf_counter()

with PREPARED.open("rb") as f:
    prepared = pickle.load(f)

union_ids = {}
for aspect in ASPECTS:
    ids = [prepared[r][aspect]["gt"]["gt_ids"] for r in REGIMES]
    union_ids[aspect] = np.unique(np.concatenate(ids).astype(object))

id_parts = []
term_parts = []
for code, aspect in enumerate(ASPECTS):
    ids = union_ids[aspect]
    terms = prepared["NK"][aspect]["ontology_term_ids"]
    id_parts.append(pd.DataFrame({
        "EntryID": ids.astype(str),
        "aspect_code": np.full(len(ids), code, dtype=np.int8),
        "row_idx": np.arange(len(ids), dtype=np.int32),
    }))
    term_parts.append(pd.DataFrame({
        "term": terms.astype(str),
        "aspect_code": np.full(len(terms), code, dtype=np.int8),
        "term_idx": np.arange(len(terms), dtype=np.int32),
    }))

id_map = pd.concat(id_parts, ignore_index=True)
term_map = pd.concat(term_parts, ignore_index=True)

def load_states(path):
    con = duckdb.connect(database=":memory:")
    con.register("id_map", id_map)
    con.register("term_map", term_map)
    p = path.as_posix().replace("'", "''")
    df = con.execute(f"""
        SELECT i.aspect_code, i.row_idx, t.term_idx, CAST(p.score AS FLOAT) AS score
        FROM read_csv('{p}', delim='\t', header=false,
            columns={{'EntryID':'VARCHAR','term':'VARCHAR','score':'FLOAT'}}) p
        JOIN term_map t ON p.term = t.term
        JOIN id_map i ON p.EntryID = i.EntryID AND t.aspect_code = i.aspect_code
    """).df()
    con.close()

    states = {}
    for code, aspect in enumerate(ASPECTS):
        x = df.loc[df["aspect_code"] == code]
        indptr, indices, scores = pack_csr_from_pairs_with_data(
            x["row_idx"].to_numpy(dtype=np.int64),
            x["term_idx"].to_numpy(dtype=np.int32),
            x["score"].to_numpy(dtype=np.float32),
            len(union_ids[aspect]),
        )
        counts = np.diff(indptr)
        max_raw = int(counts.max()) if counts.size else 0
        if max_raw > 500:
            raise ValueError(f"{path.name} {aspect}: raw max terms={max_raw} > 500")

        graph = prepared["NK"][aspect]["graph"]
        indptr, indices, scores = propagate_max_up_csr(
            indptr, indices, scores,
            graph["parents_indptr"], graph["parents_indices"],
        )
        states[aspect] = (indptr, indices, scores)
    return states

def slice_state(state, source_ids, target_ids):
    indptr, indices, scores = state
    lookup = {x: i for i, x in enumerate(source_ids)}
    rows = np.fromiter((lookup[x] for x in target_ids), dtype=np.int64, count=len(target_ids))
    counts = indptr[rows + 1] - indptr[rows]

    out_indptr = np.empty(len(rows) + 1, dtype=np.int64)
    out_indptr[0] = 0
    np.cumsum(counts, out=out_indptr[1:])

    out_indices = np.empty(int(out_indptr[-1]), dtype=np.int32)
    out_scores = np.empty(int(out_indptr[-1]), dtype=np.float32)
    pos = 0
    for row in rows:
        s, e = int(indptr[row]), int(indptr[row + 1])
        n = e - s
        out_indices[pos:pos+n] = indices[s:e]
        out_scores[pos:pos+n] = scores[s:e]
        pos += n

    return out_indptr, out_indices, out_scores

def score_method(method):
    states = load_states(PRED / f"{method}.tsv")
    rows = []

    for regime in REGIMES:
        pred_terms = {}
        pred_scores = {}

        for aspect in ASPECTS:
            aspect_gt = prepared[regime][aspect]
            ids = aspect_gt["gt"]["gt_ids"]
            state = slice_state(states[aspect], union_ids[aspect], ids)

            if regime == "PK":
                known = aspect_gt["known_terms"]
                if not np.array_equal(known["known_ids"], ids):
                    raise ValueError(f"{regime} {aspect}: known IDs not aligned")
                state = drop_terms_csr(
                    state[0], state[1], state[2],
                    known["known_indptr"], known["known_indices"],
                )

            counts = np.diff(state[0])
            max_k = max(1, int(counts.max()) if counts.size else 0)
            pos, scores = topk_from_csr(
                state[0], state[1], state[2], top_k=max_k, max_k=max_k
            )
            pred_terms[aspect] = (ids, pos)
            pred_scores[aspect] = scores

        metrics = score_weighted_f(
            prepared_gt=prepared[regime],
            pred_terms_by_aspect=pred_terms,
            pred_scores_by_aspect=pred_scores,
            th_step=0.001,
            eval_on="withheld" if regime == "PK" else "gt",
        )
        best = metrics.sort_values(
            ["aspect", "f_micro"], ascending=[True, False]
        ).drop_duplicates("aspect")

        for x in best.itertuples():
            rows.append({
                "method": method,
                "regime": regime,
                "aspect": x.aspect,
                "tau_fast": float(x.tau),
                "f_micro_fast": float(x.f_micro),
            })

    return rows

fast_rows = []
for method in METHODS:
    m0 = time.perf_counter()
    fast_rows.extend(score_method(method))
    print(f"{method}: {time.perf_counter() - m0:.2f}s")

fast = pd.DataFrame(fast_rows)

official_parts = []
for regime in REGIMES:
    x = pd.read_csv(
        RESULTS / regime / "evaluation_best_f_micro_w.tsv", sep="\t"
    )
    x["method"] = x["filename"].str.replace(".tsv", "", regex=False)
    x["regime"] = regime
    x["aspect"] = x["ns"].map(NS_TO_ASPECT)
    official_parts.append(
        x[["method", "regime", "aspect", "tau", "f_micro_w"]]
        .rename(
            columns={
                "tau": "tau_official",
                "f_micro_w": "f_micro_official",
            }
        )
    )

official = pd.concat(official_parts, ignore_index=True)
cmp = official.merge(fast, on=["method", "regime", "aspect"], how="inner")
cmp["f_micro_fast_4"] = cmp["f_micro_fast"].round(4)
cmp["tau_fast_4"] = cmp["tau_fast"].round(4)
cmp["df"] = cmp["f_micro_fast_4"] - cmp["f_micro_official"]
cmp["dtau"] = cmp["tau_fast_4"] - cmp["tau_official"]
cmp = cmp.sort_values(["method", "regime", "aspect"])

out = ROOT / "official_vs_fast_f_micro.tsv"
cmp.to_csv(out, sep="\t", index=False)

summary_official = official.groupby("method")["f_micro_official"].mean()
summary_fast = fast.groupby("method")["f_micro_fast"].mean()
summary = pd.concat(
    [summary_official.rename("official"), summary_fast.rename("fast")], axis=1
)
summary["fast_4"] = summary["fast"].round(4)
summary["diff"] = summary["fast_4"] - summary["official"]

elapsed = time.perf_counter() - t0

print()
print(summary.sort_index().to_string(float_format=lambda x: f"{x:.6f}"))
print()
print(f"cells compared : {len(cmp)}/63")
print(f"exact f_micro @4dp: {(cmp['df'].abs() < 5e-8).sum()}/63")
print(f"max |f_micro diff|: {cmp['df'].abs().max():.6f}")
print(f"max |tau diff| : {cmp['dtau'].abs().max():.6f}")
print(f"elapsed        : {elapsed:.3f}s")
print(f"speedup        : {4590.906 / elapsed:.1f}x")
print(f"saved          : {out}")
