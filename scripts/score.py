import numpy as np
import pandas as pd

from config import dataset
from src.core.io import load_prepared_gt
from src.core.scoring import score_weighted_f, score_weighted_ndcg

ds = dataset()
gt = load_prepared_gt(ds.ground_truth)
df = pd.read_parquet("artifacts/ltr/oof/oof_ltr.parquet")

pred_terms = {}
pred_scores = {}

for aspect, aspect_gt in gt.items():
    x = df.loc[df["aspect"] == aspect]
    term_to_idx = {str(t): i for i, t in enumerate(aspect_gt["ontology_term_ids"])}
    entry_ids = x["EntryID"].drop_duplicates().to_numpy(dtype=object)
    groups = {k: g for k, g in x.groupby("EntryID", sort=False)}
    max_k = max(len(groups[k]) for k in entry_ids)
    pos = np.full((len(entry_ids), max_k), -1, dtype=np.int32)
    scores = np.zeros((len(entry_ids), max_k), dtype=np.float32)

    for i, entry_id in enumerate(entry_ids):
        g = groups[entry_id]
        terms = g["term"].map(term_to_idx).to_numpy()
        assert not pd.isna(terms).any()
        n = len(g)
        pos[i, :n] = terms.astype(np.int32)
        scores[i, :n] = g["score"].to_numpy(dtype=np.float32)

    pred_terms[aspect] = (entry_ids, pos)
    pred_scores[aspect] = scores

metrics = score_weighted_f(gt, pred_terms, pred_scores, eval_on="gt")
ndcg = score_weighted_ndcg(gt, pred_terms, pred_scores, eval_on="gt", ndcg_k=100)
best = metrics.sort_values(
    ["aspect", "f_micro"], ascending=[True, False]
).drop_duplicates("aspect")
out = best[["aspect", "tau", "f_micro"]].merge(
    ndcg[["aspect", "ndcg"]], on="aspect"
).sort_values("aspect")

print(out.to_string(index=False))
print(f"\nAVG CAFA_f_micro = {out['f_micro'].mean():.4f}")
print(f"AVG nDCG         = {out['ndcg'].mean():.4f}")
