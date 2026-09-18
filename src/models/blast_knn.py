from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.csr import pack_csr_from_pairs_with_data
from src.core.io import load_index_df, load_prepared_gt, load_terms_df
from src.core.postprocess import CSRState, PostprocessConfig, Postprocessor
from src.data.dataset import DatasetSpec


@dataclass(frozen=True)
class BlastKNNConfig:
    k_neighbors: dict[str, int] = field(default_factory=lambda: {"BPO": 30, "CCO": 50, "MFO": 30})
    evalue_max: float = 1e-3
    top_n: int = 500
    n_folds: int = 5
    seed: int = 1001
    collapse_hsps: bool = True


POSTPROCESS = PostprocessConfig(top_k=500, min_score=0.0, drop_zero_ia=True, exclude_not_descendants=True, drop_weak_preds=False)


def _terms_path(dataset: DatasetSpec) -> Path:
    return dataset.updated_train_terms if dataset.use_uniprot else dataset.train_terms


def prepare_hits(hits: pd.DataFrame, config: BlastKNNConfig) -> pd.DataFrame:
    required = ["qseqid", "sseqid", "bitscore", "evalue"]
    missing = [col for col in required if col not in hits.columns]
    if missing:
        raise ValueError(f"BLAST hits missing columns: {missing}")
    hits = hits.loc[:, required].copy()
    hits = hits.loc[hits["evalue"] <= np.float64(config.evalue_max)]
    hits = hits.loc[hits["qseqid"] != hits["sseqid"]]
    if config.collapse_hsps:
        hits = hits.sort_values(["qseqid", "sseqid", "bitscore"], ascending=[True, True, False], kind="mergesort")
        hits = hits.drop_duplicates(["qseqid", "sseqid"], keep="first")
    return hits.sort_values(["qseqid", "bitscore"], ascending=[True, False], kind="mergesort").reset_index(drop=True)


def make_term_scores(hits: pd.DataFrame, train_terms_aspect: pd.DataFrame, aspect: str, k_neighbors: int) -> pd.DataFrame:
    if hits.empty or train_terms_aspect.empty:
        return pd.DataFrame(columns=["qseqid", "term", "aspect", "score"])
    donors = pd.unique(train_terms_aspect["seq_key"])
    hits = hits.loc[hits["sseqid"].isin(donors)]
    hits = hits.groupby("qseqid", sort=False, as_index=False).head(int(k_neighbors)).copy()
    if hits.empty:
        return pd.DataFrame(columns=["qseqid", "term", "aspect", "score"])
    denom = hits.groupby("qseqid", sort=False)["bitscore"].transform("sum").to_numpy(dtype=np.float64, copy=False)
    hits["weight"] = (hits["bitscore"].to_numpy(dtype=np.float64, copy=False) / denom).astype(np.float32, copy=False)
    scores = hits.loc[:, ["qseqid", "sseqid", "weight"]].merge(train_terms_aspect, left_on="sseqid", right_on="seq_key", how="inner", copy=False)
    scores = scores.groupby(["qseqid", "term"], sort=False, as_index=False)["weight"].sum().rename(columns={"weight": "score"})
    scores["score"] = scores["score"].astype(np.float32, copy=False).clip(0.0, 1.0)
    scores["aspect"] = aspect
    return scores.loc[:, ["qseqid", "term", "aspect", "score"]]


def _folds(n: int, n_folds: int, seed: int):
    order = np.arange(n, dtype=np.int64)
    np.random.RandomState(seed).shuffle(order)
    sizes = np.full(n_folds, n // n_folds, dtype=np.int64)
    sizes[:n % n_folds] += 1
    start = 0
    for size in sizes:
        valid = order[start:start + int(size)]
        train = np.concatenate([order[:start], order[start + int(size):]])
        start += int(size)
        yield train, valid


def _long_to_npz(df: pd.DataFrame, index_df: pd.DataFrame, prepared_gt: dict, data_type: str, out_dir: Path) -> None:
    entry_ids = pd.unique(index_df["EntryID"]).astype(object, copy=False)
    entry_pos = {str(entry_id): i for i, entry_id in enumerate(entry_ids.tolist())}
    postprocess = Postprocessor(POSTPROCESS, index_df, prepared_gt)
    prefix = "oof" if data_type == "oof" else "submit"
    target = out_dir / prefix
    target.mkdir(parents=True, exist_ok=True)
    for aspect, aspect_gt in prepared_gt.items():
        block = df.loc[df["aspect"] == aspect, ["EntryID", "term", "score"]].copy()
        if block.empty:
            term_pos = np.empty(0, dtype=np.int32)
            row_pos = np.empty(0, dtype=np.int64)
            scores = np.empty(0, dtype=np.float32)
        else:
            term_pos = postprocess.map_terms_to_pos(aspect, block["term"].to_numpy(dtype=object, copy=False))
            row_pos = np.fromiter((entry_pos[str(entry_id)] for entry_id in block["EntryID"]), dtype=np.int64, count=len(block))
            scores = block["score"].to_numpy(dtype=np.float32, copy=False)
        indptr, indices, values = pack_csr_from_pairs_with_data(row_pos, term_pos, scores, len(entry_ids))
        state = CSRState(indptr=indptr, indices=indices, scores=values)
        topk_pos, topk_scores = postprocess.postprocess_state(state=state, data_type=data_type, entry_ids=entry_ids, aspect_name=aspect, propagate=False, add_nonexp_terms=False, add_exp_terms=False, drop_known=(data_type == "test"))
        np.savez_compressed(target / f"{prefix}_for_ltr_{aspect}.npz", entry_ids=entry_ids, term_pos=topk_pos, scores=topk_scores)


def build_blast_component(dataset: DatasetSpec, train_hits_path: str | Path, test_hits_path: str | Path, out_dir: str | Path | None = None, config: BlastKNNConfig | None = None) -> Path:
    config = BlastKNNConfig() if config is None else config
    out_dir = dataset.prepared_dir.parent / "predictors/blast_knn" if out_dir is None else Path(out_dir)
    train_index = load_index_df(dataset.train_index)
    test_index = load_index_df(dataset.test_index)
    prepared_gt = load_prepared_gt(dataset.ground_truth)
    train_terms = load_terms_df(_terms_path(dataset))
    entry_to_seq = dict(zip(train_index["EntryID"], train_index["seq_key"]))
    train_terms["seq_key"] = train_terms["EntryID"].map(entry_to_seq)
    train_terms = train_terms.dropna(subset=["seq_key"]).loc[:, ["seq_key", "term", "aspect"]].drop_duplicates()
    train_hits = prepare_hits(pd.read_parquet(train_hits_path), config)
    test_hits = prepare_hits(pd.read_parquet(test_hits_path), config)
    seq_keys = np.unique(train_index["seq_key"].to_numpy(copy=False))
    seq_map = train_index.loc[:, ["EntryID", "seq_key"]].drop_duplicates().rename(columns={"seq_key": "qseqid"})
    oof_parts = []
    for train_idx, valid_idx in _folds(len(seq_keys), config.n_folds, config.seed):
        train_keys = seq_keys[train_idx]
        valid_keys = seq_keys[valid_idx]
        fold_hits = train_hits.loc[train_hits["qseqid"].isin(valid_keys) & train_hits["sseqid"].isin(train_keys)]
        for aspect, k in config.k_neighbors.items():
            terms = train_terms.loc[(train_terms["aspect"] == aspect) & train_terms["seq_key"].isin(train_keys), ["seq_key", "term"]].drop_duplicates()
            scores = make_term_scores(fold_hits, terms, aspect, k)
            if scores.empty:
                continue
            part = seq_map.merge(scores, on="qseqid", how="inner", copy=False).loc[:, ["EntryID", "term", "score", "aspect"]]
            part = part.sort_values(["EntryID", "aspect", "score"], ascending=[True, True, False], kind="mergesort").groupby(["EntryID", "aspect"], sort=False, as_index=False).head(config.top_n)
            oof_parts.append(part)
    oof = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame(columns=["EntryID", "term", "score", "aspect"])
    _long_to_npz(oof, train_index, prepared_gt, "oof", out_dir)
    test_map = test_index.loc[:, ["EntryID", "seq_key"]].drop_duplicates().rename(columns={"seq_key": "qseqid"})
    test_parts = []
    test_keys = pd.unique(test_map["qseqid"])
    test_hits = test_hits.loc[test_hits["qseqid"].isin(test_keys)]
    for aspect, k in config.k_neighbors.items():
        terms = train_terms.loc[train_terms["aspect"] == aspect, ["seq_key", "term"]].drop_duplicates()
        scores = make_term_scores(test_hits, terms, aspect, k)
        if scores.empty:
            continue
        part = test_map.merge(scores, on="qseqid", how="inner", copy=False).loc[:, ["EntryID", "term", "score", "aspect"]]
        part = part.sort_values(["EntryID", "aspect", "score"], ascending=[True, True, False], kind="mergesort").groupby(["EntryID", "aspect"], sort=False, as_index=False).head(config.top_n)
        test_parts.append(part)
    submit = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=["EntryID", "term", "score", "aspect"])
    _long_to_npz(submit, test_index, prepared_gt, "test", out_dir)
    return out_dir
