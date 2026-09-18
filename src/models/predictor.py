from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy import sparse

from src.core.io import load_embeddings_for_index, load_index_df, load_prepared_gt
from src.core.postprocess import PostprocessConfig, Postprocessor
from src.data.dataset import DatasetSpec


@dataclass(frozen=True)
class PredictorSpec:
    name: str
    model: Literal["hmlp", "mlp", "pyboost"]
    embeddings: tuple[str, ...]
    min_freq: dict[str, int]
    min_pos_per_row: int = 2
    config_overrides: dict[str, object] | None = None


def make_model(spec: PredictorSpec):
    if spec.model == "hmlp":
        from src.models.hmlp import HMLPConfig, HMLPModel
        config = HMLPConfig()
        model_cls = HMLPModel
    elif spec.model == "mlp":
        from src.models.mlp import MLPConfig, MLPModel
        config = MLPConfig()
        model_cls = MLPModel
    elif spec.model == "pyboost":
        from src.models.pyboost import PyBoostConfig, PyBoostModel
        config = PyBoostConfig()
        model_cls = PyBoostModel
    else:
        raise ValueError(f"Unknown model: {spec.model}")
    if spec.config_overrides:
        config = replace(config, **spec.config_overrides)
    return model_cls(config)


LTR_POSTPROCESS = PostprocessConfig(
    top_k=500,
    min_score=0.0,
    drop_zero_ia=True,
    exclude_not_descendants=True,
    drop_weak_preds=False,
)


def build_targets_from_gt(
    aspect_gt: dict,
    index_df: pd.DataFrame,
    filter_0_ia: bool = False,
) -> tuple[np.ndarray, sparse.csr_matrix, np.ndarray]:
    gt = aspect_gt["gt"]
    gt_ids = gt["gt_ids"]
    gt_indptr = gt["gt_indptr"]
    gt_indices = gt["gt_indices"]

    mapping = index_df.loc[:, ["EntryID", "seq_key"]].drop_duplicates()
    prot_to_seq = mapping.set_index("EntryID")["seq_key"]
    seq_key_by_row = prot_to_seq.reindex(gt_ids).to_numpy()
    missing_mask = pd.isna(seq_key_by_row)
    if bool(missing_mask.any()):
        raise ValueError(f"{int(missing_mask.sum())} EntryID missing in index_df.")

    seq_keys, inv = np.unique(seq_key_by_row, return_inverse=True)
    nnz_per_row = np.diff(gt_indptr)
    rows_rep = np.repeat(inv, nnz_per_row)
    used_obo_cols = np.unique(gt_indices)
    used_obo_cols.sort()

    n_ontology = int(aspect_gt["ontology_term_ids"].shape[0])
    col_remap = np.full(n_ontology, -1, dtype=np.int32)
    col_remap[used_obo_cols] = np.arange(int(used_obo_cols.size), dtype=np.int32)
    cols = col_remap[gt_indices].astype(np.int32, copy=False)

    y = sparse.csr_matrix(
        (np.ones(int(cols.size), dtype=np.float32), (rows_rep, cols)),
        shape=(int(seq_keys.shape[0]), int(used_obo_cols.size)),
    )
    y.data[:] = 1.0
    term_ids = aspect_gt["ontology_term_ids"][used_obo_cols]

    if filter_0_ia:
        ia_sel = aspect_gt["ia"].astype(np.float32, copy=False)[used_obo_cols]
        keep_cols = np.flatnonzero(ia_sel > 0.0)
        if int(keep_cols.size) == 0:
            raise ValueError("No terms with ia > 0.0 among GT terms.")
        y = y[:, keep_cols].tocsr()
        term_ids = term_ids[keep_cols]

    keep_rows = np.flatnonzero(y.getnnz(axis=1) > 0)
    return seq_keys[keep_rows], y[keep_rows, :].tocsr(), term_ids


def filter_by_term_freq(
    seq_keys: np.ndarray,
    y: sparse.csr_matrix,
    term_ids: np.ndarray,
    min_freq: int,
    min_pos_per_row: int = 2,
) -> tuple[np.ndarray, sparse.csr_matrix, np.ndarray]:
    n_rows = int(y.shape[0])
    freq = np.bincount(y.indices, minlength=y.shape[1])
    keep_cols = np.flatnonzero((freq > int(min_freq)) & (freq < n_rows))
    if keep_cols.size == 0:
        raise ValueError("No terms to keep after filtering.")

    y_kept = y[:, keep_cols]
    term_ids_kept = term_ids[keep_cols]
    row_nnz = y_kept.indptr[1:] - y_kept.indptr[:-1]
    keep_rows = np.flatnonzero(row_nnz >= max(1, int(min_pos_per_row)))
    if keep_rows.size == 0:
        raise ValueError("No rows to keep after row filtering.")

    return seq_keys[keep_rows], y_kept[keep_rows, :], term_ids_kept


def align_xy(
    seq_keys_x: np.ndarray,
    x: np.ndarray,
    seq_keys_y: np.ndarray,
    y: sparse.csr_matrix,
) -> tuple[np.ndarray, np.ndarray, sparse.csr_matrix]:
    y_row = {key: i for i, key in enumerate(seq_keys_y.tolist())}
    keep = np.fromiter(
        (key in y_row for key in seq_keys_x.tolist()),
        dtype=bool,
        count=seq_keys_x.shape[0],
    )
    seq_keys = seq_keys_x[keep]
    rows = np.fromiter(
        (y_row[key] for key in seq_keys.tolist()),
        dtype=np.int32,
        count=seq_keys.shape[0],
    )
    return seq_keys, x[keep], y[rows]


def embedding_dirs(dataset: DatasetSpec, spec: PredictorSpec) -> tuple[Path, ...]:
    root = dataset.prepared_dir.parent / "embeddings"
    return tuple(root / name for name in spec.embeddings)


def predictor_dir(dataset: DatasetSpec, spec: PredictorSpec) -> Path:
    return dataset.prepared_dir.parent / "predictors" / spec.name


def train_predictor(
    dataset: DatasetSpec,
    spec: PredictorSpec,
    out_dir: Path | None = None,
    debug: bool = True,
) -> None:
    out_dir = predictor_dir(dataset, spec) if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    oof_dir = out_dir / "oof"
    oof_dir.mkdir(parents=True, exist_ok=True)

    train_index = load_index_df(dataset.train_index)
    prepared_gt = load_prepared_gt(dataset.ground_truth)
    seq_keys_x, x = load_embeddings_for_index(
        train_index, embedding_dirs(dataset, spec), verbose=True
    )
    postprocess = Postprocessor(LTR_POSTPROCESS, train_index, prepared_gt)

    for aspect, min_freq in spec.min_freq.items():
        aspect_gt = prepared_gt[aspect]
        seq_keys_y, y, term_ids = build_targets_from_gt(aspect_gt, train_index)
        seq_keys_y, y, term_ids = filter_by_term_freq(
            seq_keys_y, y, term_ids, min_freq, spec.min_pos_per_row
        )
        seq_keys, x_aspect, y_aspect = align_xy(seq_keys_x, x, seq_keys_y, y)
        aspect_dir = out_dir / aspect
        aspect_dir.mkdir(parents=True, exist_ok=True)

        model = make_model(spec)
        result = model.fit(
            features=x_aspect,
            targets=y_aspect,
            term_ids=term_ids.tolist(),
            aspect_gt=aspect_gt,
            out_dir=aspect_dir,
            debug=debug,
            compute_metrics=True,
        )

        logits = result["oof_logits"].astype(np.float32, copy=False)
        probs = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32, copy=False)
        term_pos = postprocess.map_terms_to_pos(aspect, term_ids)
        entry_ids, probs = postprocess.map_seqs_to_prots(seq_keys, probs)
        state = postprocess.select_candidates(term_pos, probs)
        topk_pos, topk_scores = postprocess.postprocess_state(
            state=state,
            data_type="oof",
            entry_ids=entry_ids,
            aspect_name=aspect,
            propagate=False,
            add_nonexp_terms=False,
            add_exp_terms=False,
            drop_known=False,
        )
        np.savez_compressed(
            oof_dir / f"oof_for_ltr_{aspect}.npz",
            entry_ids=entry_ids,
            term_pos=topk_pos,
            scores=topk_scores,
        )


def predict_predictor(
    dataset: DatasetSpec,
    spec: PredictorSpec,
    model_dir: str | Path,
    index_df: pd.DataFrame | None = None,
    save_dir: str | Path | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    model_dir = Path(model_dir)
    if index_df is None:
        index_df = load_index_df(dataset.test_index)
    else:
        index_df = index_df.copy()

    prepared_gt = load_prepared_gt(dataset.ground_truth)
    seq_keys_x, x = load_embeddings_for_index(
        index_df, embedding_dirs(dataset, spec), verbose=True
    )
    postprocess = Postprocessor(LTR_POSTPROCESS, index_df, prepared_gt)
    output = {}

    if save_dir is not None:
        submit_dir = Path(save_dir) / "submit"
        submit_dir.mkdir(parents=True, exist_ok=True)
    else:
        submit_dir = None

    for aspect in spec.min_freq:
        aspect_gt = prepared_gt[aspect]
        aspect_dir = model_dir / aspect
        term_ids = np.load(aspect_dir / "term_ids.npy", allow_pickle=True)
        model = make_model(spec)
        logits = model.predict_logits_ensemble(
            features=x,
            aspect_gt=aspect_gt,
            aspect_dir=aspect_dir,
        )
        probs = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32, copy=False)
        term_pos = postprocess.map_terms_to_pos(aspect, term_ids)
        entry_ids, probs = postprocess.map_seqs_to_prots(seq_keys_x, probs)
        state = postprocess.select_candidates(term_pos, probs)
        topk_pos, topk_scores = postprocess.postprocess_state(
            state=state,
            data_type="test",
            entry_ids=entry_ids,
            aspect_name=aspect,
            propagate=False,
            add_nonexp_terms=False,
            add_exp_terms=False,
            drop_known=True,
        )
        output[aspect] = (entry_ids, topk_pos, topk_scores)

        if submit_dir is not None:
            np.savez_compressed(
                submit_dir / f"submit_for_ltr_{aspect}.npz",
                entry_ids=entry_ids,
                term_pos=topk_pos,
                scores=topk_scores,
            )

    return output


def build_predictor(dataset: DatasetSpec, spec: PredictorSpec, debug: bool = True) -> Path:
    out_dir = predictor_dir(dataset, spec)
    train_predictor(dataset, spec, out_dir=out_dir, debug=debug)
    predict_predictor(dataset, spec, model_dir=out_dir, save_dir=out_dir)
    return out_dir
