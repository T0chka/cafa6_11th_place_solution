from dataclasses import dataclass, field
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd

from src.candidates.union import align_state_to_protein_axis, pack_union_dataset
from src.core.io import build_submission_df, load_index_df, load_prepared_gt, save_submit_parquet, save_submit_tsv
from src.core.postprocess import CSRState, PostprocessConfig, Postprocessor
from src.data.dataset import DatasetSpec
from src.ltr.features import build_extra_by_term, drop_queries_without_signal, expand_bitmask_features, gather_nonexp_bitmask, labels_from_gt, load_npz_as_state, prepare_nonexp_features


@dataclass(frozen=True)
class LTRMember:
    name: str
    path: Path


@dataclass(frozen=True)
class LTRConfig:
    extra_term_features: tuple[str, ...] = ("term_ia",)
    exclude_nonexp_codes: tuple[str, ...] = ()
    n_folds: int = 5
    fold_seed: int = 1001
    rounds: int = 2000
    early_stopping_rounds: int = 50
    xgb_params: dict[str, object] = field(default_factory=lambda: {
        "objective": "rank:ndcg",
        "tree_method": "hist",
        "device": "cuda",
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 1.0,
        "subsample": 0.8,
        "colsample_bynode": 0.8,
        "reg_lambda": 2.0,
        "reg_alpha": 0.0,
        "max_bin": 256,
        "seed": 33,
    })


POSTPROCESS_IN = PostprocessConfig(top_k_q=0.5, min_score=0.0, drop_zero_ia=True, exclude_not_descendants=True, drop_weak_preds=False)
POSTPROCESS_OUT = PostprocessConfig(top_k=500, top_k_q=0.0, min_score=0.0, drop_zero_ia=True, exclude_not_descendants=True, drop_weak_preds=False)


def default_members(dataset: DatasetSpec) -> list[LTRMember]:
    root = dataset.prepared_dir.parent / "predictors"
    return [
        LTRMember("hmlp_esm2", root / "hmlp_esm2"),
        LTRMember("mlp_t5_esm1b", root / "mlp_t5_esm1b"),
        LTRMember("pyb_t5", root / "pyb_t5"),
        LTRMember("blast_knn", root / "blast_knn"),
        LTRMember("naive_prior", root / "naive_prior"),
        LTRMember("nonexp", root / "nonexp"),
    ]


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


def prepare_member_states(members: list[LTRMember], data_type: str, entry_ids: np.ndarray, aspect: str, postprocess: Postprocessor) -> list[CSRState]:
    states = []
    prefix = "oof" if data_type == "oof" else "submit"
    for member in members:
        path = member.path / prefix / f"{prefix}_for_ltr_{aspect}.npz"
        member_ids, state = load_npz_as_state(path)
        state = postprocess.topk_state(state, aspect)
        states.append(align_state_to_protein_axis(entry_ids, member_ids, state))
    return states


class LTRPacker:
    def __init__(self, aspect_gt: dict, member_states: list[CSRState], extra_by_term: np.ndarray | None, nonexp_features) -> None:
        self.gt_indptr = aspect_gt["gt"]["gt_indptr"]
        self.gt_indices = aspect_gt["gt"]["gt_indices"]
        self.n_members = len(member_states)
        indptrs = nb.typed.List()
        indices = nb.typed.List()
        scores = nb.typed.List()
        for state in member_states:
            indptrs.append(state.indptr)
            indices.append(state.indices)
            scores.append(state.scores)
        self.indptrs = indptrs
        self.indices = indices
        self.scores = scores
        self.extra_by_term = extra_by_term
        self.nonexp_features = nonexp_features

    def pack(self, prots_idx: np.ndarray):
        groups, prots_idx, term_indptr, term_indices, x = pack_union_dataset(prots_idx, self.indptrs, self.indices, self.scores)
        if self.extra_by_term is not None:
            x = np.column_stack([x, self.extra_by_term[term_indices]])
        if self.nonexp_features is not None:
            mask = gather_nonexp_bitmask(prots_idx, term_indptr, term_indices, self.nonexp_features)
            x = np.column_stack([x, expand_bitmask_features(mask, self.nonexp_features.bit_positions)])
        return groups, x.astype(np.float32, copy=False), prots_idx, term_indptr, term_indices

    def labels(self, prots_idx: np.ndarray, term_indptr: np.ndarray, term_indices: np.ndarray) -> np.ndarray:
        return labels_from_gt(prots_idx, term_indptr, term_indices, self.gt_indptr, self.gt_indices)


def feature_names(members: list[LTRMember], config: LTRConfig, nonexp_features) -> list[str]:
    names = [member.name for member in members]
    names.extend(config.extra_term_features)
    if nonexp_features is not None:
        names.extend(nonexp_features.feature_names)
    return names


def monotone_unit_arctan(values: np.ndarray) -> np.ndarray:
    out = np.arctan(values).astype(np.float32, copy=False)
    out /= np.float32(np.pi)
    out += np.float32(0.5)
    return out


def concat_states(states: list[CSRState]) -> CSRState:
    indices = np.concatenate([state.indices for state in states]) if states else np.empty(0, dtype=np.int32)
    scores = np.concatenate([state.scores for state in states]) if states else np.empty(0, dtype=np.float32)
    n_rows = sum(int(state.indptr.size) - 1 for state in states)
    indptr = np.empty(n_rows + 1, dtype=np.int64)
    indptr[0] = 0
    q = 0
    nnz = 0
    for state in states:
        rows = int(state.indptr.size) - 1
        indptr[q + 1:q + rows + 1] = state.indptr[1:] + nnz
        q += rows
        nnz += int(state.indptr[-1])
    return CSRState(indptr=indptr, indices=indices, scores=scores)


def train_ltr(dataset: DatasetSpec, members: list[LTRMember] | None = None, out_dir: str | Path | None = None, config: LTRConfig | None = None) -> Path:
    import xgboost as xgb

    config = LTRConfig() if config is None else config
    members = default_members(dataset) if members is None else members
    out_dir = dataset.prepared_dir.parent / "ltr" if out_dir is None else Path(out_dir)
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    train_index = load_index_df(dataset.train_index)
    prepared_gt = load_prepared_gt(dataset.ground_truth)
    post_in = Postprocessor(POSTPROCESS_IN, train_index, prepared_gt)
    post_out = Postprocessor(POSTPROCESS_OUT, train_index, prepared_gt)
    oof_parts = []
    for aspect, aspect_gt in prepared_gt.items():
        entry_ids = aspect_gt["gt"]["gt_ids"]
        axis = np.arange(int(entry_ids.size), dtype=np.int32)
        states = prepare_member_states(members, "oof", entry_ids, aspect, post_in)
        extra = build_extra_by_term(aspect_gt, config.extra_term_features)
        nonexp = prepare_nonexp_features(aspect_gt, entry_ids, config.exclude_nonexp_codes)
        packer = LTRPacker(aspect_gt, states, extra, nonexp)
        names = feature_names(members, config, nonexp)
        k = int(post_in._topk_by_aspect[aspect])
        params = dict(config.xgb_params)
        params["eval_metric"] = f"ndcg@{k}"
        params["lambdarank_num_pair_per_sample"] = k
        fold_prots = []
        fold_states = []
        for fold, (tr, va) in enumerate(_folds(len(axis), config.n_folds, config.fold_seed), start=1):
            tr_groups, tr_x, tr_prots, tr_indptr, tr_terms = packer.pack(axis[tr])
            tr_y = packer.labels(tr_prots, tr_indptr, tr_terms)
            tr_groups, tr_x, tr_prots, tr_indptr, tr_terms, tr_y = drop_queries_without_signal(tr_groups, tr_x, tr_prots, tr_indptr, tr_terms, tr_y, packer.n_members)
            va_groups, va_x, va_prots, va_indptr, va_terms = packer.pack(axis[va])
            va_y = packer.labels(va_prots, va_indptr, va_terms)
            va_groups, va_x, va_prots, va_indptr, va_terms, va_y = drop_queries_without_signal(va_groups, va_x, va_prots, va_indptr, va_terms, va_y, packer.n_members)
            dtr = xgb.DMatrix(tr_x, label=tr_y, missing=np.nan, group=tr_groups, feature_names=names)
            dva = xgb.DMatrix(va_x, label=va_y, missing=np.nan, group=va_groups, feature_names=names)
            booster = xgb.train(params=params, dtrain=dtr, num_boost_round=config.rounds, evals=[(dtr, "train"), (dva, "valid")], callbacks=[xgb.callback.EarlyStopping(rounds=config.early_stopping_rounds, save_best=True)], verbose_eval=50)
            booster.save_model(str(model_dir / f"ltr_{aspect}_fold{fold}.json"))
            scores = monotone_unit_arctan(booster.predict(dva).astype(np.float32, copy=False))
            fold_prots.append(va_prots)
            fold_states.append(CSRState(indptr=va_indptr, indices=va_terms, scores=scores))
        prots = np.concatenate(fold_prots)
        ids = entry_ids[prots]
        state = concat_states(fold_states)
        topk_pos, topk_scores = post_out.postprocess_state(state=state, data_type="oof", entry_ids=ids, aspect_name=aspect, propagate=True, add_nonexp_terms=False, add_exp_terms=False, drop_known=False)
        topk_scores = np.round(topk_scores, 3).astype(np.float32, copy=False)
        part = build_submission_df(ids, topk_pos, topk_scores, aspect_gt)
        part["aspect"] = aspect
        oof_parts.append(part)
    save_submit_parquet(pd.concat(oof_parts, ignore_index=True), out_dir / "oof/oof_ltr.parquet")
    return out_dir


def predict_ltr(dataset: DatasetSpec, members: list[LTRMember] | None = None, model_dir: str | Path | None = None, out_dir: str | Path | None = None, config: LTRConfig | None = None) -> pd.DataFrame:
    import xgboost as xgb

    config = LTRConfig() if config is None else config
    members = default_members(dataset) if members is None else members
    root = dataset.prepared_dir.parent
    model_dir = root / "ltr/models" if model_dir is None else Path(model_dir)
    out_dir = root / "final" if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_index = load_index_df(dataset.test_index)
    test_ids = pd.unique(test_index["EntryID"]).astype(object, copy=False)
    prepared_gt = load_prepared_gt(dataset.ground_truth)
    post_in = Postprocessor(POSTPROCESS_IN, test_index, prepared_gt)
    post_out = Postprocessor(POSTPROCESS_OUT, test_index, prepared_gt)

    parts = []
    for aspect, aspect_gt in prepared_gt.items():
        states = prepare_member_states(members, "submit", test_ids, aspect, post_in)
        extra = build_extra_by_term(aspect_gt, config.extra_term_features)
        nonexp = prepare_nonexp_features(aspect_gt, test_ids, config.exclude_nonexp_codes)
        packer = LTRPacker(aspect_gt, states, extra, nonexp)
        names = feature_names(members, config, nonexp)
        groups, x, prots, term_indptr, term_indices = packer.pack(np.arange(int(test_ids.size), dtype=np.int32))
        dmx = xgb.DMatrix(x, missing=np.nan, group=groups, feature_names=names)

        scores = np.zeros(int(term_indices.size), dtype=np.float32)
        for fold in range(1, config.n_folds + 1):
            booster = xgb.Booster()
            booster.load_model(str(model_dir / f"ltr_{aspect}_fold{fold}.json"))
            scores += booster.predict(dmx).astype(np.float32, copy=False)

        scores = monotone_unit_arctan(scores / np.float32(config.n_folds))
        ids = test_ids[prots]
        state = CSRState(indptr=term_indptr, indices=term_indices, scores=scores)
        topk_pos, topk_scores = post_out.postprocess_state(
            state=state, data_type="test", entry_ids=ids, aspect_name=aspect,
            propagate=True, add_nonexp_terms=False, add_exp_terms=False, drop_known=True,
        )
        part = build_submission_df(ids, topk_pos, topk_scores, aspect_gt)
        part["aspect"] = aspect
        parts.append(part)

    submission = pd.concat(parts, ignore_index=True)
    save_submit_tsv(submission, out_dir / "submission.tsv")
    return submission
