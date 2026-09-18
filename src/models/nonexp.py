from pathlib import Path

import numpy as np
import pandas as pd

from src.candidates.union import align_state_to_protein_axis
from src.core.io import load_index_df, load_prepared_gt
from src.core.postprocess import CSRState, PostprocessConfig, Postprocessor
from src.data.dataset import DatasetSpec
from src.ltr.features import term_ia


POSTPROCESS = PostprocessConfig(top_k=500, min_score=-np.inf, drop_zero_ia=True, exclude_not_descendants=True, drop_weak_preds=False)


def _state_for_axis(aspect_gt: dict, entry_ids: np.ndarray) -> CSRState:
    nonexp = aspect_gt["nonexp_codes"]
    weights = term_ia(aspect_gt)[:, 0]
    state = CSRState(indptr=nonexp["nonexp_indptr"], indices=nonexp["nonexp_indices"], scores=weights[nonexp["nonexp_indices"]])
    return align_state_to_protein_axis(entry_ids, nonexp["nonexp_ids"], state)


def _save(index_df: pd.DataFrame, prepared_gt: dict, data_type: str, out_dir: Path) -> None:
    entry_ids = pd.unique(index_df["EntryID"]).astype(object, copy=False)
    postprocess = Postprocessor(POSTPROCESS, index_df, prepared_gt)
    prefix = "oof" if data_type == "oof" else "submit"
    target = out_dir / prefix
    target.mkdir(parents=True, exist_ok=True)
    for aspect, aspect_gt in prepared_gt.items():
        state = _state_for_axis(aspect_gt, entry_ids)
        topk_pos, _ = postprocess.postprocess_state(state=state, data_type=data_type, entry_ids=entry_ids, aspect_name=aspect, propagate=False, add_nonexp_terms=False, add_exp_terms=False, drop_known=(data_type == "test"))
        scores = np.full(topk_pos.shape, -np.inf, dtype=np.float32)
        scores[topk_pos != -1] = np.float32(1.0)
        np.savez_compressed(target / f"{prefix}_for_ltr_{aspect}.npz", entry_ids=entry_ids, term_pos=topk_pos.astype(np.int32, copy=False), scores=scores)


def build_nonexp_component(dataset: DatasetSpec, out_dir: str | Path | None = None) -> Path:
    if not dataset.use_uniprot:
        raise ValueError("nonexp component requires UniProt evidence data")
    out_dir = dataset.prepared_dir.parent / "predictors/nonexp" if out_dir is None else Path(out_dir)
    prepared_gt = load_prepared_gt(dataset.ground_truth)
    _save(load_index_df(dataset.train_index), prepared_gt, "oof", out_dir)
    _save(load_index_df(dataset.test_index), prepared_gt, "test", out_dir)
    return out_dir
