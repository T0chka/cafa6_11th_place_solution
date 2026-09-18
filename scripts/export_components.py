import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from config import dataset
from src.candidates.union import align_state_to_protein_axis
from src.core.io import build_submission_df, load_index_df, load_prepared_gt, save_submit_tsv
from src.core.postprocess import Postprocessor
from src.ltr.features import load_npz_as_state
from src.ltr.ranker import POSTPROCESS_OUT, default_members

ds = dataset()
test_index = load_index_df(ds.test_index)
test_ids = pd.unique(test_index["EntryID"]).astype(object, copy=False)
gt = load_prepared_gt(ds.ground_truth)
post = Postprocessor(POSTPROCESS_OUT, test_index, gt)
out_root = Path("artifacts/evaluation/predictions")
out_root.mkdir(parents=True, exist_ok=True)

for member in default_members(ds):
    parts = []
    for aspect, aspect_gt in gt.items():
        path = member.path / "submit" / f"submit_for_ltr_{aspect}.npz"
        member_ids, state = load_npz_as_state(path)
        state = align_state_to_protein_axis(test_ids, member_ids, state)
        pos, scores = post.postprocess_state(
            state=state,
            data_type="test",
            entry_ids=test_ids,
            aspect_name=aspect,
            propagate=True,
            add_nonexp_terms=False,
            add_exp_terms=False,
            drop_known=True,
        )
        part = build_submission_df(test_ids, pos, scores, aspect_gt)
        part["aspect"] = aspect
        parts.append(part)

    df = pd.concat(parts, ignore_index=True)
    path = out_root / f"{member.name}.tsv"
    save_submit_tsv(df, path)
    print(member.name, len(df), path)
final_path = Path("artifacts/final/submission.tsv")
if not final_path.exists():
    raise FileNotFoundError(f"Final LTR submission not found: {final_path}")
ltr_path = out_root / "ltr.tsv"
shutil.copyfile(final_path, ltr_path)
print("ltr", ltr_path)
