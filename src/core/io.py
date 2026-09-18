"""
Small I/O helpers used across data preparation and modeling.

Inputs:
- CAFA term TSV files with columns EntryID, term, aspect.
- prepared index Parquet files with columns EntryID, seq_key, length, sequence.

Outputs:
- validated pandas DataFrames.
- aspect codes P/C/F are normalized to BPO/CCO/MFO.
"""

import pandas as pd
from pathlib import Path


ASPECT_CODE_TO_LONG = {"P": "BPO", "C": "CCO", "F": "MFO"}


def normalize_aspect(aspect: str | pd.Series) -> str | pd.Series:
    if isinstance(aspect, pd.Series):
        return aspect.replace(ASPECT_CODE_TO_LONG)
    return ASPECT_CODE_TO_LONG.get(aspect, aspect)


def load_terms_df(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    required = ["EntryID", "term", "aspect"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Terms file is missing columns {missing}: {path}")
    df = df.loc[:, required].copy()
    df["EntryID"] = df["EntryID"].astype(str)
    df["term"] = df["term"].astype(str)
    df["aspect"] = normalize_aspect(df["aspect"])
    return df


def load_index_df(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = ["EntryID", "seq_key", "length", "sequence"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Index file is missing columns {missing}: {path}")
    return df


def load_prepared_gt(path: str | Path) -> dict:
    import pickle

    with open(path, "rb") as handle:
        return pickle.load(handle)


def _load_single_embeddings(
    embed_dir: str | Path,
    keys_df: pd.DataFrame,
    seq_keys,
    label: str = "",
):
    import duckdb
    import numpy as np

    parts = str(Path(embed_dir) / "part-*.parquet")
    con = duckdb.connect(database=":memory:")
    try:
        con.register("keys", keys_df)
        query = (
            "select e.seq_key, e.embedding from read_parquet("
            f"'{parts}') e join keys k using (seq_key)"
        )
        out = con.execute(query).df()
    finally:
        con.close()

    got = set(out["seq_key"].tolist())
    missing = [key for key in seq_keys if key not in got]
    if missing:
        where = f" in {label}" if label else ""
        raise ValueError(f"Missing embeddings{where} for {len(missing)} seq_key.")

    out = out.set_index("seq_key").loc[seq_keys].reset_index()
    return np.vstack(out["embedding"].tolist()).astype(np.float32, copy=False)


def load_embeddings_for_index(
    index_df: pd.DataFrame,
    embed_dirs: list[str | Path] | tuple[str | Path, ...],
    verbose: bool = False,
):
    import numpy as np

    seq_keys = index_df["seq_key"].drop_duplicates().to_numpy()
    keys_df = pd.DataFrame({"seq_key": pd.Series(seq_keys, dtype="string")})
    matrices = []
    for i, embed_dir in enumerate(embed_dirs):
        matrices.append(
            _load_single_embeddings(embed_dir, keys_df, seq_keys, label=str(embed_dir))
        )
    if not matrices:
        raise ValueError("At least one embedding directory is required.")
    x = matrices[0]
    if len(matrices) > 1:
        x = np.concatenate(matrices, axis=1).astype(np.float32, copy=False)
    if verbose:
        print(f"Loaded embeddings: seq_keys={seq_keys.shape}, x={x.shape}")
    return seq_keys.astype("U"), x


def build_submission_df(entry_ids, topk_pos, topk_scores, aspect_gt):
    import numpy as np

    rows, cols = np.nonzero(topk_pos >= 0)
    if rows.size == 0:
        return pd.DataFrame(columns=["EntryID", "term", "score"])
    term_ids = aspect_gt["ontology_term_ids"]
    return pd.DataFrame({
        "EntryID": entry_ids[rows],
        "term": term_ids[topk_pos[rows, cols]],
        "score": topk_scores[rows, cols],
    })


def save_submit_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.loc[:, ["EntryID", "term", "score", "aspect"]].to_parquet(path, index=False)


def save_submit_tsv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.loc[df["score"] > 0, ["EntryID", "term", "score"]]
    out.to_csv(path, sep="\t", header=False, index=False, float_format="%.3g")
