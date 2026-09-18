"""
Generate sequence embeddings for the prepared dataset.

Inputs:
- DatasetSpec.train_index and DatasetSpec.test_index, each with columns:
  EntryID, seq_key, length, sequence.
- EmbeddingSpec describing the pretrained protein language model.

Output:
artifacts/embeddings/<embedding-name>/part-XXXXX.parquet

Each Parquet row contains:
- seq_key: SHA1 sequence key used by the prepared dataset;
- length: sequence length actually represented after truncation;
- embedding: float32 mean-pooled embedding vector.

Embedding behavior:
- train and test sequences are deduplicated jointly by seq_key;
- sequences are sorted by length before batching;
- sequences are truncated to 1022 residues;
- token batches are limited to approximately 8192 residues/tokens;
- ESM representations are mean-pooled over residue tokens;
- ProtT5 uses Rostlab/prot_t5_xl_uniref50 and the same residue replacement,
  tokenization and pooling logic as the original solution;
- CUDA OOM causes recursive batch splitting;
- output is restartable: seq_keys already present in part-*.parquet are skipped;
- output parts contain at most 4096 rows and use zstd compression.

The ESM1b artifact name used by the original solution was "esm1b_650M".
Its fair-esm checkpoint is esm1b_t33_650M_UR50S.

This module has no dataset-specific paths and no command-line interface.
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from src.data.dataset import DatasetSpec
from src.embeddings.specs import EmbeddingSpec


DEVICE = "cuda"
MAX_TOKENS = 8192
MAX_SEQ_LEN = 1022
ROWS_PER_PARQUET = 4096
PRINT_EVERY_BATCHES = 200


def embedding_dir(dataset: DatasetSpec, spec: EmbeddingSpec) -> Path:
    return dataset.prepared_dir.parent / "embeddings" / spec.name


def _token_batches(df: pd.DataFrame, max_tokens: int):
    lengths = df["length"].to_numpy()
    seqs = df["sequence"].tolist()
    keys = df["seq_key"].tolist()
    i = 0
    while i < len(df):
        total = 0
        j = i
        while j < len(df):
            need = int(lengths[j]) + 2
            if j > i and total + need > max_tokens:
                break
            total += need
            j += 1
        yield keys[i:j], seqs[i:j], lengths[i:j], total
        i = j


def existing_seq_keys(out_dir: Path) -> tuple[set[str], list[Path]]:
    part_paths = sorted(out_dir.glob("part-*.parquet"))
    keys: set[str] = set()
    for path in part_paths:
        table = pq.read_table(path, columns=["seq_key"])
        keys.update(table.column("seq_key").to_pylist())
    return keys, part_paths


def _next_part_index(part_paths: list[Path]) -> int:
    if not part_paths:
        return 0
    last = part_paths[-1].stem.split("-")[-1]
    return int(last) + 1 if last.isdigit() else len(part_paths)


def _write_parquet(rows: list[tuple[str, int, np.ndarray]], out_dir: Path, part_idx: int) -> Path:
    keys, lengths, embeddings = zip(*rows)
    table = pa.table(
        {
            "seq_key": pa.array(keys),
            "length": pa.array(lengths, type=pa.int32()),
            "embedding": pa.array(
                [x.astype("float32").tolist() for x in embeddings],
                type=pa.list_(pa.float32()),
            ),
        }
    )
    out_path = out_dir / f"part-{part_idx:05d}.parquet"
    pq.write_table(table, out_path, compression="zstd")
    return out_path


def _load_model(spec: EmbeddingSpec, device: str):
    if spec.backend == "prott5":
        from transformers import AutoTokenizer, T5EncoderModel

        tokenizer = AutoTokenizer.from_pretrained(spec.model_id, use_fast=False)
        model = T5EncoderModel.from_pretrained(spec.model_id).to(device).eval()
        return model, tokenizer, None

    import esm

    loader = getattr(esm.pretrained, spec.model_id)
    model, alphabet = loader()
    model = model.eval().to(device)
    return model, alphabet.get_batch_converter(), model.num_layers


def _run_batch(
    keys: list[str],
    seqs: list[str],
    lengths,
    spec: EmbeddingSpec,
    model,
    helper,
    layer,
    device: str,
) -> list[tuple[str, int, np.ndarray]]:
    seqs = [seq[:MAX_SEQ_LEN] for seq in seqs]
    lengths = [min(int(length), MAX_SEQ_LEN) for length in lengths]
    rows: list[tuple[str, int, np.ndarray]] = []

    if spec.backend == "prott5":
        tokenizer = helper
        seqs = [
            seq.translate(str.maketrans({"U": "X", "Z": "X", "O": "X", "B": "X"}))
            for seq in seqs
        ]
        spaced = [" ".join(list(seq)) for seq in seqs]
        encoded = tokenizer(
            spaced,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN + 2,
            return_tensors="pt",
            return_attention_mask=True,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        hidden_states = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"]
        for i, key in enumerate(keys):
            valid = mask[i].bool()
            if valid.sum().item() >= 2:
                valid[0] = False
                valid[valid.nonzero()[-1].item()] = False
            emb = hidden_states[i][valid].mean(0).cpu().numpy()
            rows.append((key, int(lengths[i]), emb))
        return rows

    batch = [(key, seq) for key, seq in zip(keys, seqs)]
    _, _, tokens = helper(batch)
    tokens = tokens.to(device)
    reps = model(tokens, repr_layers=[layer])["representations"][layer]
    for i, length in enumerate(lengths):
        emb = reps[i, 1 : length + 1].mean(0).cpu().numpy()
        rows.append((keys[i], int(length), emb))
    return rows


def _run_with_split(
    keys,
    seqs,
    lengths,
    spec: EmbeddingSpec,
    model,
    helper,
    layer,
    device: str,
) -> list[tuple[str, int, np.ndarray]]:
    stack = [(keys, seqs, list(lengths))]
    rows: list[tuple[str, int, np.ndarray]] = []
    while stack:
        keys_i, seqs_i, lengths_i = stack.pop()
        try:
            rows.extend(
                _run_batch(
                    keys_i, seqs_i, lengths_i, spec, model, helper, layer, device
                )
            )
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(keys_i) == 1:
                raise
            mid = len(keys_i) // 2
            stack.append((keys_i[mid:], seqs_i[mid:], lengths_i[mid:]))
            stack.append((keys_i[:mid], seqs_i[:mid], lengths_i[:mid]))
    return rows


def generate_embeddings(
    dataset: DatasetSpec,
    spec: EmbeddingSpec,
    device: str = DEVICE,
) -> None:
    out_dir = embedding_dir(dataset, spec)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(dataset.train_index)
    test_df = pd.read_parquet(dataset.test_index)
    combined = pd.concat(
        [
            train_df[["seq_key", "sequence", "length"]],
            test_df[["seq_key", "sequence", "length"]],
        ],
        ignore_index=True,
    )
    unique_df = (
        combined.drop_duplicates(subset=["seq_key"], keep="first")
        .sort_values("length")
        .reset_index(drop=True)
    )

    total_unique = len(unique_df)
    done_keys, part_paths = existing_seq_keys(out_dir)
    part = _next_part_index(part_paths)
    if done_keys:
        unique_df = unique_df.loc[~unique_df["seq_key"].isin(done_keys)].copy()

    remaining = len(unique_df)
    print(f"[embed] model: {spec.name}")
    print(f"[embed] out_dir: {out_dir}")
    print(f"[embed] unique total: {total_unique}")
    print(f"[embed] already computed: {len(done_keys)}")
    print(f"[embed] remaining: {remaining}")
    if remaining == 0:
        print("[embed] nothing to do")
        return

    print(f"[embed] loading: {spec.model_id}")
    model, helper, layer = _load_model(spec, device)

    buffer: list[tuple[str, int, np.ndarray]] = []
    processed = 0
    batches = 0
    t0 = time.time()
    last_print = t0

    with torch.inference_mode():
        for keys, seqs, lengths, tokens in _token_batches(unique_df, MAX_TOKENS):
            rows = _run_with_split(
                keys, seqs, lengths, spec, model, helper, layer, device
            )
            buffer.extend(rows)
            processed += len(keys)
            batches += 1

            while len(buffer) >= ROWS_PER_PARQUET:
                out_path = _write_parquet(
                    buffer[:ROWS_PER_PARQUET], out_dir, part
                )
                buffer = buffer[ROWS_PER_PARQUET:]
                part += 1
                print(f"[embed] wrote {ROWS_PER_PARQUET} rows -> {out_path}")

            now = time.time()
            if batches % PRINT_EVERY_BATCHES == 0 or now - last_print >= 30:
                rate = processed / max(now - t0, 1e-9)
                pct = 100.0 * processed / remaining
                print(
                    f"[embed] {processed}/{remaining} ({pct:.2f}%), "
                    f"batches={batches}, tokens~{tokens}, seq/s={rate:.1f}"
                )
                last_print = now

    if buffer:
        out_path = _write_parquet(buffer, out_dir, part)
        print(f"[embed] wrote {len(buffer)} rows -> {out_path}")

    elapsed = time.time() - t0
    print(f"[embed] done: computed {processed} embeddings in {elapsed:.1f}s")

    required = set(train_df["seq_key"].unique()) | set(test_df["seq_key"].unique())
    embedded, _ = existing_seq_keys(out_dir)
    missing = required - embedded
    print(f"[verify] unique seq_keys from indices: {len(required)}")
    print(f"[verify] embedded seq_keys: {len(embedded)}")
    if missing:
        raise RuntimeError(f"{len(missing)} required seq_keys are missing embeddings.")
    print("[verify] all seq_keys from indices have embeddings")


def embed_sequences(
    sequences: pd.DataFrame,
    spec: EmbeddingSpec,
    device: str = DEVICE,
) -> pd.DataFrame:
    required = {"seq_key", "sequence", "length"}
    missing = required - set(sequences.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    model, helper, layer = _load_model(spec, device)
    rows: list[tuple[str, int, np.ndarray]] = []
    ordered = sequences.loc[:, ["seq_key", "sequence", "length"]].reset_index(drop=True)

    with torch.inference_mode():
        for keys, seqs, lengths, _ in _token_batches(ordered, MAX_TOKENS):
            rows.extend(
                _run_with_split(
                    keys, seqs, lengths, spec, model, helper, layer, device
                )
            )

    return pd.DataFrame(rows, columns=["seq_key", "length", "embedding"])
