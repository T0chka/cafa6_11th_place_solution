import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from src.data.dataset import DatasetSpec


HIT_COLUMNS = ["qseqid", "sseqid", "bitscore", "evalue"]


def _write_unique_fasta(index_path: Path, out_path: Path) -> None:
    df = pd.read_parquet(index_path, columns=["seq_key", "sequence"])
    df = df.drop_duplicates("seq_key", keep="first")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for seq_key, sequence in df.itertuples(index=False):
            f.write(f">{seq_key}\n{sequence}\n")


def _run_blast(query: Path, db: Path, out_path: Path, threads: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "blastp",
            "-query", str(query),
            "-db", str(db),
            "-max_target_seqs", "500",
            "-evalue", "1e-3",
            "-num_threads", str(threads),
            "-outfmt", "6 qseqid sseqid bitscore evalue",
            "-out", str(out_path),
        ],
        check=True,
    )


def _to_parquet(tsv_path: Path, parquet_path: Path) -> None:
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        header=None,
        names=HIT_COLUMNS,
        dtype={"qseqid": "string", "sseqid": "string"},
    )
    df["bitscore"] = pd.to_numeric(df["bitscore"], errors="raise")
    df["evalue"] = pd.to_numeric(df["evalue"], errors="raise")
    df.to_parquet(parquet_path, index=False)
    tsv_path.unlink()


def ensure_blast_hits(dataset: DatasetSpec, threads: int | None = None) -> tuple[Path, Path]:
    root = dataset.prepared_dir.parent / "blast"
    train_hits = root / "hits_train_vs_train.parquet"
    test_hits = root / "hits_testsuperset.parquet"
    if train_hits.exists() and test_hits.exists():
        return train_hits, test_hits

    for executable in ("makeblastdb", "blastp"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"{executable} is required on PATH")

    db_dir = root / "db"
    train_fasta = db_dir / "train_unique.fasta"
    test_fasta = db_dir / "test_unique.fasta"
    db_prefix = db_dir / "train"
    db_dir.mkdir(parents=True, exist_ok=True)

    if not train_fasta.exists():
        _write_unique_fasta(dataset.train_index, train_fasta)
    if not test_fasta.exists():
        _write_unique_fasta(dataset.test_index, test_fasta)

    if not (db_prefix.with_suffix(".pin").exists() or db_prefix.with_suffix(".pdb").exists()):
        subprocess.run(
            ["makeblastdb", "-in", str(train_fasta), "-dbtype", "prot", "-parse_seqids", "-out", str(db_prefix)],
            check=True,
        )

    n_threads = int(threads or min(16, os.cpu_count() or 1))
    if not train_hits.exists():
        train_tsv = root / "hits_train_vs_train.tsv"
        _run_blast(train_fasta, db_prefix, train_tsv, n_threads)
        _to_parquet(train_tsv, train_hits)
    if not test_hits.exists():
        test_tsv = root / "hits_testsuperset.tsv"
        _run_blast(test_fasta, db_prefix, test_tsv, n_threads)
        _to_parquet(test_tsv, test_hits)

    return train_hits, test_hits
