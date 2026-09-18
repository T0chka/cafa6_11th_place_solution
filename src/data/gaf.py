"""
Normalize a raw UniProt-GOA GAF 2.2 snapshot to the compact Parquet schema used
by src/data/uniprot.py.

Input:
- gzip-compressed GAF 2.2 file.

Output:
- Parquet with columns EntryID, term, aspect, taxon, Qualifier, ECO.

This step changes only storage format. Annotation filtering, GO
canonicalization, evidence-code handling, and train/test restrictions are done
later by src/data/uniprot.py.
"""

from pathlib import Path

import pandas as pd


GAF_COLUMNS = [
    "DB", "EntryID", "DB_Object_Symbol", "Qualifier", "term", "DB_Reference",
    "ECO", "With_From", "aspect", "DB_Object_Name", "DB_Object_Synonym",
    "DB_Object_Type", "taxon", "Date", "Assigned_By", "Annotation_Extension",
    "Gene_Product_Form_ID",
]


def normalize_gaf(gaf_gz: Path, out_parquet: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    rows = 0
    try:
        chunks = pd.read_csv(
            gaf_gz, sep="\t", comment="!", header=None, names=GAF_COLUMNS,
            dtype=str, compression="gzip", chunksize=1_000_000, low_memory=False,
        )
        for chunk in chunks:
            chunk = chunk.loc[:, ["EntryID", "term", "aspect", "taxon", "Qualifier", "ECO"]].copy()
            for col in chunk.columns:
                chunk[col] = chunk[col].fillna("").astype(str)
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_parquet, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(chunk)
            print(f"[gaf] rows: {rows:,}", flush=True)
    finally:
        if writer is not None:
            writer.close()

    if rows == 0:
        raise ValueError(f"No annotation rows were read from {gaf_gz}")
    print(f"[gaf] wrote: {out_parquet}")
