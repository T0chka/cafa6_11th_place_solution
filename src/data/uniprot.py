"""
Update competition annotations from a UniProt-GOA snapshot.

Inputs from DatasetSpec:
- original competition train terms (EntryID, term, aspect);
- competition test FASTA, used to define the test EntryID universe;
- GO ontology OBO snapshot;
- normalized UniProt-GOA Parquet with columns:
  EntryID, term, aspect, taxon, Qualifier, ECO.

Outputs in DatasetSpec.prepared_dir:
1) uniprot_not_qualifier.parquet
   Unique canonical (EntryID, term, aspect, taxon) annotations for which at
   least one UniProt source row has qualifier NOT and an experimental evidence
   code.

2) uniprot_terms_test_prots.parquet
   Unique canonical annotations for test proteins. A pair is excluded when any
   source row for the same (EntryID, term, aspect, taxon) has qualifier NOT.
   is_exp_confirmed is 1 when at least one source row has an experimental
   evidence code, otherwise 0.

3) train_terms_uni_updated.tsv
   Updated training labels. For every EntryID with at least one eligible
   experimental UniProt annotation, all original competition labels for that
   EntryID are replaced by the experimental UniProt subset. In
   train_test_only mode, replacement/addition is limited to EntryIDs already
   present in the original train set or competition test set.

4) new_train_entry_ids.tsv
   EntryIDs with eligible experimental UniProt annotations that are absent from
   both the original train and test universes. This file is written in both
   update modes; all_new_prots may subsequently add these proteins to training
   once their sequences are provided.

5) uniprot_nonexp_ohe.parquet
   Unique (EntryID, term, aspect, taxon) annotations with one-hot columns for
   non-experimental evidence codes. Experimental codes and ND are not encoded.
   A pair is excluded when any source row for that same pair has qualifier NOT.
   In train_test_only mode only original train/test EntryIDs are retained.

The update semantics:
- NOT is pair-forbidding for test annotations and non-experimental evidence;
- updated experimental train labels are selected row-wise as EXP/non-NOT and
  are not additionally pair-forbidden by a separate NOT source row;
- ND is excluded only from the non-experimental evidence-code component, not
  from test annotations or experimental training-label replacement.
"""

from collections.abc import Iterator
from pathlib import Path

import duckdb
import pandas as pd

from src.core.io import load_terms_df
from src.core.ontology import parse_obo_snapshot
from src.data.dataset import DatasetSpec


ASPECT_TO_NS = {
    "BPO": "biological_process",
    "CCO": "cellular_component",
    "MFO": "molecular_function",
}
EXP_CODES = (
    "EXP", "IDA", "IPI", "IMP", "IGI", "IEP",
    "HTP", "HDA", "HMP", "HGI", "HEP", "TAS", "IC",
)
NOT_RE = r"(^|\|)NOT(\||$)"
DUCKDB_THREADS = 8


def _sql_str(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _parse_entry_id(header: str) -> str:
    parts = header.strip().split("|")
    if len(parts) >= 2 and parts[1]:
        return parts[1]
    first = header.strip().split()[0]
    if first:
        return first
    raise ValueError(f"Unexpected FASTA header: {header!r}")


def _iter_fasta_ids(path: Path) -> Iterator[str]:
    with path.open() as handle:
        for line in handle:
            if line.startswith(">"):
                yield _parse_entry_id(line[1:].strip())


def update_uniprot_annotations(spec: DatasetSpec) -> None:
    update_mode = str(spec.update_mode).strip().lower()
    if update_mode not in {"train_test_only", "all_new_prots"}:
        raise ValueError("update_mode must be 'train_test_only' or 'all_new_prots'.")
    if not spec.uniprot_gaf_parquet.exists():
        raise FileNotFoundError(f"UniProt-GOA parquet not found: {spec.uniprot_gaf_parquet}")

    spec.prepared_dir.mkdir(parents=True, exist_ok=True)
    terms_df = load_terms_df(spec.train_terms)
    test_ids_df = pd.DataFrame({"EntryID": pd.Index(list(_iter_fasta_ids(spec.test_fasta)), dtype="object").unique()})
    train_ids_df = pd.DataFrame({"EntryID": terms_df["EntryID"].astype(str).unique()})

    namespaces = list(ASPECT_TO_NS.values())
    snapshot = parse_obo_snapshot(str(spec.ontology_obo), namespaces)
    valid_go_ids = set()
    for namespace in namespaces:
        valid_go_ids.update(snapshot.edges_by_ns[namespace].keys())

    valid_terms_df = pd.DataFrame({"term": list(valid_go_ids)})
    alt_map_df = pd.DataFrame(
        {"alt": list(snapshot.alt_to_canon.keys()), "canon": list(snapshot.alt_to_canon.values())}
    )

    con = duckdb.connect(database=":memory:")
    try:
        con.execute(f"PRAGMA threads={int(DUCKDB_THREADS)}")
        con.execute("PRAGMA preserve_insertion_order=false")
        con.register("valid_terms_df", valid_terms_df)
        con.register("alt_map_df", alt_map_df)
        con.register("test_ids_df", test_ids_df)
        con.register("train_ids_df", train_ids_df)
        con.register("old_terms_df", terms_df)
        con.execute("CREATE TEMP TABLE valid_terms AS SELECT * FROM valid_terms_df")
        con.execute("CREATE TEMP TABLE alt_map AS SELECT * FROM alt_map_df")
        con.execute("CREATE TEMP TABLE test_ids AS SELECT * FROM test_ids_df")
        con.execute("CREATE TEMP TABLE train_ids AS SELECT * FROM train_ids_df")
        con.execute("CREATE TEMP TABLE old_terms AS SELECT * FROM old_terms_df")

        if update_mode == "train_test_only":
            con.execute(
                "CREATE TEMP TABLE allowed_ids AS "
                "SELECT EntryID FROM train_ids UNION SELECT EntryID FROM test_ids"
            )
            allowed_filter_sql = "AND EntryID IN (SELECT EntryID FROM allowed_ids)"
            allowed_u_filter_sql = "AND u.EntryID IN (SELECT EntryID FROM allowed_ids)"
        else:
            allowed_filter_sql = ""
            allowed_u_filter_sql = ""

        exp_list = ", ".join(_sql_str(code) for code in EXP_CODES)
        uniprot_path_sql = _sql_str(str(spec.uniprot_gaf_parquet))
        base_select_sql = f"""
            SELECT
                u.EntryID AS EntryID,
                COALESCE(m.canon, u.term) AS term,
                CASE u.aspect
                    WHEN 'P' THEN 'BPO'
                    WHEN 'F' THEN 'MFO'
                    WHEN 'C' THEN 'CCO'
                    ELSE u.aspect
                END AS aspect,
                u.taxon AS taxon,
                COALESCE(u.Qualifier, '') AS Qualifier,
                u.ECO AS ECO
            FROM read_parquet({uniprot_path_sql}) AS u
            LEFT JOIN alt_map AS m ON u.term = m.alt
            WHERE COALESCE(m.canon, u.term) IN (SELECT term FROM valid_terms)
        """

        con.execute(
            f"""
            COPY (
                WITH base AS ({base_select_sql})
                SELECT EntryID, term, aspect, taxon
                FROM base
                WHERE regexp_matches(Qualifier, '{NOT_RE}')
                  AND ECO IN ({exp_list})
                GROUP BY EntryID, term, aspect, taxon
            ) TO $1 (FORMAT 'parquet')
            """,
            [str(spec.not_terms_uniprot)],
        )

        con.execute(
            f"""
            COPY (
                WITH base AS (
                    {base_select_sql}
                    AND u.EntryID IN (SELECT EntryID FROM test_ids)
                )
                SELECT
                    EntryID, term, aspect, taxon,
                    MAX(CASE WHEN ECO IN ({exp_list}) THEN 1 ELSE 0 END) AS is_exp_confirmed
                FROM base
                GROUP BY EntryID, term, aspect, taxon
                HAVING MAX(CASE WHEN regexp_matches(Qualifier, '{NOT_RE}') THEN 1 ELSE 0 END) = 0
            ) TO $1 (FORMAT 'parquet')
            """,
            [str(spec.test_terms_uniprot)],
        )

        con.execute(
            f"""
            COPY (
                WITH exp_terms AS (
                    SELECT DISTINCT EntryID
                    FROM ({base_select_sql})
                    WHERE ECO IN ({exp_list})
                      AND NOT regexp_matches(Qualifier, '{NOT_RE}')
                )
                SELECT e.EntryID
                FROM exp_terms AS e
                LEFT JOIN train_ids AS tr ON e.EntryID = tr.EntryID
                LEFT JOIN test_ids AS te ON e.EntryID = te.EntryID
                WHERE tr.EntryID IS NULL AND te.EntryID IS NULL
                ORDER BY e.EntryID
            ) TO $1 (DELIMITER '\t', HEADER TRUE)
            """,
            [str(spec.new_train_ids)],
        )

        con.execute(
            f"""
            COPY (
                WITH exp_terms_all AS (
                    SELECT DISTINCT EntryID, term, aspect
                    FROM ({base_select_sql})
                    WHERE ECO IN ({exp_list})
                      AND NOT regexp_matches(Qualifier, '{NOT_RE}')
                ),
                new_terms_all AS (
                    SELECT EntryID, term, aspect
                    FROM exp_terms_all
                    WHERE 1 = 1
                    {allowed_filter_sql}
                ),
                replaced_ids AS (
                    SELECT DISTINCT EntryID FROM new_terms_all
                )
                SELECT EntryID, term, aspect
                FROM old_terms
                WHERE EntryID NOT IN (SELECT EntryID FROM replaced_ids)
                UNION ALL
                SELECT EntryID, term, aspect
                FROM new_terms_all
            ) TO $1 (DELIMITER '\t', HEADER TRUE)
            """,
            [str(spec.updated_train_terms)],
        )

        nonexp_codes_rows = con.execute(
            f"""
            WITH base AS (
                {base_select_sql}
                {allowed_u_filter_sql}
            )
            SELECT DISTINCT ECO
            FROM base
            WHERE ECO IS NOT NULL AND ECO <> ''
              AND ECO <> 'ND'
              AND ECO NOT IN ({exp_list})
            ORDER BY ECO
            """
        ).fetchall()
        nonexp_codes = [row[0] for row in nonexp_codes_rows]
        if not nonexp_codes:
            raise ValueError("No non-experimental ECO codes found after filtering.")

        column_specs = [(str(code), f"nonexp_{str(code)}") for code in nonexp_codes]
        nonexp_cols_sql = ",\n".join(
            "MAX(CASE WHEN ECO = " + _sql_str(code) + " THEN 1 ELSE 0 END) AS " + f'"{col}"'
            for code, col in column_specs
        )
        output_cols_sql = ", ".join(f'"{col}"' for _, col in column_specs)

        con.execute(
            f"""
            COPY (
                WITH base AS (
                    {base_select_sql}
                    {allowed_u_filter_sql}
                ),
                agg AS (
                    SELECT
                        EntryID, term, aspect, taxon,
                        MAX(CASE WHEN regexp_matches(Qualifier, '{NOT_RE}') THEN 1 ELSE 0 END) AS has_not,
                        {nonexp_cols_sql},
                        MAX(CASE
                            WHEN ECO IS NOT NULL AND ECO <> ''
                             AND ECO <> 'ND'
                             AND ECO NOT IN ({exp_list})
                            THEN 1 ELSE 0 END) AS has_any_nonexp
                    FROM base
                    GROUP BY EntryID, term, aspect, taxon
                )
                SELECT EntryID, term, aspect, taxon, {output_cols_sql}
                FROM agg
                WHERE has_not = 0 AND has_any_nonexp = 1
            ) TO $1 (FORMAT 'parquet')
            """,
            [str(spec.nonexp_codes_uniprot)],
        )
    finally:
        con.close()

    print(f"[uniprot] wrote: {spec.not_terms_uniprot}")
    print(f"[uniprot] wrote: {spec.test_terms_uniprot}")
    print(f"[uniprot] wrote: {spec.updated_train_terms}")
    print(f"[uniprot] wrote: {spec.new_train_ids}")
    print(f"[uniprot] wrote: {spec.nonexp_codes_uniprot}")
