from pathlib import Path

import kagglehub

from src.data.dataset import DatasetSpec


COMPETITION = "cafa-6-protein-function-prediction"
ARTIFACTS = Path("artifacts")
UNIPROT_SNAPSHOT = ARTIFACTS / "cache/uniprot/goa_uniprot_2025_12_04.parquet"
UNIPROT_GAF = ARTIFACTS / "cache/uniprot/goa_uniprot_gcrp.gaf.gz"


def dataset() -> DatasetSpec:
    root = Path(kagglehub.competition_download(COMPETITION))
    return DatasetSpec(
        train_fasta=root / "Train/train_sequences.fasta",
        test_fasta=root / "Test/testsuperset.fasta",
        train_terms=root / "Train/train_terms.tsv",
        ontology_obo=root / "Train/go-basic.obo",
        ia_file=root / "IA.tsv",
        prepared_dir=ARTIFACTS / "prepared",
        cache_dir=ARTIFACTS / "cache",
        uniprot_gaf_parquet=UNIPROT_SNAPSHOT,
        uniprot_gaf_gz=UNIPROT_GAF,
        update_mode="train_test_only",
        use_uniprot=True,
    )
