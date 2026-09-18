"""
Embedding model specifications used by the modeling pipeline.

This module contains model identity only. Dataset selection, input/output paths
and execution are handled elsewhere.

The three specifications below are the embeddings required by the final
base predictors:
- ESM2_3B: ESM-2 3B, 2560-dimensional mean-pooled representation;
- PROT_T5: ProtT5-XL-UniRef50, 1024-dimensional mean-pooled representation;
- ESM1B_650M: ESM-1b 650M, 1280-dimensional mean-pooled representation.

Artifact names are stable so existing local embeddings can be reused without conversion.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class EmbeddingSpec:
    name: str
    backend: Literal["esm", "prott5"]
    model_id: str
    dimension: int


ESM2_3B = EmbeddingSpec(
    name="esm2_t36_3B_UR50D",
    backend="esm",
    model_id="esm2_t36_3B_UR50D",
    dimension=2560,
)

PROT_T5 = EmbeddingSpec(
    name="prot_t5",
    backend="prott5",
    model_id="Rostlab/prot_t5_xl_uniref50",
    dimension=1024,
)

ESM1B_650M = EmbeddingSpec(
    name="esm1b_650M",
    backend="esm",
    model_id="esm1b_t33_650M_UR50S",
    dimension=1280,
)
