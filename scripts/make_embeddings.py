from config import dataset
from src.embeddings.generate import generate_embeddings
from src.embeddings.specs import ESM1B_650M, ESM2_3B, PROT_T5


if __name__ == "__main__":
    ds = dataset()
    for spec in (ESM2_3B, PROT_T5, ESM1B_650M):
        generate_embeddings(ds, spec)
