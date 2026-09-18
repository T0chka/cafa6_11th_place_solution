from config import dataset
from src.models.blast_knn import build_blast_component
from src.models.blast_search import ensure_blast_hits


if __name__ == "__main__":
    ds = dataset()
    train_hits, test_hits = ensure_blast_hits(ds)
    build_blast_component(ds, train_hits_path=train_hits, test_hits_path=test_hits)
