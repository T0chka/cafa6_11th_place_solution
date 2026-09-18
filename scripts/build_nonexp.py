from config import dataset
from src.models.nonexp import build_nonexp_component


if __name__ == "__main__":
    build_nonexp_component(dataset())
