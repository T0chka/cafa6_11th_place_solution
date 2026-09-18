from config import dataset
from src.models.naive_prior import build_naive_component


if __name__ == "__main__":
    build_naive_component(dataset())
