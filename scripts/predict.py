from config import dataset
from src.ltr.ranker import predict_ltr


if __name__ == "__main__":
    predict_ltr(dataset())
