from config import dataset
from src.ltr.ranker import train_ltr


if __name__ == "__main__":
    train_ltr(dataset())
