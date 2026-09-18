from config import dataset
from src.data.prepare import prepare_dataset


if __name__ == "__main__":
    prepare_dataset(dataset())
