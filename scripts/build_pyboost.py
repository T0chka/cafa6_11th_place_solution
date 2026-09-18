from config import dataset
from src.models.predictor import PredictorSpec, build_predictor


SPEC = PredictorSpec(
    name="pyb_t5",
    model="pyboost",
    embeddings=("prot_t5",),
    min_freq={"BPO": 10, "CCO": 0, "MFO": 0},
)


if __name__ == "__main__":
    build_predictor(dataset(), SPEC)
