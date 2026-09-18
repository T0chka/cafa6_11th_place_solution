from config import dataset
from src.models.predictor import PredictorSpec, build_predictor


SPEC = PredictorSpec(
    name="hmlp_esm2",
    model="hmlp",
    embeddings=("esm2_t36_3B_UR50D",),
    min_freq={"BPO": 10, "CCO": 0, "MFO": 0},
)


if __name__ == "__main__":
    build_predictor(dataset(), SPEC)
