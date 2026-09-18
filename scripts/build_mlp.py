from config import dataset
from src.models.predictor import PredictorSpec, build_predictor


SPEC = PredictorSpec(
    name="mlp_t5_esm1b",
    model="mlp",
    embeddings=("prot_t5", "esm1b_650M"),
    min_freq={"BPO": 10, "CCO": 0, "MFO": 0},
)


if __name__ == "__main__":
    build_predictor(dataset(), SPEC)
