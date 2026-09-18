# CAFA6 protein function prediction solution

This repository contains the 11th-place solution for the Kaggle [CAFA6 Protein Function Prediction](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction) competition. The original competition write-up is available [here](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/writeups/11th-place-solution). The pipeline combines six complementary protein-function predictors with an XGBoost learning-to-rank ensemble and ontology-aware postprocessing.


## Solution overview


| Component      | Input / signal                                   | Main configuration                                                                                                                                                                                                                                                                                                                                                                                                                          |
| -------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `hmlp_esm2`    | ESM2 3B embeddings (`esm2_t36_3B_UR50D`, 2560)   | Hierarchy-aware MLP; 5 folds; hidden 2048; dropout 0.5; batch 512; 100 epochs; max LR `3e-4`; weight decay 0.25; hierarchy-aware parent loss                                                                                                                                                                                                                                                                                                |
| `mlp_t5_esm1b` | ProtT5 (1024) + ESM1b (1280) embeddings          | MLP with residual blocks; 5 folds; hidden 2048; dropout 0.6; batch 512; 200 epochs; max LR `1e-3`; weight decay 0.2; plateau scheduler                                                                                                                                                                                                                                                                                                      |
| `pyb_t5`       | ProtT5 embeddings (1024)                         | PyBoost; depth 6; learning rate 0.03; colsample 0.8; 5000 trees; `sketch_method="proj"`; `sketch_outputs=1`; prediction batch size 60,000                                                                                                                                                                                                                                                                                                   |
| `blast_knn`    | BLAST similarity to annotated training sequences | Hits with E-value ≤ `1e-3`; self-hits removed and multiple HSPs collapsed to the highest-bitscore hit per sequence pair; top 30 neighbors for BPO/MFO and 50 for CCO                                                                                               |
| `naive_prior`  | Training annotation frequencies                  | number of distinct training proteins annotated with the term / number of annotated training proteins in that aspect                                                                                                                                                                                                             |
| `nonexp`       | Non-experimental UniProt annotations             | Evidence codes `IBA`, `IEA`, `IGC`, `IKR`, `ISA`, `ISM`, `ISO`, `ISS`, `NAS`, and `RCA` |

The final ranker uses the six component scores. Candidate terms are formed from the union of component predictions, ranked with XGBoost, propagated through the GO hierarchy, and reduced to the final top 500 predictions per protein. Protein-level `NOT` annotations are excluded where available.


## Setup

Python 3.12 is required. Dependencies are managed with `uv`.

```bash
uv sync
```

The configured environment pins the GPU-sensitive packages used for this solution, including CuPy, NumPy, PyTorch, PyBoost, and XGBoost. Check NVIDIA driver/CUDA compatibility before installing the configured PyTorch wheel on a new GPU machine.

Competition inputs are downloaded through `kagglehub`. The UniProt-GOA inputs used by data preparation are configured in `config.py` and cached under `artifacts/cache/uniprot/`.

BLAST requires `makeblastdb` and `blastp` on `PATH`. Existing BLAST hit Parquet files are reused automatically.

Generated data, embeddings, model files, caches, and evaluation outputs live under `artifacts/` and are excluded from Git.

## Pipeline

Run the full solution from the repository root:

```bash
uv run python -m scripts.prepare_data
uv run python -m scripts.make_embeddings

uv run python -m scripts.build_hmlp
uv run python -m scripts.build_mlp
uv run python -m scripts.build_pyboost
uv run python -m scripts.build_blast
uv run python -m scripts.build_naive
uv run python -m scripts.build_nonexp

uv run python -m scripts.train_ltr
uv run python -m scripts.predict
```

The first two commands prepare the training/test indices, ontology-aligned annotation structures, UniProt-derived features, and protein embeddings. Each
`build_*` command trains or constructs one base component and writes its OOF and test predictions. `train_ltr` fits the final ranker. `predict` creates only the
final TSV submission written to: `artifacts/final/submission.tsv`


## Evaluation

The original Kaggle submission scored **0.42114** on the private leaderboard. This repository reproduces the same solution pipeline and allows the regenerated submission to be evaluated on the released CAFA6 final evaluation data from [Figshare](https://figshare.com/articles/dataset/_/32861969) using the official [CAFA-evaluator-PK](https://github.com/claradepaolis/CAFA-evaluator-PK). The regenerated `artifacts/final/submission.tsv` achieves a mean `f_micro_w` score of **0.423544**, differing from the Kaggle private-leaderboard score by only **0.002404**. This shows that the repository reproduces the competition solution closely.

The official `f_micro_w` scores on the released CAFA6 final evaluation data are:

| Prediction         | NK BPO | NK CCO | NK MFO | LK BPO | LK CCO | LK MFO | PK BPO | PK CCO | PK MFO | Mean `f_micro_w` |
| ------------------ | -----: | -----: | -----: | -----: | -----: | -----: | -----: | -----: | -----: | ---------------: |
| **LTR ensemble**   |  .3152 |  .5571 |  .7169 |  .4345 |  .5662 |  .6175 |  .0934 |  .2641 |  .2470 |      **.423544** |
| MLP ProtT5 + ESM1b |  .2837 |  .5236 |  .6600 |  .4217 |  .5337 |  .5808 |  .0948 |  .2385 |  .1976 |          .392711 |
| HMLP ESM2          |  .2961 |  .5279 |  .6523 |  .4115 |  .5369 |  .5388 |  .0901 |  .2343 |  .1956 |          .387056 |
| PyBoost ProtT5     |  .2699 |  .4941 |  .5865 |  .3602 |  .4951 |  .5308 |  .0856 |  .2239 |  .1784 |          .358278 |
| BLAST-KNN          |  .2293 |  .4584 |  .6069 |  .2932 |  .4909 |  .5080 |  .0496 |  .2037 |  .1783 |          .335367 |
| Non-experimental   |  .2167 |  .3849 |  .5121 |  .3587 |  .3856 |  .5003 |  .0629 |  .2040 |  .1887 |          .312656 |
| Naive prior        |  .0896 |  .2494 |  .0938 |  .0711 |  .2336 |  .0757 |  .0255 |  .2165 |  .0307 |          .120656 |


## Fast local evaluator

The repository also includes a substantially faster local implementation of the CAFA evaluation procedure. For the regenerated LTR submission, the fast local evaluator gives a mean f_micro_w score of 0.423435, compared with 0.423544 from the official evaluator, a difference of only 0.000109.

It was validated on all seven predictions used in the solution: the six individual components and the final LTR ensemble. Scoring all seven predictions took **83.9 s** with the fast evaluator, compared with **4590.9 s** for the recorded official evaluation run, corresponding to a **54.7× speedup**.


`scripts.prepare_evaluation` prepares the released CAFA6 evaluation data after the release files have been placed under `artifacts/evaluation/reference/raw/32861969/release_kaggle_final_May2026/`.

Export all six base-component predictions plus the final LTR submission and prepare the released evaluation data:

```bash
uv run python -m scripts.export_components
uv run python -m scripts.prepare_evaluation
```

The official `CAFA-evaluator-PK` must then be run separately for the exported predictions and the NK, LK, and PK regimes. Its `evaluation_best_f_micro_w.tsv` outputs must be available at:

```text
artifacts/evaluation/results/NK/evaluation_best_f_micro_w.tsv
artifacts/evaluation/results/LK/evaluation_best_f_micro_w.tsv
artifacts/evaluation/results/PK/evaluation_best_f_micro_w.tsv
```

Once these official results are present, run:

```bash
uv run python -m scripts.evaluate
```

`scripts.evaluate` scores the same exported predictions with the fast local evaluator, selects the best threshold by `f_micro`, compares the results with the official `f_micro_w` outputs, and writes the comparison to `artifacts/evaluation/official_vs_fast_f_micro.tsv`.


A validation run on the seven exported predictions produced:

| Prediction         | Official `f_micro_w` | Fast `f_micro_w` | Difference |
| ------------------ | -------------------: | -------------: | ---------: |
| LTR                |              .423544 |        .423435 |   -.000109 |
| MLP ProtT5 + ESM1b |              .392711 |        .392669 |   -.000042 |
| HMLP ESM2          |              .387056 |        .386786 |   -.000270 |
| PyBoost ProtT5     |              .358278 |        .358266 |   -.000012 |
| BLAST-KNN          |              .335367 |        .335270 |   -.000097 |
| Non-experimental   |              .312656 |        .312658 |   +.000002 |
| Naive prior        |              .120656 |        .120660 |   +.000004 |

Across all 63 regime-by-aspect cells, 54 matched exactly at four decimal places. The mean absolute `f_micro` difference was **0.000095**, the maximum absolute difference was **0.002318**, and the maximum threshold difference was **0.025**.

The local scorer intentionally preserves the official float64 threshold grid while model scores remain float32. This matters for tied scores that lie exactly on nominal thresholds.

## Repository layout

```text
config.py                 dataset and artifact configuration
scripts/                  executable pipeline stages
src/data/                 CAFA/UniProt preparation
src/embeddings/           embedding generation
src/models/               six base predictors
src/candidates/           candidate union construction
src/ltr/                  ranker features and XGBoost LTR
src/core/                 ontology, CSR, postprocessing, scoring and I/O
```
