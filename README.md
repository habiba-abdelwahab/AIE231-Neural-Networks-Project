# Skin lesion classification with neural networks (HAM10000)

## Problem
Given a dermatoscopic photo of a skin lesion, predict which of 7 diagnoses it is
(akiec, bcc, bkl, df, mel, nv, vasc) with a confidence. Missing a melanoma is the most
costly error, so the primary metric is **balanced accuracy** (mean per-class recall) and we
also report **melanoma recall**. 

Dataset: HAM10000, Tschandl et al. (2018), *Scientific Data* 5:180161, CC BY-NC 4.0. See `data/README.md`.

## Results (validation; test numbers added once, at the end)
| Model | Balanced acc. | Melanoma recall | Macro-F1 |
|---|---|---|---|
| Majority class (nv) | | | |
| Logistic regression, 32×32 pixels | | | |
| Model A: custom CNN | | | |
| Model B: VGG16 frozen | | | |
| Model C: VGG16 fine-tuned (3 seeds) | | | |

## Team
| Member | Role | Owns |
|---|---|---|
| Habiba Osama Mohamed Mohamed | Tech Lead + EDA & baselines | repo structure, configs, `src/train.py`, `src/models.py` (baselines, Model A), `notebooks/00_colab_runner.ipynb`, `notebooks/01_eda.ipynb` |
| Ganna Mohamed | Data Engineer | `data/`, `src/data.py` |
| Mariam Ahmed Mansour Abdullah | Model Architect | `src/config.py`, `src/augment.py`, `src/models.py` (Models B, C), `src/manual_step.py`, `notebooks/03_architecture.ipynb` |
| Mohamed Tamer Ahmed | Experimentation Engineer | `src/experiment_log.py`, optimizer configs, `notebooks/02_results.ipynb` |
| Ahmed Ibrahim Mohamed | Evaluation & Deployment | `src/evaluate.py`, `src/predict.py`, `notebooks/04_evaluation.ipynb`, `app/` |

## Setup
```bash
pip install -r requirements.txt
python data/download.py          # about 3 GB into data/raw/ham10000/
```
On Colab, just open `notebooks/01_eda.ipynb` with a GPU runtime and run all cells.

## Train
```bash
python -m src.train --config configs/baseline_majority.yaml --author <github-login>
python -m src.train --config configs/baseline_logreg.yaml --author <github-login>
python -m src.train --config configs/model_a_cnn.yaml --author <github-login>
python -m src.train --config configs/model_b_vgg16_frozen.yaml --author <github-login>
python -m src.train --config configs/model_c_vgg16_finetune.yaml --author <github-login>
```
On Colab use `notebooks/00_colab_runner.ipynb`. Every run appends a row to
`experiments/runs_<github-login>.csv` and saves its curves in `experiments/curves/`.

## Layout
`configs/` one YAML per experiment · `data/` download script · `notebooks/` EDA and results ·
`src/` data, models, train, evaluate, predict · `experiments/` run log · `models/` saved model ·
`app/` demo · `report/` PDF and slides
