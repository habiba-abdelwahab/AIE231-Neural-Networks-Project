# Skin lesion classification with neural networks (HAM10000)

## Problem
Given a dermatoscopic photo of a skin lesion, predict which of 7 diagnoses it is
(akiec, bcc, bkl, df, mel, nv, vasc) with a confidence. Missing a melanoma is the most
costly error, so the primary metric is **balanced accuracy** (mean per-class recall) and we
also report **melanoma recall**. Course project, not a diagnostic tool.

Dataset: HAM10000, Tschandl et al. (2018), *Scientific Data* 5:180161, CC BY-NC 4.0. See `data/README.md`.

## Results (validation; test numbers added once, at the end)
| Model | Balanced acc. | Melanoma recall | Macro-F1 |
|---|---|---|---|
| Majority class (nv) | | | |
| Logistic regression, 32×32 pixels | | | |
| Model A: custom CNN | | | |
| Model B: VGG16 frozen | | | |
| Model C: VGG16 fine-tuned (3 seeds) | | | |

## Setup
```bash
pip install -r requirements.txt
python data/download.py          # about 3 GB into data/raw/ham10000/
```
On Colab, just open `notebooks/01_eda.ipynb` with a GPU runtime and run all cells.

## Baselines
```bash
python -m src.train --config configs/baseline_majority.yaml
python -m src.train --config configs/baseline_logreg.yaml
```
Every run appends a row to `experiments/experiments.csv`.

## Layout
`configs/` one YAML per experiment · `data/` download script · `notebooks/` EDA and results ·
`src/` data, models, train, evaluate, predict · `experiments/` run log · `models/` saved model ·
`app/` demo · `report/` PDF and slides
