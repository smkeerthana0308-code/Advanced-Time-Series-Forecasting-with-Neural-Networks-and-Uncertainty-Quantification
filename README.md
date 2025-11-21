# Advanced Time Series Forecasting Project (Submission-ready)

This repository provides a ready-to-run starter project for the Cultus assignment:
**Advanced Time Series Forecasting with Neural Networks and Uncertainty Quantification**.

## Contents
- `data/` : place to store or generate datasets
- `scripts/generate_data.py` : synthetic multivariate time series generator (>=5 vars, 1500+ obs)
- `scripts/train_gru_mc_dropout.py` : PyTorch GRU model with MC Dropout for probabilistic forecasts
- `scripts/baseline_sarimax.py` : example SARIMAX baseline for a single series
- `scripts/evaluate.py` : evaluation code to compute RMSE, MAE and simple interval metrics
- `requirements.txt` : suggested Python packages
- `report.md` : analytical write-up template for submission

## How to run (local machine)
1. Create a virtual environment and install requirements:
```bash
python -m venv venv
source venv/bin/activate   # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. Generate synthetic data (saves to `data/synthetic_multivar.npz`):
```bash
python scripts/generate_data.py --n_obs 2000 --n_vars 5 --out data/synthetic_multivar.npz
```

3. Train the GRU model (creates checkpoints under `artifacts/`):
```bash
python scripts/train_gru_mc_dropout.py --data data/synthetic_multivar.npz --epochs 10
```

4. Run baseline and evaluation:
```bash
python scripts/baseline_sarimax.py --data data/synthetic_multivar.npz --out artifacts/sarimax_preds.npz
python scripts/evaluate.py --true artifacts/test_targets.npz --pred artifacts/gru_preds.npz --pred_uncert artifacts/gru_uncert.npz
```

## Notes
- The training script uses PyTorch. You can adapt it for TensorFlow if preferred.
- The scripts include explanatory comments and are intentionally clear so you can extend them for the Cultus assessment requirements.


**Status:** Updated to address review feedback: multivariate SARIMAX baseline, MC Dropout samples saved, and evaluation summary table generation.
