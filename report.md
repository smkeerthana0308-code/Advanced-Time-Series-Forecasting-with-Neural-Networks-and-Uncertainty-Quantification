# Project Report: Advanced Time Series Forecasting with Uncertainty Quantification

## 1. Problem definition
Implement a multivariate forecasting model and quantify predictive uncertainty (e.g., 90% prediction intervals).
Compare the deep learning model against a classical baseline (SARIMAX).

## 2. Dataset
- Synthetic dataset generated with 5 variables and configurable length.
- Exhibits seasonality, trend, and cross-correlations.
- Saved format: `numpy .npz` containing `X` and `y` arrays and time index.

## 3. Model architecture
- Sequence-to-sequence GRU with dropout applied to recurrent and dense layers.
- Monte Carlo Dropout used at inference: multiple stochastically dropped-forward passes to estimate predictive distribution.

## 4. Baseline
- SARIMAX model trained on the target series (univariate baseline).

## 5. Metrics
- Point-forecast metrics: RMSE, MAE.
- Uncertainty metrics: Coverage Probability (fraction of true values inside predicted interval), Mean Interval Width (sharpness).

## 6. Results (to fill after experiments)
- Include tables comparing RMSE/MAE and Coverage/Width for deep model vs baseline.

## 7. Reproducibility
- All scripts provided. Use `requirements.txt` to install dependencies.
- Provide seeds and hyperparameters in the scripts for reproducible runs.



## 11. Aligned results

Aligned SARIMAX predictions to GRU test horizon by taking the last n points to match sequence-based test set.

```json
{
  "sarimax": {
    "rmse": 2.2951109247327675,
    "mae": 1.871800211942368,
    "coverage_90": 0.7316455696202532,
    "mean_width": 5.818530816027351
  },
  "gru_mc": null
}
```

## 6. Results (filled example)
Below is an example results summary table comparing the deep learning GRU (with MC Dropout) and the SARIMAX baseline.
| Model | RMSE | MAE | Coverage (90%) | Mean Interval Width |
|---|---:|---:|---:|---:|
| GRU (MC Dropout) | *fill after run* | *fill after run* | *fill* | *fill* |
| SARIMAX (with exog) | *fill after run* | *fill after run* | *fill* | *fill* |

Include plots:
- Time series test target vs. point forecasts.
- Prediction intervals (90%) from both models.
- Bar chart comparing RMSE/MAE and table of interval metrics.

## 7. Reproducibility
- Environment: Use `requirements.txt`. Example:
```
python==3.10
numpy
pandas
torch
statsmodels
scikit-learn
```
- Random seeds and hyperparameters used for experiments should be recorded. Example command used for final runs:
```
python scripts/generate_data.py --n_obs 2000 --n_vars 5 --seed 42 --out data/synthetic_multivar.npz
python scripts/train_gru_mc_dropout.py --data data/synthetic_multivar.npz --epochs 20 --mc_samples 200 --seq_len 24 --batch_size 64 --seed 42
python scripts/baseline_sarimax.py --data data/synthetic_multivar.npz --out artifacts/sarimax_preds.npz
python scripts/evaluate.py --sarimax artifacts/sarimax_preds.npz --gru artifacts/gru_mc_samples.npz
```
- Artifacts saved in `artifacts/` include model checkpoints, `gru_mc_samples.npz`, and `sarimax_preds.npz` which together allow regenerating the reported tables.

