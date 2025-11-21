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

