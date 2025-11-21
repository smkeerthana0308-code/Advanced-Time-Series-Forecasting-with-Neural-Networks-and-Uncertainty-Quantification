#!/usr/bin/env python3
"""SARIMAX baseline using exogenous variables for a multivariate baseline.
Fits SARIMAX on the target series with remaining variables as exog.
Saves 90% prediction intervals and point forecasts.
"""
import argparse
import numpy as np
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--out', type=str, default='artifacts/sarimax_preds.npz')
    args = parser.parse_args()
    arr = np.load(args.data)
    X = arr['X']  # shape (n_obs, n_vars)
    y = arr['y']
    # use all other variables as exogenous for SARIMAX
    if X.shape[1] > 1:
        exog = X[:, 1:]
    else:
        exog = None
    # simple split
    split = int(0.8 * len(y))
    y_train, y_test = y[:split], y[split:]
    exog_train = exog[:split] if exog is not None else None
    exog_test = exog[split:] if exog is not None else None

    mod = SARIMAX(y_train, exog=exog_train, order=(1,0,1), seasonal_order=(0,0,0,0),
                  enforce_stationarity=False, enforce_invertibility=False)
    res = mod.fit(disp=False)
    steps = len(y_test)
    if exog_test is not None:
        preds = res.get_forecast(steps=steps, exog=exog_test)
    else:
        preds = res.get_forecast(steps=steps)
    mean = preds.predicted_mean
    conf = preds.conf_int(alpha=0.10)  # 90% interval
    lower = conf.iloc[:,0].values
    upper = conf.iloc[:,1].values
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    np.savez_compressed(args.out, pred_mean=mean, lower=lower, upper=upper, y_test=y_test)
    print('Saved SARIMAX predictions to', args.out)

if __name__ == '__main__':
    main()
