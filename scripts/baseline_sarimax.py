#!/usr/bin/env python3
"""Simple SARIMAX baseline example using statsmodels.
This runs a quick univariate SARIMAX on the target series (first variable).
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
    X = arr['X']
    y = arr['y']
    # simple split
    split = int(0.8 * len(y))
    y_train, y_test = y[:split], y[split:]
    # Fit a small SARIMAX model (p,d,q) = (1,0,1) as starter
    mod = SARIMAX(y_train, order=(1,0,1), seasonal_order=(0,0,0,0), enforce_stationarity=False, enforce_invertibility=False)
    res = mod.fit(disp=False)
    steps = len(y_test)
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
