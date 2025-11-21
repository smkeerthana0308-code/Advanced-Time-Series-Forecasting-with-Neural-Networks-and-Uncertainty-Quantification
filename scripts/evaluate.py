#!/usr/bin/env python3
"""Simple evaluation utilities for predictions and intervals."""
import numpy as np
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

def rmse(a,b): return math.sqrt(mean_squared_error(a,b))

def coverage_and_width(true, lower, upper):
    inside = (true >= lower) & (true <= upper)
    coverage = inside.mean()
    width = (upper - lower).mean()
    return coverage, width

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--true', type=str, required=True, help='npz with y true values (1d array)')
    parser.add_argument('--pred', type=str, required=True, help='npz with pred mean (1d array)')
    parser.add_argument('--pred_uncert', type=str, required=False, help='npz with pred samples or lower/upper')
    args = parser.parse_args()
    true = np.load(args.true)['y_test']
    pred = np.load(args.pred)['pred_mean']
    print('RMSE:', rmse(true, pred))
    print('MAE:', mean_absolute_error(true, pred))
    if args.pred_uncert:
        arr = np.load(args.pred_uncert)
        if 'pred_samples' in arr:
            samples = arr['pred_samples']  # shape (n_samples, n_points)
            lower = np.percentile(samples, 5, axis=0)
            upper = np.percentile(samples, 95, axis=0)
            cov, wid = coverage_and_width(true, lower, upper)
            print('Coverage (90%):', cov)
            print('Mean interval width:', wid)
        elif 'lower' in arr and 'upper' in arr:
            lower = arr['lower']
            upper = arr['upper']
            cov, wid = coverage_and_width(true, lower, upper)
            print('Coverage:', cov)
            print('Mean interval width:', wid)

if __name__ == '__main__':
    main()
