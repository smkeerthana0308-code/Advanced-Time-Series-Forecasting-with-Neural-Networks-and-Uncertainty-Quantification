#!/usr/bin/env python3
"""Evaluation script that compares SARIMAX and GRU (MC Dropout) results and prints a summary table.
Accepts .npz artifacts from the respective scripts.
"""
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
    parser.add_argument('--sarimax', type=str, required=True, help='npz output from baseline_sarimax.py')
    parser.add_argument('--gru', type=str, required=True, help='npz output from GRU MC samples (artifacts/gru_mc_samples.npz)')
    args = parser.parse_args()

    sar = np.load(args.sarimax)
    gru = np.load(args.gru)

    y_sar = sar['y_test']
    y_gru = gru['y_test']
    # ensure same length and alignment
    assert len(y_sar) == len(y_gru), 'Mismatch in test lengths between SARIMAX and GRU outputs'

    sar_mean = sar['pred_mean']
    sar_lower = sar['lower']
    sar_upper = sar['upper']

    if 'pred_samples' in gru:
        samples = gru['pred_samples']  # shape (n_samples, n_points)
        gru_mean = samples.mean(axis=0)
        gru_lower = np.percentile(samples, 5, axis=0)
        gru_upper = np.percentile(samples, 95, axis=0)
    else:
        gru_mean = gru['pred_mean']
        gru_lower = gru.get('lower', gru_mean - 1.0)
        gru_upper = gru.get('upper', gru_mean + 1.0)

    # compute metrics
    sar_rmse = rmse(y_sar, sar_mean)
    sar_mae = mean_absolute_error(y_sar, sar_mean)
    sar_cov, sar_wid = coverage_and_width(y_sar, sar_lower, sar_upper)

    gru_rmse = rmse(y_gru, gru_mean)
    gru_mae = mean_absolute_error(y_gru, gru_mean)
    gru_cov, gru_wid = coverage_and_width(y_gru, gru_lower, gru_upper)

    # print summary table
    print('\nSummary comparison (test set)')
    header = f"{'Model':<25}{'RMSE':>10}{'MAE':>10}{'Coverage(90%)':>18}{'Mean Width':>14}"
    print(header)
    print('-'*len(header))
    print(f"{'SARIMAX (exog)':<25}{sar_rmse:10.4f}{sar_mae:10.4f}{sar_cov:18.3f}{sar_wid:14.4f}")
    print(f"{'GRU (MC Dropout)':<25}{gru_rmse:10.4f}{gru_mae:10.4f}{gru_cov:18.3f}{gru_wid:14.4f}")
    print('\nDetailed metrics can be saved or exported by modifying this script.')

if __name__ == '__main__':
    main()
