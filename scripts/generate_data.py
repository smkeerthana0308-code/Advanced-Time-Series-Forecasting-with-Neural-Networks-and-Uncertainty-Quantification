#!/usr/bin/env python3
"""Generate synthetic multivariate time series data and save to .npz file."""
import numpy as np
import argparse
import os

def make_multivar(n_obs=2000, n_vars=5, seed=42):
    rng = np.random.RandomState(seed)
    t = np.arange(n_obs)
    X = []
    for i in range(n_vars):
        # trend component
        trend = 0.001 * (i+1) * t
        # seasonal component (sinusoids with different frequencies and phases)
        seasonal = 2.0 * np.sin(2 * np.pi * t / (50 + 10*i) + rng.rand()*2*np.pi)
        # autoregressive-ish component (lagged)
        ar = 0.0
        noise = rng.normal(scale=0.5 + 0.1*i, size=n_obs)
        series = trend + seasonal + noise
        X.append(series)
    X = np.vstack(X).T  # shape (n_obs, n_vars)
    # create supervised targets for one-step-ahead forecasting for one target (first var)
    # Here y is next-step of variable 0
    y = np.roll(X[:,0], -1)
    # discard last row (no valid target)
    X = X[:-1]
    y = y[:-1]
    return X, y, t[:-1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_obs', type=int, default=2000)
    parser.add_argument('--n_vars', type=int, default=5)
    parser.add_argument('--out', type=str, default='data/synthetic_multivar.npz')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    X, y, t = make_multivar(n_obs=args.n_obs, n_vars=args.n_vars, seed=args.seed)
    np.savez_compressed(args.out, X=X, y=y, t=t)
    print('Saved synthetic data to', args.out)

if __name__ == '__main__':
    main()
