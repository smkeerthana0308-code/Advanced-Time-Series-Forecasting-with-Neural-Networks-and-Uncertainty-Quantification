#!/usr/bin/env python3
"""Example showing how to produce MC Dropout samples from a trained model checkpoint.
This file is a usage example and does NOT run training by itself.
"""
import torch, numpy as np
from scripts.train_gru_mc_dropout import GRUForecast, create_sequences
def mc_samples(model, X, n_samples=100, device='cpu'):
    model.train()  # enable dropout at inference time
    samples = []
    X = torch.from_numpy(X).to(device).float()
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(X).cpu().numpy()
            samples.append(out)
    return np.vstack(samples)  # shape (n_samples, n_points)
