#!/usr/bin/env python3
"""Train a GRU-based model with MC Dropout for uncertainty estimation.
This is a simple, easy-to-read starter implementation using PyTorch.
It trains a one-step-ahead predictor for the first variable.
"""
import argparse
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

class GRUForecast(nn.Module):
    def __init__(self, in_dim, hid_dim=64, n_layers=1, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(in_dim, hid_dim, batch_first=True, dropout=dropout, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hid_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, h = self.gru(x)
        # use last step hidden
        last = out[:, -1, :]
        out = self.dropout(last)
        out = self.fc(out)
        return out.squeeze(-1)

def create_sequences(X, y, seq_len=24):
    Xs, ys = [], []
    for i in range(len(X)-seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

def train(args):
    data = np.load(args.data)
    X = data['X'].astype(np.float32)
    y = data['y'].astype(np.float32)
    seq_len = args.seq_len
    Xs, ys = create_sequences(X, y, seq_len=seq_len)
    # train-test split
    split = int(0.8 * len(Xs))
    X_train, y_train = Xs[:split], ys[:split]
    X_test, y_test = Xs[split:], ys[split:]
    # dataloaders
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GRUForecast(in_dim=X.shape[1], hid_dim=args.hidden, n_layers=1, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_loss = 1e9
    os.makedirs('artifacts', exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
        train_loss = running / len(train_loader.dataset)
        # validation
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                val_running += ((pred - yb)**2).sum().item()
        val_loss = val_running / len(test_loader.dataset)
        print(f'Epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}')
        # save checkpoint
        ckpt = os.path.join('artifacts', 'gru_latest.pt')
        torch.save({'model_state': model.state_dict()}, ckpt)
        # also save test predictions for later uncertainty estimation (single deterministic pass)
        np.savez_compressed('artifacts/test_data.npz', X_test=X_test, y_test=y_test)

    # Save a small inference helper for MC Dropout runs
    torch.save({'model_state': model.state_dict()}, os.path.join('artifacts', 'gru_final.pt'))

# After training, run MC Dropout inference on test set and save samples
def mc_samples(model, X_np, n_samples=100, device=device):
    model.train()  # enable dropout at inference
    X_t = torch.from_numpy(X_np).to(device).float()
    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(X_t).cpu().numpy()
            samples.append(out)
    samples = np.vstack(samples)  # shape (n_samples, n_points)
    return samples

# Produce MC samples and save
mc_n = getattr(args, 'mc_samples', 100)
print(f'Running MC Dropout with {mc_n} samples...')
samples = mc_samples(model, X_test, n_samples=mc_n)
pred_mean = samples.mean(axis=0)
os.makedirs('artifacts', exist_ok=True)
np.savez_compressed('artifacts/gru_mc_samples.npz', pred_samples=samples, pred_mean=pred_mean, y_test=y_test)
print('Saved GRU MC samples to artifacts/gru_mc_samples.npz')

    print('Training complete. Artifacts in ./artifacts')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seq_len', type=int, default=24)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--mc_samples', type=int, default=100, help='Number of MC Dropout samples to draw for uncertainty estimation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
