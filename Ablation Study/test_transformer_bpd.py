# -*- coding: utf-8 -*-

"""
Stability Test for
Vanilla Transformer (BPD Dataset, Orderers=3)


Output:
MAE, RMSE, MAPE → mean ± std
R2 → mean
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ======================================================
# Vanilla Transformer (must match training)
# ======================================================
class VanillaTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_heads=4, num_layers=3):
        super().__init__()

        self.embed = nn.Linear(input_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.embed(x).unsqueeze(1)
        z = self.encoder(x).mean(dim=1)
        return self.head(z)


# ======================================================
# Load dataset (match BPD training)
# ======================================================
def load_dataset(csv_path, scaler_X):

    df = pd.read_csv(csv_path)

    # Send Rates as arrival
    arrival = df["Send Rates"].values.astype(float)

    # Fixed Orderers = 3
    orderers = np.full(len(df), 3.0, dtype=float)

    block = df["Block Size"].values.astype(float)

    X = np.stack([arrival, orderers, block], axis=1)

    Y = df[["Throughput", "Avg Latency"]].values.astype(float)
    Y = np.nan_to_num(Y, nan=0.0)
    Y[:, 1] = np.clip(Y[:, 1], 1e-3, None)

    X_norm = scaler_X.transform(X)

    return torch.tensor(X_norm).float(), Y


# ======================================================
# Metrics
# ======================================================
def evaluate(y_true, y_pred):

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    epsilon = 1e-8
    mape = np.mean(
        np.abs((y_true - y_pred) / (y_true + epsilon))
    ) * 100

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2_score(y_true, y_pred),
    }


# ======================================================
# Main
# ======================================================
def main():

    model_paths = {
        "seed_42": "model_baselines/VanillaTransformer_BPD1_42.pth",
        "seed_2024": "model_baselines/VanillaTransformer_BPD1_2024.pth",
    }

    results = []

    for tag, path in model_paths.items():

        print(f"\n===== Evaluating {tag} =====")

        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        model = VanillaTransformer(input_dim=3)
        model.load_state_dict(ckpt["model"])
        model.eval()

        X_norm, Y_true = load_dataset(
            "./data/BPD1.csv",
            ckpt["scaler_X"]
        )

        with torch.no_grad():
            Y_pred_norm = model(X_norm).numpy()

        # inverse normalization
        Y_pred = ckpt["scaler_Y"].inverse_transform(Y_pred_norm)

        metrics = {
            "throughput": evaluate(Y_true[:, 0], Y_pred[:, 0]),
            "latency": evaluate(Y_true[:, 1], Y_pred[:, 1]),
        }

        results.append(metrics)

    # ==================================================
    # Stability Summary
    # ==================================================
    print("\n\n===== BPD Stability Result =====")

    task_name = {
        "throughput": "Throughput",
        "latency": "Avg Latency"
    }

    for task in ["throughput", "latency"]:
        print(f"\n{task_name[task]}")

        for metric in ["MAE", "RMSE", "MAPE"]:
            vals = [r[task][metric] for r in results]
            print(f"{metric:<6}: {np.mean(vals):.6f} ± {np.std(vals):.6f}")

        r2_vals = [r[task]["R2"] for r in results]
        print(f"{'R2':<6}: {np.mean(r2_vals):.6f}")


if __name__ == "__main__":
    main()