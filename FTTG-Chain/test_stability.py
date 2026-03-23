# -*- coding: utf-8 -*-
"""
Stability Test for HybridModel

python test_stability.py

Metrics:
MAE, RMSE, MAPE → mean ± std
R2 → mean only
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from raft_gcn import RaftGCN
from raft_mlp import RaftMLP
from raft_gat import RaftGAT
from model import FTTransformerMulti


# ======================================================
# Model definition 
# ======================================================
class HybridFTTRaft(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.ftt = FTTransformerMulti(
            input_dim=num_features,
            embed_dim=64,
            num_heads=4,
            num_layers=3
        )

        self.raft = RaftGAT(
            hidden=32,
            out_dim=64,
            layers=2,
            max_orderers=9  
        )
        # self.raft = RaftMLP(
        #       hidden=32,
        #       out_dim=64,
        #       layers=2,
        #       max_orderers=9
        #   )
        # self.raft = RaftGCN(
        #     hidden=32,
        #     out_dim=64,
        #     layers=2,
        #     max_orderers=9
        # )
        self.gate = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(64)

        self.head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x_norm, x_raw, topo):
        z_num, _ = self.ftt(x_norm, return_feat=True)
        z_raft = self.raft(x_raw, topo)

        alpha = self.gate(x_raw) * 0.3
        z = self.norm(z_num + alpha * z_raft)

        out = self.head(z)

        throughput = out[:, 0:1]
        latency_raw = out[:, 1:2]
        latency = torch.exp(torch.clamp(latency_raw, -5.0, 5.0))

        return torch.cat([throughput, latency], dim=1)


# ======================================================
# Dataset loading
# ======================================================
def load_dataset(csv_path, scaler_X, scaler_Y):
    df = pd.read_csv(csv_path)

    arrival = np.log1p(df["Actual Transaction Arrival Rate"].values.astype(float))
    orderers = df["Orderers"].values.astype(float)
    block = df["Block Size"].values.astype(float)

    X_raw = np.stack([arrival, orderers, block], axis=1)
    X_norm = scaler_X.transform(X_raw)
    topo = orderers.reshape(-1, 1)

    Y = df[["Throughput", "Avg Latency"]].values.astype(float)
    Y = np.nan_to_num(Y, nan=0.0)

    Y[:, 1] = np.clip(
        Y[:, 1],
        np.percentile(Y[:, 1], 1),
        np.percentile(Y[:, 1], 99)
    )

    return (
        torch.tensor(X_norm).float(),
        torch.tensor(X_raw).float(),
        torch.tensor(topo).float(),
        Y
    )


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
        "seed_42": "modelH/Hybrid_FTT_RaftGAT_42.pth",  # "seed_42": "modelH/Hybrid_FTT_RaftGCN_42.pth"   "seed_42": "modelH/Hybrid_FTT_RaftMLP_42.pth"
        "seed_2024": "modelH/Hybrid_FTT_RaftGAT_2024.pth",   # "seed_2024": "modelH/Hybrid_FTT_RaftGCN_42.pth"   "seed_2024": "modelH/Hybrid_FTT_RaftMLP_42.pth"

    }

    results = []

    for tag, path in model_paths.items():

        print(f"\n===== Evaluating {tag} =====")

        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        model = HybridFTTRaft(num_features=3)
        model.load_state_dict(ckpt["model"])
        model.eval()

        X_norm, X_raw, topo, Y_true = load_dataset(
            "./data/HFBTP.csv",
            ckpt["scaler_X"],
            ckpt["scaler_Y"]
        )

        with torch.no_grad():
            Y_pred_norm = model(X_norm, X_raw, topo).numpy()

        Y_pred = ckpt["scaler_Y"].inverse_transform(Y_pred_norm)

        metrics = {
            "throughput": evaluate(Y_true[:, 0], Y_pred[:, 0]),
            "latency": evaluate(Y_true[:, 1], Y_pred[:, 1]),
        }

        results.append(metrics)

    # ==================================================
    # Stability Statistics
    # ==================================================
    print("\n\n===== Stability Result =====")

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