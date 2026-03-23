# -*- coding: utf-8 -*-

"""
Stability Test:
VT + RaftGAT (BPD Dataset, Orderers=3)

Output:
MAE, RMSE, MAPE → mean ± std
R2 → mean
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from raft_gcn import RaftGCN
from raft_gat import RaftGAT
from raft_mlp import RaftMLP


# ======================================================
# Vanilla Transformer
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
# Hybrid Model (VT + RaftGAT)
# ======================================================
class VanillaFTTRaft(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.ftt = VanillaTransformer(num_features)

        self.raft = RaftGAT(
            hidden=32,
            out_dim=64,
            layers=2,
            max_orderers=3
        )
        # self.raft = RaftMLP(
        #       hidden=32,
        #       out_dim=64,
        #       layers=2,
        #       max_orderers=3
        #   )
        # self.raft = RaftGCN(
        #     hidden=32,
        #     out_dim=64,
        #     layers=2,
        #     max_orderers=3   
        # )
        self.fusion = nn.Sequential(
            nn.Linear(2 + 64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x_norm, x_raw, topo):

        pred_ftt = self.ftt(x_norm)
        feat_raft = self.raft(x_raw, topo)

        z = torch.cat([pred_ftt, feat_raft], dim=1)
        out = self.fusion(z)

        throughput = out[:, 0:1]
        latency_raw = out[:, 1:2]
        latency = torch.exp(torch.clamp(latency_raw, -5.0, 5.0))

        return torch.cat([throughput, latency], dim=1)


# ======================================================
# Load Dataset (strictly match training)
# ======================================================
def load_dataset(csv_path, scaler_X):

    df = pd.read_csv(csv_path)

    arrival = np.log1p(df["Send Rates"].values.astype(float))
    block = df["Block Size"].values.astype(float)
    orderers = np.full(len(df), 3.0)

    X_raw = np.stack([arrival, orderers, block], axis=1)
    topo = orderers.reshape(-1, 1)

    Y = df[["Throughput", "Avg Latency"]].values.astype(float)
    Y = np.nan_to_num(Y, nan=0.0)

    Y[:, 1] = np.clip(
        Y[:, 1],
        np.percentile(Y[:, 1], 1),
        np.percentile(Y[:, 1], 99)
    )

    X_norm = scaler_X.transform(X_raw)

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
        "seed_42": "modelB1/VT_RaftGAT_BPD_42.pth",
        "seed_2024": "modelB1/VT_RaftGAT_BPD_2024.pth",
    }

    results = []

    for tag, path in model_paths.items():

        print(f"\n===== Evaluating {tag} =====")

        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        model = VanillaFTTRaft(num_features=3)
        model.load_state_dict(ckpt["model"])
        model.eval()

        X_norm, X_raw, topo, Y_true = load_dataset(
            "./data/BPD1.csv",
            ckpt["scaler_X"]
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
    # Stability Summary
    # ==================================================
    print("\n\n===== BPD Hybrid Stability Result =====")

    for task in ["throughput", "latency"]:
        print(f"\n{task.capitalize()}")

        for metric in ["MAE", "RMSE", "MAPE"]:
            vals = [r[task][metric] for r in results]
            print(f"{metric:<6}: {np.mean(vals):.6f} ± {np.std(vals):.6f}")

        r2_vals = [r[task]["R2"] for r in results]
        print(f"{'R2':<6}: {np.mean(r2_vals):.6f}")


if __name__ == "__main__":
    main()