# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import Dataset, DataLoader
from model import FTTransformerMulti
from raft_mlp import RaftMLP
from raft_gcn import RaftGCN
from raft_gat import RaftGAT


# =========================================================
# Dataset
# =========================================================
class BlockChainDataset(Dataset):
    def __init__(self, x_norm, x_raw, topo, y):
        self.x_norm = torch.from_numpy(x_norm).float()
        self.x_raw = torch.from_numpy(x_raw).float()
        self.topo = torch.from_numpy(topo).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.x_norm)

    def __getitem__(self, idx):
        return self.x_norm[idx], self.x_raw[idx], self.topo[idx], self.y[idx]


# =========================================================
# Hybrid Model
# =========================================================
class HybridFTTRaft(nn.Module):
    def __init__(self, num_features, max_orderers=3):
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
            max_orderers=max_orderers
        )
        # self.raft = RaftMLP(
        #       hidden=32,
        #       out_dim=64,
        #       layers=2,
        #       max_orderers=max_orderers
        #   )
        # self.raft = RaftGCN(
        #     hidden=32,
        #     out_dim=64,
        #     layers=2,
        #     max_orderers=max_orderers
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


# =========================================================
# Evaluation
# =========================================================
def evaluate(model_path, dataset_path):

    print(f"\n===== Evaluating {model_path} =====")

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

    sx = ckpt["scaler_X"]
    sy = ckpt["scaler_Y"]

    df = pd.read_csv(dataset_path)

    arrival = np.log1p(df["Send Rates"].values.astype(float))
    orderers = np.full(len(df), 3.0, dtype=float)
    block = df["Block Size"].values.astype(float)

    X_raw = np.stack([arrival, orderers, block], axis=1)
    topo = orderers.reshape(-1, 1)

    Y = df[["Throughput", "Avg Latency"]].values.astype(float)
    Y = np.nan_to_num(Y, nan=0.0)

    Y[:, 1] = np.clip(
        Y[:, 1],
        np.percentile(Y[:, 1], 1),
        np.percentile(Y[:, 1], 99)
    )

    dataset = BlockChainDataset(
        sx.transform(X_raw),
        X_raw,
        topo,
        Y
    )

    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    model = HybridFTTRaft(num_features=3)
    model.load_state_dict(ckpt["model"])
    model.eval()

    preds = []
    trues = []

    with torch.no_grad():
        for x_norm, x_raw, topo, y in loader:
            out = model(x_norm, x_raw, topo)
            preds.append(out.numpy())
            trues.append(y.numpy())

    preds = np.vstack(preds)
    trues = np.vstack(trues)


    preds = sy.inverse_transform(preds)

    results = {}

    for i, name in enumerate(["Throughput", "Avg Latency"]):

        y_true = trues[:, i]
        y_pred = preds[:, i]

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
        r2 = r2_score(y_true, y_pred)

        results[name] = (mae, rmse, mape, r2)

    return results


# =========================================================
# Main
# =========================================================
def main():

    dataset = "./data/BPD1.csv"  #../data/BPD2.csv

    models = [
        "modelB1/Hybrid_FTT_RaftGAT42.pth",    # modelB2/Hybrid_FTT_RaftGAT42.pth
        "modelB1/Hybrid_FTT_RaftGAT2024.pth",   # modelB2/Hybrid_FTT_RaftGAT42.pth
    ]

    all_results = {"Throughput": [], "Avg Latency": []}

    for m in models:
        res = evaluate(m, dataset)
        for k in res:
            all_results[k].append(res[k])

    print("\n================ Stability Results ================")

    for metric_name in ["Throughput", "Avg Latency"]:

        values = np.array(all_results[metric_name])

        mean = values.mean(axis=0)
        std = values.std(axis=0)

        print(f"\n{metric_name}")
        print(f"MAE   : {mean[0]:.6f} ± {std[0]:.6f}")
        print(f"RMSE  : {mean[1]:.6f} ± {std[1]:.6f}")
        print(f"MAPE  : {mean[2]:.6f} ± {std[2]:.6f}")
        print(f"R2    : {mean[3]:.6f}")


if __name__ == "__main__":
    main()