# infer_hybrid_cpu.py
# -*- coding: utf-8 -*-

"""
Inference script for Hybrid FT-Transformer + RaftGAT (CPU ONLY)

Example:
python infer_hybrid.py --model modelH/Hybrid_FTT_RaftGAT.pth --arrival 100 --orderers 3   --block 50
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import os

from raft_gat import RaftGAT
from model import FTTransformerMulti


# =========================================================
# Hybrid Model
# =========================================================
class HybridFTTRaftGAT(nn.Module):
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
# Inference (CPU + system constraint)
# =========================================================
@torch.no_grad()
def infer(model, scaler_x, scaler_y, arrival, orderers, block):

    # ===== preprocessing  =====
    arrival_log = np.log1p(arrival)

    x_raw = np.array([[arrival_log, orderers, block]], dtype=np.float32)
    topo = np.array([[orderers]], dtype=np.float32)
    x_norm = scaler_x.transform(x_raw)

    x_raw = torch.from_numpy(x_raw).float()
    x_norm = torch.from_numpy(x_norm).float()
    topo = torch.from_numpy(topo).float()

    # ===== forward =====
    pred_norm = model(x_norm, x_raw, topo).numpy()
    pred = scaler_y.inverse_transform(pred_norm)

    throughput, latency = pred[0]

    # =====constraint =====
    # throughput cannot exceed arrival rate
    throughput = min(throughput, arrival)

    return float(throughput), float(latency)


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--arrival", type=float, required=True)
    parser.add_argument("--orderers", type=float, required=True)
    parser.add_argument("--block", type=float, required=True)
    args = parser.parse_args()

    # ---------- load checkpoint (CPU ONLY) ----------
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)

    model = HybridFTTRaftGAT(num_features=3)
    model.load_state_dict(ckpt["model"])
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    scaler_x = ckpt["scaler_X"]
    scaler_y = ckpt["scaler_Y"]

    # ---------- inference ----------
    throughput, latency = infer(
        model,
        scaler_x,
        scaler_y,
        args.arrival,
        args.orderers,
        args.block
    )

    # ---------- print ----------
    print("\n========== Inference Result (CPU) ==========")
    print(f"Arrival Rate : {args.arrival}")
    print(f"Orderers     : {args.orderers}")
    print(f"Block Size   : {args.block}")
    print("--------------------------------------------")
    print(f"Pred Throughput : {throughput:.3f}")
    print(f"Pred Avg Latency: {latency:.3f}")
    print("============================================")

    # ---------- save to CSV (script directory) ----------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "infer.csv")

    row = {
        "arrival rate": args.arrival,
        "block_size": args.block,
        "throughput": throughput,
        "Avg latency": latency
    }

    df_row = pd.DataFrame([row])

    if os.path.exists(csv_path):
        df_row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(csv_path, index=False)

    print(f"✔ Result saved to {csv_path}")


if __name__ == "__main__":
    main()
