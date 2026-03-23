# -*- coding: utf-8 -*-

"""
FTT-Only Model (Ablation Study)
No Graph / No Topology

Run:
python train_ftt_only.py \
  --dataset ./data/HFBTP.csv \
  --max_epochs 200 \
  --use_gpu
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from model import FTTransformerMulti


# =========================================================
# Reproducibility
# =========================================================
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
pl.seed_everything(seed, workers=True)
torch.set_float32_matmul_precision("medium")


# =========================================================
# Dataset
# =========================================================
class BlockChainDataset(Dataset):
    def __init__(self, x_norm, y_norm):
        self.x = torch.from_numpy(x_norm).float()
        self.y = torch.from_numpy(y_norm).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# =========================================================
# FTT Only Model
# =========================================================
class FTTOnly(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.ftt = FTTransformerMulti(
            input_dim=num_features,
            embed_dim=64,
            num_heads=4,
            num_layers=3
        )

    def forward(self, x):
        pred = self.ftt(x)
        throughput = pred[:, 0:1]
        latency_raw = pred[:, 1:2]
        latency = torch.exp(torch.clamp(latency_raw, -5.0, 5.0))

        return torch.cat([throughput, latency], dim=1)


# =========================================================
# Multi-task Uncertainty Loss
# =========================================================
class MultiTaskUncertaintyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_sigma_t = nn.Parameter(torch.zeros(1))
        self.log_sigma_l = nn.Parameter(torch.zeros(1))
        self.mse = nn.MSELoss()
        self.min_log_sigma = -1.5

    def forward(self, pred, target, use_uncertainty=True):
        pred_t, pred_l = pred[:, 0], pred[:, 1]
        tgt_t, tgt_l = target[:, 0], target[:, 1]

        mse_t = self.mse(pred_t, tgt_t)
        mse_l = self.mse(
            torch.log(pred_l + 1e-6),
            torch.log(tgt_l + 1e-6)
        )

        if not use_uncertainty:
            return mse_t + mse_l

        log_sigma_t = torch.clamp(self.log_sigma_t, min=self.min_log_sigma)
        log_sigma_l = torch.clamp(self.log_sigma_l, min=self.min_log_sigma)

        loss = (
            torch.exp(-log_sigma_t) * mse_t +
            torch.exp(-log_sigma_l) * mse_l +
            (log_sigma_t + log_sigma_l)
        )

        return loss


# =========================================================
# Lightning Module
# =========================================================
class LitFTT(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.loss_fn = MultiTaskUncertaintyLoss()
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)

        use_uncertainty = self.current_epoch >= 20
        loss = self.loss_fn(pred, y, use_uncertainty)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


# =========================================================
# Main
# =========================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="./data/HFBTP.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)

    arrival = np.log1p(df["Actual Transaction Arrival Rate"].values.astype(float))
    orderers = df["Orderers"].values.astype(float)
    block = df["Block Size"].values.astype(float)

    X_raw = np.stack([arrival, orderers, block], axis=1)

    Y = df[["Throughput", "Avg Latency"]].values.astype(float)
    Y = np.nan_to_num(Y, nan=0.0)

    Y[:, 1] = np.clip(Y[:, 1],
                      np.percentile(Y[:, 1], 1),
                      np.percentile(Y[:, 1], 99))

    sx = MinMaxScaler().fit(X_raw)
    sy = MinMaxScaler().fit(Y)

    dataset = BlockChainDataset(
        sx.transform(X_raw),
        sy.transform(Y)
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    model = FTTOnly(num_features=3)
    lit = LitFTT(model, args.lr)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="cuda" if args.use_gpu and torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=1.0,
        deterministic=True
    )

    trainer.fit(lit, loader)

    os.makedirs("modelH", exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "scaler_X": sx, "scaler_Y": sy},
        "modelH/FTT_only_42.pth"
    )

    print("✔ FTT-only Training finished.")


if __name__ == "__main__":
    main()