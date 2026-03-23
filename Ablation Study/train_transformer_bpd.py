# -*- coding: utf-8 -*-

"""
Vanilla Transformer Encoder for Blockchain Performance Prediction
(BPD Dataset, Orderers = 3)

Run:
python train_transformer_bpd.py \
  --dataset ./data/BPD1.csv \
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


# ===============================
# Reproducibility
# ===============================
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
pl.seed_everything(seed, workers=True)


# ===============================
# Dataset
# ===============================
class BlockChainDataset(Dataset):
    def __init__(self, x_norm, y_norm):
        self.x = torch.from_numpy(x_norm).float()
        self.y = torch.from_numpy(y_norm).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ===============================
# Vanilla Transformer
# ===============================
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
        # x: [B, F] → [B, 1, D]
        x = self.embed(x).unsqueeze(1)
        z = self.encoder(x).mean(dim=1)
        return self.head(z)


# ===============================
# Lightning Module
# ===============================
class LitVanilla(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.loss_fn = nn.MSELoss()
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


# ===============================
# Main
# ===============================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()

    # -------------------------------------------------
    # Load BPD dataset
    # -------------------------------------------------
    df = pd.read_csv(args.dataset)

    # Send Rates -> arrival rate
    arrival = df["Send Rates"].values.astype(float)

    # Inject fixed Orderers = 3 (BPD assumption)
    orderers = np.full(len(df), 3.0, dtype=float)

    # Block size
    block = df["Block Size"].values.astype(float)

    # Input features 
    X = np.stack([arrival, orderers, block], axis=1)

    # Targets
    Y = df[["Throughput", "Avg Latency"]].values.astype(float)
    Y = np.nan_to_num(Y, nan=0.0)

    # Latency numerical safety
    Y[:, 1] = np.clip(Y[:, 1], 1e-3, None)

    # Normalization
    sx = MinMaxScaler().fit(X)
    sy = MinMaxScaler().fit(Y)

    dataset = BlockChainDataset(
        x_norm=sx.transform(X),
        y_norm=sy.transform(Y)
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    model = VanillaTransformer(input_dim=3)
    lit = LitVanilla(model, args.lr)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="cuda" if args.use_gpu and torch.cuda.is_available() else "cpu",
        devices=1
    )

    trainer.fit(lit, loader)

    # -------------------------------------------------
    # Save model
    # -------------------------------------------------
    os.makedirs("model_baselines", exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "scaler_X": sx,
            "scaler_Y": sy
        },
        "model_baselines/VanillaTransformer_BPD1_42.pth"
    )

    print("✔ Vanilla Transformer training finished on BPD dataset.")


if __name__ == "__main__":
    main()
