"""
Hyper-scale EchoCeption-XL training with validation-based checkpointing.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(__file__))

from grandmaster_features import apply_grandmaster_features
from metabolic_twin.config import DIABETES_100K_DATA_PATH, HYPERSCALE_CHECKPOINT_PATH
from training_utils import ValidationCheckpoint, load_model_state, progress, update_progress


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = HYPERSCALE_CHECKPOINT_PATH

if torch.cuda.is_available():
    print(f"Launching Hyper-Scale Train on: {DEVICE} ({torch.cuda.get_device_name(0)})")
else:
    print(f"Launching Hyper-Scale Train on: {DEVICE} (CPU)")


class EchoCeptionXL(nn.Module):
    def __init__(self, input_dim, reservoir_dim=4096, output_dim=1):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, 64)
        self.input_attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)

        self.reservoir_dim = reservoir_dim
        self.input_to_res = nn.Linear(64, reservoir_dim, bias=False)
        self.res_recurrent = nn.Linear(reservoir_dim, reservoir_dim, bias=False)

        with torch.no_grad():
            nn.init.uniform_(self.input_to_res.weight, -0.1, 0.1)
            nn.init.uniform_(self.res_recurrent.weight, -0.1, 0.1)
            try:
                v = torch.randn(reservoir_dim, 1)
                for _ in range(5):
                    v = self.res_recurrent.weight @ v
                spectral_radius = torch.norm(v) / torch.norm(torch.randn(reservoir_dim, 1))
                self.res_recurrent.weight.data *= 0.9 / (spectral_radius + 1e-5)
            except Exception:
                pass

        self.input_to_res.weight.requires_grad = False
        self.res_recurrent.weight.requires_grad = False

        dim = reservoir_dim
        self.b1 = nn.Linear(dim, dim // 4)
        self.b2 = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim // 4),
        )
        self.b3 = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim // 4),
        )
        self.perceiver = nn.Linear(dim, dim // 4)

        self.concat_dim = (dim // 4) * 4
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.concat_dim),
            nn.Dropout(0.3),
            nn.Linear(self.concat_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, x):
        emb = self.input_proj(x).unsqueeze(1)
        attn_out, _ = self.input_attn(emb, emb, emb)
        x_attn = attn_out.squeeze(1)

        res = torch.tanh(self.input_to_res(x_attn))
        res = torch.tanh(self.input_to_res(x_attn) + self.res_recurrent(res))

        o1 = self.b1(res)
        o2 = self.b2(res)
        o3 = self.b3(res)
        o4 = self.perceiver(res)
        concat = torch.cat([o1, o2, o3, o4], dim=1)
        return self.classifier(concat)


def load_data():
    df = pd.read_csv(DIABETES_100K_DATA_PATH)
    df = apply_grandmaster_features(df)
    df_num = df.select_dtypes(include=[np.number])
    X = df_num.drop(columns=["diabetes"], errors="ignore").values
    y = df["diabetes"].values
    return X, y


def evaluate_auc(model, loader):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE)
            batch_preds = torch.sigmoid(model(batch_x))
            preds.append(batch_preds.cpu().numpy())
            labels.append(batch_y.numpy())
    y_pred = np.vstack(preds)
    y_true = np.vstack(labels)
    return roc_auc_score(y_true, y_pred)


def train_hyperscale():
    X, y = load_data()
    print(f"Data Loaded: {X.shape}")

    X_train_raw, X_temp_raw, y_train_raw, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train_raw)
    print(f"SMOTE Resampled: {X_train_balanced.shape}")

    train_ds = TensorDataset(
        torch.FloatTensor(X_train_balanced),
        torch.FloatTensor(y_train_balanced).unsqueeze(1),
    )
    val_ds = TensorDataset(torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val).unsqueeze(1))
    test_ds = TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test).unsqueeze(1))

    train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4096)
    test_loader = DataLoader(test_ds, batch_size=4096)

    model = EchoCeptionXL(input_dim=X.shape[1], reservoir_dim=8192).to(DEVICE)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-2,
        steps_per_epoch=len(train_loader),
        epochs=50,
    )
    criterion = nn.BCEWithLogitsLoss()
    checkpoint = ValidationCheckpoint(CHECKPOINT_PATH, metric_name="val_auc", mode="max")

    print("\nStarting 50 Epoch Hyperscale Run...")
    epoch_bar = progress(range(1, 51), desc="EchoCeptionXL epochs")
    for epoch in epoch_bar:
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        val_auc = evaluate_auc(model, val_loader)
        checkpoint.update(
            model,
            epoch,
            val_auc,
            extra_metadata={"train_loss": avg_train_loss},
        )
        update_progress(epoch_bar, train_loss=avg_train_loss, val_auc=val_auc)

    load_model_state(model, CHECKPOINT_PATH, map_location=DEVICE)
    test_auc = evaluate_auc(model, test_loader)

    print(f"\nBest validation AUC: {checkpoint.best_metric:.5f} at epoch {checkpoint.best_epoch}")
    print(f"Final Hyperscale Test AUC: {test_auc:.5f}")
    if test_auc > 0.98:
        print("Mission accomplished: broke the 98% barrier.")
    else:
        print("Close, but not quite 98%. Try ensemble next.")


if __name__ == "__main__":
    train_hyperscale()
