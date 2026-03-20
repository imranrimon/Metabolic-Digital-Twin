"""
Population graph training with validation-based checkpointing.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.dirname(__file__))

from grandmaster_features import apply_grandmaster_features
from metabolic_twin.config import DIABETES_100K_DATA_PATH, GRAPH_SOTA_CHECKPOINT_PATH
from training_utils import ValidationCheckpoint, load_model_state, progress, update_progress


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = GRAPH_SOTA_CHECKPOINT_PATH

if torch.cuda.is_available():
    print(f"Launching RTX 8000 Graph System on: {DEVICE} ({torch.cuda.get_device_name(0)})")
else:
    print(f"Launching RTX 8000 Graph System on: {DEVICE} (CPU)")


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.activation = nn.ReLU()

    def forward(self, x, adj):
        out = self.linear(x)
        out = torch.mm(adj, out)
        return self.activation(out)


class PopulationGraphNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, output_dim=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.gc1 = GraphConvolution(hidden_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, x, adj):
        local_emb = self.encoder(x)
        graph_emb = self.gc1(local_emb, adj)
        graph_emb = self.gc2(graph_emb, adj)
        combined = torch.cat([local_emb, graph_emb], dim=1)
        return self.classifier(combined)


def build_population_graph(X, k=10):
    print("Building 100k Node Graph...")
    start = time.time()

    node_count = X.shape[0]
    indices = []
    values = []

    indices.append(torch.stack([torch.arange(node_count), torch.arange(node_count)]).to(DEVICE))
    values.append(torch.ones(node_count).to(DEVICE))

    print("  - Sorting for implicit community detection...")
    risk_proxies = X[:, 0]
    argsort = torch.argsort(risk_proxies)

    for offset in range(1, k + 1):
        src = argsort[:-offset]
        dst = argsort[offset:]
        edge = torch.stack([src, dst], dim=0)
        reverse_edge = torch.stack([dst, src], dim=0)
        indices.append(edge)
        indices.append(reverse_edge)
        values.append(torch.ones(src.shape[0]).to(DEVICE))
        values.append(torch.ones(src.shape[0]).to(DEVICE))

    indices = torch.cat(indices, dim=1)
    values = torch.cat(values)
    adj = torch.sparse_coo_tensor(indices, values, (node_count, node_count)).to(DEVICE)
    print(f"  - Graph built in {time.time() - start:.2f}s. Edges: {indices.shape[1]}")
    return adj


def masked_auc(logits, labels, indices):
    preds = torch.sigmoid(logits[indices]).detach().cpu().numpy()
    y_true = labels[indices].detach().cpu().numpy()
    return roc_auc_score(y_true, preds)


def train_graph_system():
    df = pd.read_csv(DIABETES_100K_DATA_PATH)
    df = apply_grandmaster_features(df)
    df_num = df.select_dtypes(include=[np.number])
    X = df_num.drop(columns=["diabetes"], errors="ignore").values
    y = df["diabetes"].values

    X_tensor = torch.FloatTensor(X).to(DEVICE)
    y_tensor = torch.FloatTensor(y).reshape(-1, 1).to(DEVICE)
    adj = build_population_graph(X_tensor, k=20)

    model = PopulationGraphNet(input_dim=X.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    checkpoint = ValidationCheckpoint(CHECKPOINT_PATH, metric_name="val_auc", mode="max")

    node_count = X.shape[0]
    indices = torch.randperm(node_count, device=DEVICE)
    train_end = int(0.7 * node_count)
    val_end = int(0.85 * node_count)
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    print("\nStarting Transductive Graph Training...")
    epoch_bar = progress(range(1, 101), desc="PopulationGraph epochs")
    for epoch in epoch_bar:
        model.train()
        optimizer.zero_grad()
        logits = model(X_tensor, adj)
        loss = criterion(logits[train_idx], y_tensor[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(X_tensor, adj)
            val_auc = masked_auc(logits, y_tensor, val_idx)

        checkpoint.update(
            model,
            epoch,
            val_auc,
            extra_metadata={"train_loss": float(loss.item())},
        )
        update_progress(epoch_bar, train_loss=float(loss.item()), val_auc=val_auc)

    load_model_state(model, CHECKPOINT_PATH, map_location=DEVICE)
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor, adj)
        test_auc = masked_auc(logits, y_tensor, test_idx)

    print(f"\nBest validation AUC: {checkpoint.best_metric:.5f} at epoch {checkpoint.best_epoch}")
    print(f"Final Population Graph Test AUC: {test_auc:.5f}")

    os.makedirs("results", exist_ok=True)
    with open("results/graph_benchmark.csv", "w", encoding="utf-8") as handle:
        handle.write(f"Model,AUC,Hardware\nPopulationGraphNet,{test_auc:.5f},RTX 8000")


if __name__ == "__main__":
    train_graph_system()
