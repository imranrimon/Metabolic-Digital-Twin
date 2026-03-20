"""
EchoCeption benchmark with standard and hyperspherical classifier heads.
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models_novel import EchoCeptionNet, EchoCeptionSphereNet, FocalLoss, HypersphericalPrototypeLoss
from training_utils import ValidationCheckpoint, load_model_state, progress, update_progress


DATA_PATH = "f:/Diabetics Project/data/diabetes-prediction-dataset/diabetes_prediction_dataset.csv"
RESULTS_PATH = "f:/Diabetics Project/results/echoception_benchmark.csv"
REFERENCE_RESULTS_PATH = "f:/Diabetics Project/results/grandmaster_benchmark.csv"
MODEL_PATHS = {
    "EchoCeptionNet (Focal)": "f:/Diabetics Project/models/echoception_focal_best.pth",
    "EchoCeptionNet": "f:/Diabetics Project/models/echoception_bce_best.pth",
    "EchoCeptionSphereNet": "f:/Diabetics Project/models/echoception_sphere_best.pth",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--max-train", type=int, default=0, help="Optional cap on train rows.")
    parser.add_argument("--max-test", type=int, default=0, help="Optional cap on test rows.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_cap(X, y, max_rows, random_state=42):
    if max_rows is None or max_rows <= 0 or len(X) <= max_rows:
        return X, y

    X_cap, _, y_cap, _ = train_test_split(
        X,
        y,
        train_size=max_rows,
        random_state=random_state,
        stratify=y,
    )
    return X_cap, y_cap


def load_data(max_train=0, max_test=0, random_state=42):
    df = pd.read_csv(DATA_PATH)
    df = pd.get_dummies(df, drop_first=True)
    X = df.drop("diabetes", axis=1).values
    y = df["diabetes"].values

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=random_state,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.17647058823529413,
        random_state=random_state,
        stratify=y_train_full,
    )

    X_train, y_train = stratified_cap(X_train, y_train, max_train, random_state=random_state)
    X_test, y_test = stratified_cap(X_test, y_test, max_test, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test


def standardize(X_train, X_val, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_val), scaler.transform(X_test)


def binary_metrics(y_true, probs):
    preds = (probs > 0.5).astype(int)
    return {
        "AUC": roc_auc_score(y_true, probs),
        "Accuracy": accuracy_score(y_true, preds),
        "F1-Score": f1_score(y_true, preds),
        "Precision": precision_score(y_true, preds, zero_division=0),
        "Recall": recall_score(y_true, preds, zero_division=0),
    }


def evaluate_binary_model(model, X_tensor, y_true, probability_fn):
    model.eval()
    with torch.no_grad():
        probs = probability_fn(model, X_tensor)
    return binary_metrics(y_true, probs), probs


def train_standard_variant(X_train, X_val, X_test, y_train, y_val, y_test, args, use_focal=True):
    variant_name = "EchoCeptionNet (Focal)" if use_focal else "EchoCeptionNet"
    print(f"\nTraining {variant_name}...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, X_val, X_test = standardize(X_train, X_val, X_test)

    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(-1).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)

    model = EchoCeptionNet(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = FocalLoss(alpha=0.25, gamma=2.0) if use_focal else nn.BCEWithLogitsLoss()
    checkpoint = ValidationCheckpoint(MODEL_PATHS[variant_name], metric_name="val_auc", mode="max")

    start = time.time()
    epoch_bar = progress(range(1, args.epochs + 1), desc=variant_name)
    for epoch in epoch_bar:
        model.train()
        epoch_loss = 0.0
        perm = torch.randperm(X_train_tensor.size(0), device=device)
        for i in range(0, X_train_tensor.size(0), args.batch_size):
            idx = perm[i:i + args.batch_size]
            optimizer.zero_grad()
            logits = model(X_train_tensor[idx])
            loss = criterion(logits, y_train_tensor[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        val_metrics, _ = evaluate_binary_model(
            model,
            X_val_tensor,
            y_val,
            lambda current_model, features: torch.sigmoid(current_model(features)).cpu().numpy().flatten(),
        )
        avg_train_loss = epoch_loss / max(1, (X_train_tensor.size(0) // args.batch_size))
        checkpoint.update(model, epoch, val_metrics["AUC"], extra_metadata={"train_loss": avg_train_loss})
        update_progress(epoch_bar, train_loss=avg_train_loss, val_auc=val_metrics["AUC"])

    runtime = time.time() - start
    load_model_state(model, MODEL_PATHS[variant_name], map_location=device)
    metrics, _ = evaluate_binary_model(
        model,
        X_test_tensor,
        y_test,
        lambda current_model, features: torch.sigmoid(current_model(features)).cpu().numpy().flatten(),
    )

    print(
        f"  Time: {runtime:.2f}s | Test AUC: {metrics['AUC']:.4f} | "
        f"Val AUC: {checkpoint.best_metric:.4f}"
    )
    return {
        "Model": variant_name,
        "Head": "Standard",
        "Loss": "Focal" if use_focal else "BCE",
        "EmbeddingDim": 0,
        "Time(s)": runtime,
        "TrainRows": len(X_train),
        "TestRows": len(X_test),
        "BestValAUC": checkpoint.best_metric,
        "BestEpoch": checkpoint.best_epoch,
        **metrics,
    }


def train_sphere_variant(X_train, X_val, X_test, y_train, y_val, y_test, args):
    print(f"\nTraining EchoCeptionSphereNet (embedding_dim={args.embedding_dim}, margin={args.margin})...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, X_val, X_test = standardize(X_train, X_val, X_test)

    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)

    model = EchoCeptionSphereNet(input_dim=X_train.shape[1], embedding_dim=args.embedding_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = HypersphericalPrototypeLoss(scale=model.scale, margin=args.margin)
    checkpoint = ValidationCheckpoint(MODEL_PATHS["EchoCeptionSphereNet"], metric_name="val_auc", mode="max")

    start = time.time()
    epoch_bar = progress(range(1, args.epochs + 1), desc="EchoCeptionSphereNet")
    for epoch in epoch_bar:
        model.train()
        epoch_loss = 0.0
        perm = torch.randperm(X_train_tensor.size(0), device=device)
        for i in range(0, X_train_tensor.size(0), args.batch_size):
            idx = perm[i:i + args.batch_size]
            optimizer.zero_grad()
            outputs = model(X_train_tensor[idx], return_details=True)
            loss = criterion(outputs["cosine_logits"], y_train_tensor[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        val_metrics, _ = evaluate_binary_model(
            model,
            X_val_tensor,
            y_val,
            lambda current_model, features: torch.softmax(current_model(features), dim=1)[:, 1].cpu().numpy(),
        )
        avg_train_loss = epoch_loss / max(1, (X_train_tensor.size(0) // args.batch_size))
        checkpoint.update(model, epoch, val_metrics["AUC"], extra_metadata={"train_loss": avg_train_loss})
        update_progress(epoch_bar, train_loss=avg_train_loss, val_auc=val_metrics["AUC"])

    runtime = time.time() - start
    load_model_state(model, MODEL_PATHS["EchoCeptionSphereNet"], map_location=device)
    metrics, _ = evaluate_binary_model(
        model,
        X_test_tensor,
        y_test,
        lambda current_model, features: torch.softmax(current_model(features), dim=1)[:, 1].cpu().numpy(),
    )

    print(
        f"  Time: {runtime:.2f}s | Test AUC: {metrics['AUC']:.4f} | "
        f"Val AUC: {checkpoint.best_metric:.4f}"
    )
    return {
        "Model": "EchoCeptionSphereNet",
        "Head": "Hyperspherical",
        "Loss": f"CosFace(m={args.margin})",
        "EmbeddingDim": args.embedding_dim,
        "Time(s)": runtime,
        "TrainRows": len(X_train),
        "TestRows": len(X_test),
        "BestValAUC": checkpoint.best_metric,
        "BestEpoch": checkpoint.best_epoch,
        **metrics,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 60)
    print("BENCHMARKING ECHOCEPTION VARIANTS")
    print("=" * 60)

    X_train, X_val, X_test, y_train, y_val, y_test = load_data(
        max_train=args.max_train,
        max_test=args.max_test,
        random_state=args.seed,
    )
    print(f"Train rows: {len(X_train)} | Val rows: {len(X_val)} | Test rows: {len(X_test)}")

    results = [
        train_standard_variant(X_train, X_val, X_test, y_train, y_val, y_test, args, use_focal=True),
        train_standard_variant(X_train, X_val, X_test, y_train, y_val, y_test, args, use_focal=False),
        train_sphere_variant(X_train, X_val, X_test, y_train, y_val, y_test, args),
    ]

    try:
        reference_df = pd.read_csv(REFERENCE_RESULTS_PATH)
        best_auc = reference_df["AUC"].max()
    except Exception:
        best_auc = 0.9790

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print(f"Best external reference AUC: {best_auc:.4f}")
    print("=" * 60)

    results_df = pd.DataFrame(results).sort_values("AUC", ascending=False)
    print(results_df.to_string(index=False))
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"Saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
