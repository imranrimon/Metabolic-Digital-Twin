"""
Stacking ensemble benchmark using optimized XGBoost, LightGBM, and FT-Transformer.
"""

import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from metabolic_twin.config import (
    BEST_HYPERPARAMS_PATH,
    DIABETES_100K_DATA_PATH,
    GRANDMASTER_BENCHMARK_PATH,
    STACKING_META_MODEL_PATH,
)
from models_sota import FTTransformerModel
from training_utils import progress, update_progress

sys.path.append("src")
from grandmaster_features import apply_grandmaster_features


warnings.filterwarnings("ignore")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

DATA_PATH = DIABETES_100K_DATA_PATH
RESULTS_PATH = GRANDMASTER_BENCHMARK_PATH
META_MODEL_PATH = STACKING_META_MODEL_PATH
HYPERPARAMS_PATH = BEST_HYPERPARAMS_PATH


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = pd.get_dummies(df, drop_first=True)
    df = apply_grandmaster_features(df)

    X = df.drop("diabetes", axis=1).values
    y = df["diabetes"].values
    return X, y, df.columns.drop("diabetes")


def train_ft_transformer_fold(X_train_scaled, y_train_res, X_val_scaled):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_t = torch.FloatTensor(X_train_scaled).to(device)
    y_t = torch.FloatTensor(y_train_res).unsqueeze(-1).to(device)
    X_v = torch.FloatTensor(X_val_scaled).to(device)

    cat_train = torch.zeros(X_t.shape[0], 1, dtype=torch.long).to(device)
    cat_val = torch.zeros(X_v.shape[0], 1, dtype=torch.long).to(device)

    model = FTTransformerModel(X_t.shape[1], [2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(device))

    batch_size = 1024
    model.train()
    epoch_bar = progress(range(1, 6), desc="FTTransformer fold", leave=False)
    for _ in epoch_bar:
        perm = torch.randperm(X_t.size(0), device=device)
        epoch_loss = 0.0
        for start in range(0, X_t.size(0), batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            outputs = model(X_t[idx], cat_train[idx])
            loss = criterion(outputs, y_t[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        update_progress(epoch_bar, loss=epoch_loss)

    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(X_v, cat_val)).cpu().numpy().flatten()

    del model, X_t, y_t, X_v, cat_train, cat_val
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return preds


def get_oof_predictions(model_name, params, X, y, n_folds=5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    scores = []

    print(f"Generating OOF for {model_name}...")
    fold_bar = progress(list(enumerate(skf.split(X, y), start=1)), desc=f"{model_name} folds")
    for fold, (train_idx, val_idx) in fold_bar:
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_res)
        X_val_scaled = scaler.transform(X_val)

        if model_name == "xgboost":
            clf = XGBClassifier(**params, n_jobs=-1, random_state=42, eval_metric="logloss")
            clf.fit(X_train_scaled, y_train_res)
            preds = clf.predict_proba(X_val_scaled)[:, 1]
        elif model_name == "lightgbm":
            clf = LGBMClassifier(**params, verbose=-1, random_state=42)
            clf.fit(X_train_scaled, y_train_res)
            preds = clf.predict_proba(X_val_scaled)[:, 1]
        elif model_name == "ft_transformer":
            preds = train_ft_transformer_fold(X_train_scaled, y_train_res, X_val_scaled)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        oof_preds[val_idx] = preds
        fold_auc = roc_auc_score(y_val, preds)
        scores.append(fold_auc)
        update_progress(fold_bar, fold=fold, auc=fold_auc)

    avg_auc = np.mean(scores)
    print(f"  > avg AUC: {avg_auc:.4f}")
    return oof_preds


def main():
    print("=" * 60)
    print("TRAINING STACKING ENSEMBLE (PHASE 10)")
    print("=" * 60)

    X, y, feats = load_data()
    print(f"Data Loaded: {X.shape} with {len(feats)} features")

    try:
        best_params = joblib.load(HYPERPARAMS_PATH)
        print("Loaded optimized hyperparameters.")
    except Exception:
        print("Optimized params not found, using defaults.")
        best_params = {"xgboost": {}, "lightgbm": {}}

    oof_xgb = get_oof_predictions("xgboost", best_params.get("xgboost", {}), X, y)
    oof_lgbm = get_oof_predictions("lightgbm", best_params.get("lightgbm", {}), X, y)
    oof_dl = get_oof_predictions("ft_transformer", {}, X, y)

    print("\nTraining Meta-Learner (Logistic Regression)...")
    X_meta = np.column_stack([oof_xgb, oof_lgbm, oof_dl])

    print(f"Correlation Matrix:\n{pd.DataFrame(X_meta, columns=['XGB', 'LGB', 'DL']).corr()}")

    meta_model = LogisticRegression()
    meta_scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    predictions = np.zeros(len(y))
    for train_idx, val_idx in skf.split(X_meta, y):
        meta_model.fit(X_meta[train_idx], y[train_idx])
        pred = meta_model.predict_proba(X_meta[val_idx])[:, 1]
        predictions[val_idx] = pred
        meta_scores.append(roc_auc_score(y[val_idx], pred))

    final_auc = np.mean(meta_scores)
    print(f"\nSTACKING ENSEMBLE AUC: {final_auc:.5f}")

    results = pd.DataFrame(
        {
            "Model": ["XGBoost (Opt)", "LightGBM (Opt)", "FT-Transformer", "Stacking Ensemble"],
            "AUC": [
                roc_auc_score(y, oof_xgb),
                roc_auc_score(y, oof_lgbm),
                roc_auc_score(y, oof_dl),
                final_auc,
            ],
        }
    )

    print("\n" + "=" * 60)
    print(results.to_string(index=False))
    results.to_csv(RESULTS_PATH, index=False)
    joblib.dump(meta_model, META_MODEL_PATH)

    print(f"Saved results to {RESULTS_PATH}")
    print(f"Saved meta-model to {META_MODEL_PATH}")


if __name__ == "__main__":
    main()
