"""
Tabular foundation-model benchmark for diabetes risk prediction.

Compares a TabICLv2-style baseline against the strongest classical baselines
on the same train/test split and optional runtime-limited subsets.
"""

import argparse
import os
import time

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

try:
    from tabicl import TabICLClassifier
except ImportError:  # pragma: no cover - runtime dependency check
    TabICLClassifier = None


DATA_PATH = "f:/Diabetics Project/data/diabetes-prediction-dataset/diabetes_prediction_dataset.csv"
RESULTS_PATH = "f:/Diabetics Project/results/tabular_foundation_benchmark.csv"
FEATURE_COLUMNS = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
]
CATEGORICAL_COLUMNS = ["gender", "smoking_history"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=20000, help="Optional cap for training rows.")
    parser.add_argument("--max-test", type=int, default=5000, help="Optional cap for test rows.")
    parser.add_argument("--device", type=str, default=None, help="Optional TabICL device override.")
    parser.add_argument("--n-estimators", type=int, default=8, help="TabICL ensemble size.")
    return parser.parse_args()


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


def load_data(max_train=None, max_test=None):
    df = pd.read_csv(DATA_PATH, usecols=FEATURE_COLUMNS + ["diabetes"])
    X = df[FEATURE_COLUMNS]
    y = df["diabetes"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_train, y_train = stratified_cap(X_train, y_train, max_train)
    X_test, y_test = stratified_cap(X_test, y_test, max_test)
    return X_train.reset_index(drop=True), X_test.reset_index(drop=True), y_train.reset_index(drop=True), y_test.reset_index(drop=True)


def to_tabicl_arrays(X_train: pd.DataFrame, X_test: pd.DataFrame):
    train_encoded = X_train.copy()
    test_encoded = X_test.copy()

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train_encoded[CATEGORICAL_COLUMNS] = encoder.fit_transform(X_train[CATEGORICAL_COLUMNS].astype(str))
    test_encoded[CATEGORICAL_COLUMNS] = encoder.transform(X_test[CATEGORICAL_COLUMNS].astype(str))

    return train_encoded.to_numpy(dtype=float), test_encoded.to_numpy(dtype=float)


def to_tree_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    combined = pd.concat([X_train, X_test], ignore_index=True)
    combined = pd.get_dummies(combined, drop_first=True)

    split_idx = len(X_train)
    X_train_encoded = combined.iloc[:split_idx].reset_index(drop=True)
    X_test_encoded = combined.iloc[split_idx:].reset_index(drop=True)
    return X_train_encoded, X_test_encoded


def collect_metrics(model_name, y_true, probs, train_time):
    preds = (probs > 0.5).astype(int)
    return {
        "Model": model_name,
        "AUC": roc_auc_score(y_true, probs),
        "Accuracy": accuracy_score(y_true, preds),
        "F1-Score": f1_score(y_true, preds),
        "Precision": precision_score(y_true, preds, zero_division=0),
        "Recall": recall_score(y_true, preds, zero_division=0),
        "Time(s)": train_time,
    }


def run_tree_baseline(model, model_name, X_train, X_test, y_train, y_test):
    start = time.time()
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    train_time = time.time() - start
    return collect_metrics(model_name, y_test, probs, train_time)


def run_tabicl_baseline(X_train, X_test, y_train, y_test, args):
    if TabICLClassifier is None:
        raise ImportError("TabICL is not installed. Run: python -m pip install tabicl")

    model = TabICLClassifier(
        n_estimators=args.n_estimators,
        device=args.device,
        verbose=False,
        allow_auto_download=True,
    )

    X_train_tabicl, X_test_tabicl = to_tabicl_arrays(X_train, X_test)

    start = time.time()
    model.fit(X_train_tabicl, y_train.to_numpy())
    probs = model.predict_proba(X_test_tabicl)[:, 1]
    train_time = time.time() - start
    return collect_metrics("TabICLv2", y_test, probs, train_time)


def main():
    args = parse_args()
    X_train, X_test, y_train, y_test = load_data(args.max_train, args.max_test)

    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")

    X_train_tree, X_test_tree = to_tree_features(X_train, X_test)

    results = []
    results.append(
        run_tree_baseline(
            LGBMClassifier(n_estimators=200, random_state=42, verbose=-1),
            "LightGBM",
            X_train_tree,
            X_test_tree,
            y_train,
            y_test,
        )
    )
    results.append(
        run_tree_baseline(
            XGBClassifier(n_estimators=200, random_state=42, eval_metric="logloss"),
            "XGBoost",
            X_train_tree,
            X_test_tree,
            y_train,
            y_test,
        )
    )

    try:
        results.append(run_tabicl_baseline(X_train, X_test, y_train, y_test, args))
    except Exception as exc:
        results.append(
            {
                "Model": "TabICLv2",
                "AUC": np.nan,
                "Accuracy": np.nan,
                "F1-Score": np.nan,
                "Precision": np.nan,
                "Recall": np.nan,
                "Time(s)": np.nan,
                "Error": str(exc),
            }
        )
        print(f"TabICLv2 baseline unavailable: {exc}")

    results_df = pd.DataFrame(results).sort_values("AUC", ascending=False, na_position="last")
    print(results_df.to_string(index=False))
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"Saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
