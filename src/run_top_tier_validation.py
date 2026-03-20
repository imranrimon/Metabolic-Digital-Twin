"""
Journal-grade validation add-ons for the production 100k risk model.

Uses the held-out calibration split from the production XGBoost pipeline to
generate:
- bootstrap confidence intervals for AUC,
- a bootstrap CI for the HbA1c ablation delta,
- subgroup analysis by gender and age band,
- a calibration figure and calibration summary.
"""

import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from metabolic_twin.config import (
    BEST_HYPERPARAMS_PATH,
    BOOTSTRAP_CI_PATH,
    CALIBRATION_BINS_PATH,
    CALIBRATION_FIG_PATH,
    CALIBRATION_SUMMARY_PATH,
    DIABETES_100K_DATA_PATH,
    PLOTS_DIR,
    PRODUCTION_PREPROCESS_PATH,
    PRODUCTION_XGBOOST_PATH,
    RESULTS_DIR,
    SUBGROUP_ANALYSIS_PATH,
)
from risk_pipeline import (
    build_risk_feature_frame,
    extract_category_levels,
    prepare_risk_inference_features,
)


DATA_PATH = DIABETES_100K_DATA_PATH
MODEL_PATH = PRODUCTION_XGBOOST_PATH
PREPROCESS_PATH = PRODUCTION_PREPROCESS_PATH
BOOTSTRAP_PATH = BOOTSTRAP_CI_PATH
SUBGROUP_PATH = SUBGROUP_ANALYSIS_PATH

TARGET_COLUMN = "diabetes"
AGE_BINS = [0, 40, 60, np.inf]
AGE_LABELS = ["<40", "40-59", "60+"]
RANDOM_STATE = 42
CALIBRATION_SIZE = 0.1
BOOTSTRAP_ROUNDS = 200
CALIBRATION_BINS = 10


def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def drop_hba1c_related_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_columns = [column for column in df.columns if "hba1c" not in column.lower()]
    return df.loc[:, keep_columns]


def align_test_to_train(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    aligned = pd.DataFrame(0.0, index=test_df.index, columns=train_df.columns)
    common_columns = [column for column in test_df.columns if column in aligned.columns]
    if common_columns:
        aligned.loc[:, common_columns] = test_df[common_columns].astype(float)
    return aligned


def load_xgboost_params():
    try:
        params = joblib.load(BEST_HYPERPARAMS_PATH)
        xgb_params = dict(params.get("xgboost", {}))
    except Exception:
        xgb_params = {"n_estimators": 200, "learning_rate": 0.05}

    xgb_params["n_estimators"] = min(int(xgb_params.get("n_estimators", 200)), 200)
    xgb_params["tree_method"] = "hist"
    return xgb_params


def load_production_split():
    raw_df = pd.read_csv(DATA_PATH)
    X_raw = raw_df.drop(columns=[TARGET_COLUMN]).copy()
    y = raw_df[TARGET_COLUMN].astype(int)

    X_train_raw, X_cal_raw, y_train, y_cal = train_test_split(
        X_raw,
        y,
        test_size=CALIBRATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    subgroup_frame = X_cal_raw[["gender", "age"]].copy().reset_index(drop=True)
    subgroup_frame["age_group"] = pd.cut(
        subgroup_frame["age"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=False,
    ).astype(str)

    preprocess_artifact = joblib.load(PREPROCESS_PATH)
    feature_columns = preprocess_artifact["feature_columns"]
    category_levels = preprocess_artifact.get("category_levels", {})

    X_cal = prepare_risk_inference_features(X_cal_raw, feature_columns, category_levels)

    return {
        "X_train_raw": X_train_raw.reset_index(drop=True),
        "X_cal_raw": X_cal_raw.reset_index(drop=True),
        "X_cal": X_cal,
        "y_train": y_train.reset_index(drop=True),
        "y_cal": y_cal.reset_index(drop=True),
        "subgroup_frame": subgroup_frame,
    }


def load_production_model():
    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))
    return model


def train_ablated_model(X_train_raw: pd.DataFrame, y_train):
    category_levels = extract_category_levels(X_train_raw)
    X_train_features = build_risk_feature_frame(X_train_raw, category_levels=category_levels).drop(
        columns=[TARGET_COLUMN],
        errors="ignore",
    )
    X_train_features = drop_hba1c_related_columns(X_train_features)

    model = XGBClassifier(
        **load_xgboost_params(),
        n_jobs=-1,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )
    model.fit(X_train_features, y_train)
    return model, X_train_features.columns, category_levels


def build_ablated_eval_frame(X_cal_raw: pd.DataFrame, train_columns, category_levels):
    X_cal_features = build_risk_feature_frame(X_cal_raw, category_levels=category_levels).drop(
        columns=[TARGET_COLUMN],
        errors="ignore",
    )
    X_cal_features = drop_hba1c_related_columns(X_cal_features)
    return align_test_to_train(pd.DataFrame(columns=train_columns).astype(float), X_cal_features)


def classification_metrics(y_true, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0

    return {
        "AUC": roc_auc_score(y_true, probs),
        "Accuracy": accuracy_score(y_true, preds),
        "Precision": precision_score(y_true, preds, zero_division=0),
        "Recall": recall_score(y_true, preds, zero_division=0),
        "F1": f1_score(y_true, preds, zero_division=0),
        "Specificity": specificity,
        "NPV": npv,
    }


def paired_bootstrap_auc_distributions(y_true, probs_a, probs_b=None, n_rounds=BOOTSTRAP_ROUNDS, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    probs_a = np.asarray(probs_a)
    probs_b = None if probs_b is None else np.asarray(probs_b)

    aucs_a = []
    aucs_b = []
    deltas = []

    for _ in range(n_rounds):
        idx = rng.integers(0, len(y_true), len(y_true))
        y_sample = y_true[idx]
        if np.unique(y_sample).size < 2:
            continue

        auc_a = roc_auc_score(y_sample, probs_a[idx])
        aucs_a.append(auc_a)

        if probs_b is not None:
            auc_b = roc_auc_score(y_sample, probs_b[idx])
            aucs_b.append(auc_b)
            deltas.append(auc_a - auc_b)

    return (
        np.asarray(aucs_a, dtype=float),
        np.asarray(aucs_b, dtype=float),
        np.asarray(deltas, dtype=float),
    )


def summarize_bootstrap(name, distribution):
    return {
        "Analysis": name,
        "Mean": float(np.mean(distribution)),
        "CI_Lower_95": float(np.quantile(distribution, 0.025)),
        "CI_Upper_95": float(np.quantile(distribution, 0.975)),
        "BootstrapSamples": int(len(distribution)),
    }


def subgroup_analysis(y_true, probs, subgroup_frame: pd.DataFrame):
    rows = []
    analysis_frame = subgroup_frame.copy()
    analysis_frame["y_true"] = np.asarray(y_true)
    analysis_frame["probability"] = np.asarray(probs)

    for group_column in ["gender", "age_group"]:
        for group_value, group_df in analysis_frame.groupby(group_column, dropna=False):
            if len(group_df) == 0:
                continue

            preds = (group_df["probability"] >= 0.5).astype(int)
            row = {
                "GroupType": group_column,
                "Group": str(group_value),
                "N": int(len(group_df)),
                "PositiveRate": float(group_df["y_true"].mean()),
                "MeanPredictedRisk": float(group_df["probability"].mean()),
                "Accuracy": float(accuracy_score(group_df["y_true"], preds)),
                "Precision": float(precision_score(group_df["y_true"], preds, zero_division=0)),
                "Recall": float(recall_score(group_df["y_true"], preds, zero_division=0)),
                "F1": float(f1_score(group_df["y_true"], preds, zero_division=0)),
            }

            if group_df["y_true"].nunique() >= 2:
                row["AUC"] = float(roc_auc_score(group_df["y_true"], group_df["probability"]))
            else:
                row["AUC"] = np.nan

            rows.append(row)

    return pd.DataFrame(rows)


def expected_calibration_error(y_true, probs, n_bins=CALIBRATION_BINS):
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = np.quantile(probs, quantiles)
    bin_edges[0] = 0.0
    bin_edges[-1] = 1.0

    bin_rows = []
    ece = 0.0

    for idx in range(n_bins):
        left = bin_edges[idx]
        right = bin_edges[idx + 1]
        if idx == n_bins - 1:
            mask = (probs >= left) & (probs <= right)
        else:
            mask = (probs >= left) & (probs < right)

        if not np.any(mask):
            continue

        observed = float(np.mean(y_true[mask]))
        predicted = float(np.mean(probs[mask]))
        weight = float(np.mean(mask))
        ece += abs(observed - predicted) * weight

        bin_rows.append(
            {
                "Bin": idx + 1,
                "LowerEdge": float(left),
                "UpperEdge": float(right),
                "Count": int(np.sum(mask)),
                "ObservedRate": observed,
                "PredictedRate": predicted,
                "AbsoluteGap": abs(observed - predicted),
            }
        )

    return float(ece), pd.DataFrame(bin_rows)


def save_calibration_figure(y_true, probs, auc_value):
    frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=CALIBRATION_BINS, strategy="quantile")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    axes[0].plot(mean_pred, frac_pos, marker="o", linewidth=2, color="#1f77b4", label="XGBoost")
    axes[0].set_title("Calibration Curve (Production XGBoost)")
    axes[0].set_xlabel("Mean predicted risk")
    axes[0].set_ylabel("Observed event rate")
    axes[0].legend(loc="upper left")

    axes[1].hist(probs, bins=20, color="#2ca02c", alpha=0.85, edgecolor="white")
    axes[1].set_title(f"Risk Distribution (AUC {auc_value:.4f})")
    axes[1].set_xlabel("Predicted risk")
    axes[1].set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(CALIBRATION_FIG_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ensure_dirs()
    split = load_production_split()

    full_model = load_production_model()
    full_probs = full_model.predict_proba(split["X_cal"])[:, 1]

    ablated_model, ablated_train_columns, ablated_category_levels = train_ablated_model(
        split["X_train_raw"],
        split["y_train"],
    )
    X_cal_ablated = build_ablated_eval_frame(
        split["X_cal_raw"],
        ablated_train_columns,
        ablated_category_levels,
    )
    ablated_probs = ablated_model.predict_proba(X_cal_ablated)[:, 1]

    full_bootstrap, ablated_bootstrap, delta_bootstrap = paired_bootstrap_auc_distributions(
        split["y_cal"],
        full_probs,
        ablated_probs,
    )

    bootstrap_rows = [
        summarize_bootstrap("Production XGBoost calibration-split AUC", full_bootstrap),
        summarize_bootstrap("XGBoost no-HbA1c calibration-split AUC", ablated_bootstrap),
        summarize_bootstrap("XGBoost HbA1c delta AUC", delta_bootstrap),
    ]
    pd.DataFrame(bootstrap_rows).to_csv(BOOTSTRAP_PATH, index=False)

    subgroup_df = subgroup_analysis(split["y_cal"], full_probs, split["subgroup_frame"])
    subgroup_df.to_csv(SUBGROUP_PATH, index=False)

    metrics = classification_metrics(split["y_cal"], full_probs)
    ece_value, bin_df = expected_calibration_error(split["y_cal"], full_probs)
    bin_df.to_csv(CALIBRATION_BINS_PATH, index=False)

    calibration_summary = pd.DataFrame(
        [
            {
                "Model": "Production XGBoost",
                "Dataset": "100k",
                "EvaluationSplit": "production_calibration_split",
                "AUC": metrics["AUC"],
                "BrierScore": brier_score_loss(split["y_cal"], full_probs),
                "ECE": ece_value,
                "Accuracy": metrics["Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "Specificity": metrics["Specificity"],
                "F1": metrics["F1"],
                "NPV": metrics["NPV"],
                "Threshold": 0.5,
            }
        ]
    )
    calibration_summary.to_csv(CALIBRATION_SUMMARY_PATH, index=False)
    save_calibration_figure(split["y_cal"], full_probs, metrics["AUC"])

    print("Saved:", BOOTSTRAP_PATH)
    print("Saved:", SUBGROUP_PATH)
    print("Saved:", CALIBRATION_SUMMARY_PATH)
    print("Saved:", CALIBRATION_BINS_PATH)
    print("Saved:", CALIBRATION_FIG_PATH)
    print(pd.DataFrame(bootstrap_rows).to_string(index=False))
    print(calibration_summary.to_string(index=False))


if __name__ == "__main__":
    main()
