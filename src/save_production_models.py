"""
Save Production Risk Model and Conformal Artifacts.

Trains the production XGBoost model on the 100k dataset using a
train/calibration split, then serializes:

- the fitted XGBoost model,
- the saved feature schema,
- APS-style adaptive prediction set conformal calibration artifacts.
"""

import os

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from metabolic_twin.config import (
    BEST_HYPERPARAMS_PATH,
    DIABETES_100K_DATA_PATH,
    MODELS_DIR,
    PRODUCTION_CONFORMAL_PATH,
    PRODUCTION_FEATURES_PATH,
    PRODUCTION_PREPROCESS_PATH,
    PRODUCTION_XGBOOST_PATH,
)
from conformal import AdaptivePredictionSetConformalClassifier
from risk_pipeline import load_risk_training_data

DATA_PATH = DIABETES_100K_DATA_PATH
MODEL_PATH = PRODUCTION_XGBOOST_PATH
FEATURES_PATH = PRODUCTION_FEATURES_PATH
PREPROCESS_PATH = PRODUCTION_PREPROCESS_PATH
CONFORMAL_PATH = PRODUCTION_CONFORMAL_PATH

CALIBRATION_SIZE = 0.1
CONFORMAL_ALPHA = 0.1


def main():
    print("Training production risk model with conformal calibration...")

    X, y, category_levels = load_risk_training_data(DATA_PATH)
    feature_columns = list(X.columns)
    print(f"Loaded engineered dataset: X={X.shape}, positive_rate={float(np.mean(y)):.4f}")

    X_train, X_cal, y_train, y_cal = train_test_split(
        X,
        y,
        test_size=CALIBRATION_SIZE,
        random_state=42,
        stratify=y,
    )

    try:
        params = joblib.load(BEST_HYPERPARAMS_PATH)
        xgb_params = params["xgboost"]
        print(f"Loaded XGBoost hyperparameters: {xgb_params}")
    except Exception:
        print("Using default XGBoost hyperparameters.")
        xgb_params = {"n_estimators": 500, "learning_rate": 0.05}

    model = XGBClassifier(**xgb_params, n_jobs=-1, random_state=42, eval_metric="logloss")
    model.fit(X_train, y_train)

    cal_probs = model.predict_proba(X_cal)
    cal_auc = roc_auc_score(y_cal, cal_probs[:, 1])
    print(f"Calibration-split AUC: {cal_auc:.5f}")

    conformal = AdaptivePredictionSetConformalClassifier(
        alpha=CONFORMAL_ALPHA,
        label_names={0: "Low Risk", 1: "High Risk"},
    )
    conformal.fit_from_probabilities(cal_probs, y_cal)

    artifact = conformal.to_artifact()
    artifact["summary"] = {
        "method": conformal.method,
        "target_coverage": 1 - CONFORMAL_ALPHA,
        "empirical_coverage": conformal.empirical_coverage(cal_probs, y_cal),
        "average_set_size": conformal.average_set_size(cal_probs),
        "singleton_rate": conformal.singleton_rate(cal_probs),
        "empty_rate": conformal.empty_rate(cal_probs),
        "calibration_auc": cal_auc,
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    joblib.dump(feature_columns, FEATURES_PATH)
    joblib.dump(
        {
            "feature_columns": feature_columns,
            "category_levels": category_levels,
        },
        PREPROCESS_PATH,
    )
    joblib.dump(artifact, CONFORMAL_PATH)

    print(f"Saved: {MODEL_PATH}")
    print(f"Saved: {FEATURES_PATH}")
    print(f"Saved: {PREPROCESS_PATH}")
    print(f"Saved: {CONFORMAL_PATH}")
    print(
        "Conformal summary:",
        {
            "method": artifact["summary"]["method"],
            "target_coverage": round(artifact["summary"]["target_coverage"], 3),
            "empirical_coverage": round(artifact["summary"]["empirical_coverage"], 3),
            "average_set_size": round(artifact["summary"]["average_set_size"], 3),
            "singleton_rate": round(artifact["summary"]["singleton_rate"], 3),
            "empty_rate": round(artifact["summary"]["empty_rate"], 3),
        },
    )


if __name__ == "__main__":
    main()
