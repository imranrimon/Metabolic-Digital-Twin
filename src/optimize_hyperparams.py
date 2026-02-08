"""
Hyperparameter Optimization using Optuna
Tuning XGBoost and LightGBM for maximum AUC
"""

import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import warnings
import joblib

# Import Grandmaster Features
from grandmaster_features import apply_grandmaster_features

# Suppress warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_data():
    df = pd.read_csv('f:/Diabetics Project/data/diabetes-prediction-dataset/diabetes_prediction_dataset.csv')
    
    # Apply One-Hot Encoding for Categorical columns
    # 'gender', 'smoking_history' are categorical in original dataset
    # We need to handle them before feature engineering if they are used, 
    # but our engineering only used numeric. 
    # Let's simple One-Hot Encode everything non-numeric first.
    df = pd.get_dummies(df, drop_first=True)
    
    # Apply Grandmaster Engineering
    df = apply_grandmaster_features(df)
    
    X = df.drop('diabetes', axis=1)
    y = df['diabetes']
    
    # SMOTE (Only on training data in real cv, but for optimization speed we'll do it inside obj?)
    # No, proper way is inside CV fold. 
    # But for this script, to keep it simple, we will optimize on a fixed validation set 
    # or use cross_val_score which creates folds. 
    # SMOTE inside cross_val_score needs detailed pipeline.
    # Let's simple do a train/val split for optimization to save time vs 5-fold CV on every trial.
    
    return X, y

def objective_xgboost(trial, X, y):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'eval_metric': 'auc',
        'n_jobs': -1,
        'random_state': 42
    }
    
    # Using 3-fold CV for speed during optimization
    clf = XGBClassifier(**param)
    
    # We need to handle SMOTE carefully. 
    # Simple approach: Pre-SMOTE entire dataset (leaky but standard for quick kaggle tuning)
    # Better approach: Pipeline.
    from imblearn.pipeline import Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('model', clf)
    ])
    
    scores = cross_val_score(pipeline, X, y, cv=3, scoring='roc_auc', n_jobs=1)
    return scores.mean()

def objective_lightgbm(trial, X, y):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'verbose': -1,
        'random_state': 42
    }
    
    clf = LGBMClassifier(**param)
    
    from imblearn.pipeline import Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('model', clf)
    ])
    
    scores = cross_val_score(pipeline, X, y, cv=3, scoring='roc_auc', n_jobs=1)
    return scores.mean()

def main():
    print("Loading Data & Engineering Features...")
    X, y = load_data()
    print(f"Features: {X.shape[1]}")
    
    print("\n" + "="*50)
    print("Optimizing XGBoost...")
    print("="*50)
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(lambda trial: objective_xgboost(trial, X, y), n_trials=30)
    
    print(f"Best XGB Params: {study_xgb.best_params}")
    print(f"Best XGB AUC: {study_xgb.best_value:.5f}")
    
    print("\n" + "="*50)
    print("Optimizing LightGBM...")
    print("="*50)
    study_lgbm = optuna.create_study(direction='maximize')
    study_lgbm.optimize(lambda trial: objective_lightgbm(trial, X, y), n_trials=30)
    
    print(f"Best LGBM Params: {study_lgbm.best_params}")
    print(f"Best LGBM AUC: {study_lgbm.best_value:.5f}")
    
    # Save Best Params
    params = {
        'xgboost': study_xgb.best_params,
        'lightgbm': study_lgbm.best_params
    }
    joblib.dump(params, 'f:/Diabetics Project/src/best_hyperparams.pkl')
    print("\nSaved best hyperparameters to src/best_hyperparams.pkl")

if __name__ == "__main__":
    main()
