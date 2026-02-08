"""
Stacking Ensemble for Kaggle Grandmaster Performance
Combines Optimized XGBoost + LightGBM + FT-Transformer using Logistic Regression
"""

import pandas as pd
import numpy as np
import joblib
import torch
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import gc

# Import SOTA Model
from models_sota import FTTransformerModel
import sys
sys.path.append('src')
from grandmaster_features import apply_grandmaster_features

warnings.filterwarnings("ignore")

def load_data():
    df = pd.read_csv('f:/Diabetics Project/data/diabetes-prediction-dataset/diabetes_prediction_dataset.csv')
    
    # Preprocessing identical to optimize script
    # One-Hot Encode
    df = pd.get_dummies(df, drop_first=True)
    
    # Feature Engineering
    df = apply_grandmaster_features(df)
    
    X = df.drop('diabetes', axis=1).values
    y = df['diabetes'].values
    
    return X, y, df.columns.drop('diabetes')

def get_oof_predictions(model_name, params, X, y, n_folds=5):
    """
    Generate Out-Of-Fold predictions for Stacking
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    test_preds = [] # For inference, but here we just do OOF for training meta model
    
    scores = []
    
    print(f"Generating OOF for {model_name}...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # SMOTE inside fold
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_res)
        X_val_scaled = scaler.transform(X_val)
        
        if model_name == 'xgboost':
            clf = XGBClassifier(**params, n_jobs=-1, random_state=42, eval_metric='logloss')
            clf.fit(X_train_scaled, y_train_res)
            preds = clf.predict_proba(X_val_scaled)[:, 1]
            
        elif model_name == 'lightgbm':
            clf = LGBMClassifier(**params, verbose=-1, random_state=42)
            clf.fit(X_train_scaled, y_train_res)
            preds = clf.predict_proba(X_val_scaled)[:, 1]
            
        elif model_name == 'ft_transformer':
            # Simplified DL training for stacking
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Prepare tensors
            X_t = torch.FloatTensor(X_train_scaled).to(device)
            y_t = torch.FloatTensor(y_train_res).unsqueeze(-1).to(device)
            X_v = torch.FloatTensor(X_val_scaled).to(device)
            
            # Dummy categorical (features are engineered and numeric now)
            cat_train = torch.zeros(X_t.shape[0], 1, dtype=torch.long).to(device)
            cat_val = torch.zeros(X_v.shape[0], 1, dtype=torch.long).to(device)
            
            # Using 1 categorical feature placeholder to satisfy model input
            model = FTTransformerModel(X_t.shape[1], [2]).to(device) # Adjust model input dim
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
            pos_weight = torch.tensor([1.0]).to(device) # Balanced by SMOTE
            crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
            # Fast training
            model.train()
            batch_size = 1024
            for epoch in range(5): # Short epochs for demo/stacking speed
                perm = torch.randperm(X_t.size(0))
                for i in range(0, X_t.size(0), batch_size):
                    idx = perm[i:i+batch_size]
                    opt.zero_grad()
                    out = model(X_t[idx], cat_train[idx])
                    loss = crit(out, y_t[idx])
                    loss.backward()
                    opt.step()
            
            model.eval()
            with torch.no_grad():
                preds = torch.sigmoid(model(X_v, cat_val)).cpu().numpy().flatten()
            
            del model, X_t, y_t, X_v
            torch.cuda.empty_cache()
            
        oof_preds[val_idx] = preds
        fold_auc = roc_auc_score(y_val, preds)
        scores.append(fold_auc)
        # print(f"  Fold {fold+1} AUC: {fold_auc:.4f}")
        
    avg_auc = np.mean(scores)
    print(f"  > avg AUC: {avg_auc:.4f}")
    return oof_preds

def main():
    print("="*60)
    print("TRAINING STACKING ENSEMBLE (PHASE 10)")
    print("="*60)
    
    # 1. Load Data
    X, y, feats = load_data()
    print(f"Data Loaded: {X.shape} with {len(feats)} features")
    
    # 2. Load Hyperparams
    try:
        best_params = joblib.load('f:/Diabetics Project/src/best_hyperparams.pkl')
        print("Loaded Optimized Hyperparameters.")
    except:
        print("Optimized params not found, using defaults.")
        best_params = {'xgboost': {}, 'lightgbm': {}}
        
    # 3. Generate Level-1 Predictions (OOF)
    oof_xgb = get_oof_predictions('xgboost', best_params.get('xgboost', {}), X, y)
    oof_lgbm = get_oof_predictions('lightgbm', best_params.get('lightgbm', {}), X, y)
    # Adding Deep Learning Component
    # Note: FT-Transformer might need different preprocessing (embedding), 
    # here we treat all as numeric for simplicity in ensemble
    oof_dl = get_oof_predictions('ft_transformer', {}, X, y)
    
    # 4. Train Meta-Learner
    print("\nTraining Meta-Learner (Logistic Regression)...")
    X_meta = np.column_stack([oof_xgb, oof_lgbm, oof_dl])
    
    # Simple check of correlation
    print(f"Correlation Matrix:\n{pd.DataFrame(X_meta, columns=['XGB','LGB','DL']).corr()}")
    
    meta_model = LogisticRegression()
    # Cross-validate meta-learner
    meta_scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    predictions = np.zeros(len(y)) # Cross-validated stacking predictions
    
    for train_idx, val_idx in skf.split(X_meta, y):
        meta_model.fit(X_meta[train_idx], y[train_idx])
        pred = meta_model.predict_proba(X_meta[val_idx])[:, 1]
        predictions[val_idx] = pred
        meta_scores.append(roc_auc_score(y[val_idx], pred))
        
    final_auc = np.mean(meta_scores)
    print(f"\n🏆 STACKING ENSEMBLE AUC: {final_auc:.5f}")
    
    # Save Results
    results = pd.DataFrame({
        'Model': ['XGBoost (Opt)', 'LightGBM (Opt)', 'FT-Transformer', 'Stacking Ensemble'],
        'AUC': [roc_auc_score(y, oof_xgb), roc_auc_score(y, oof_lgbm), roc_auc_score(y, oof_dl), final_auc]
    })
    
    print("\n" + "="*60)
    print(results.to_string(index=False))
    results.to_csv('f:/Diabetics Project/results/grandmaster_benchmark.csv', index=False)
    
    # Save Meta Model
    joblib.dump(meta_model, 'f:/Diabetics Project/src/stacking_meta_model.pkl')

if __name__ == "__main__":
    main()
