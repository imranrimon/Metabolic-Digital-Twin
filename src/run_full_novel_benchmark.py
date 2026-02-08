"""
Comprehensive 7-Model Benchmark
Compares SOTA Deep Learning (KAN, Mamba, TFT, TabNet, FT-Transformer) vs Traditional ML
"""

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error
from imblearn.over_sampling import SMOTE
import time
import gc
import sys

# Import models
from models_novel import KANModel, MambaModel, TemporalFusionTransformer
from models_sota import FTTransformerModel
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

def load_tabular_data():
    """Load 100k diabetes dataset for tabular models"""
    df = pd.read_csv('f:/Diabetics Project/data/diabetes-prediction-dataset/diabetes_prediction_dataset.csv')
    num_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    X = df[num_cols].values
    y = df['diabetes'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    return X_train, X_test, y_train, y_test

def load_temporal_data(seq_len=24):
    """Load Shanghai CGM data for temporal models (Mamba, TFT)"""
    # Simulate loading or process real data
    # For benchmark speed, we'll create a synthetic temporal task based on real stats
    # Real implementation would load from f:/Diabetics Project/data/ShanghaiT1DM/
    
    N = 1000 # Samples
    X = torch.randn(N, seq_len, 1) # Glucose history
    y = torch.randn(N, 1) # Future glucose
    
    return X, y

def train_model(model_name, X_train, y_train, X_test, y_test, device):
    print(f"\nTraining {model_name}...")
    start_time = time.time()
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Subsample for speed in this demo run
    n_samples = 50000
    X_train_sub = X_train_scaled[:n_samples]
    y_train_sub = y_train[:n_samples]
    
    if model_name == "TabNet":
        clf = TabNetClassifier(verbose=0, optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-2))
        clf.fit(
            X_train_sub, y_train_sub,
            eval_set=[(X_test_scaled, y_test)],
            max_epochs=20, patience=5, batch_size=1024, virtual_batch_size=128
        )
        y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
    
    elif model_name == "KAN (2024)":
        X_t = torch.FloatTensor(X_train_sub).to(device)
        y_t = torch.FloatTensor(y_train_sub).unsqueeze(-1).to(device)
        model = KANModel(4, [64, 32], 1).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for _ in range(20):
            opt.zero_grad()
            torch.nn.BCEWithLogitsLoss()(model(X_t), y_t).backward()
            opt.step()
        y_pred_proba = torch.sigmoid(model(torch.FloatTensor(X_test_scaled).to(device))).cpu().detach().numpy().flatten()
        del X_t, y_t, model
        
    elif model_name == "Mamba (SSM)":
        # Treating tabular as sequence of length 4 for Mamba
        X_t = torch.FloatTensor(X_train_sub).unsqueeze(1).repeat(1, 8, 1).to(device) # Fake seq
        y_t = torch.FloatTensor(y_train_sub).unsqueeze(-1).to(device)
        model = MambaModel(d_input=4, d_output=1, d_model=32).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for _ in range(20):
            opt.zero_grad()
            torch.nn.BCEWithLogitsLoss()(model(X_t), y_t).backward()
            opt.step()
        X_test_t = torch.FloatTensor(X_test_scaled).unsqueeze(1).repeat(1, 8, 1).to(device)
        y_pred_proba = torch.sigmoid(model(X_test_t)).cpu().detach().numpy().flatten()
        del X_t, y_t, model
        
    elif model_name == "TFT (Transformer)":
        # Treating tabular as sequence for TFT
        X_t = torch.FloatTensor(X_train_sub).unsqueeze(1).to(device) 
        y_t = torch.FloatTensor(y_train_sub).unsqueeze(-1).to(device)
        model = TemporalFusionTransformer(4, 1, hidden_dim=32).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for _ in range(20):
            opt.zero_grad()
            torch.nn.BCEWithLogitsLoss()(model(X_t), y_t).backward()
            opt.step()
        X_test_t = torch.FloatTensor(X_test_scaled).unsqueeze(1).to(device)
        y_pred_proba = torch.sigmoid(model(X_test_t)).cpu().detach().numpy().flatten()
        del X_t, y_t, model
        
    elif model_name == "FT-Transformer":
        X_t = torch.FloatTensor(X_train_sub).to(device)
        y_t = torch.FloatTensor(y_train_sub).unsqueeze(-1).to(device)
        cat = torch.zeros(n_samples, 4, dtype=torch.long).to(device)
        model = FTTransformerModel(4, [2,2,2,2]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for _ in range(20):
            opt.zero_grad()
            torch.nn.BCEWithLogitsLoss()(model(X_t, cat), y_t).backward()
            opt.step()
        cat_test = torch.zeros(len(X_test), 4, dtype=torch.long).to(device)
        y_pred_proba = torch.sigmoid(model(torch.FloatTensor(X_test_scaled).to(device), cat_test)).cpu().detach().numpy().flatten()
        del X_t, y_t, model

    elif model_name == "XGBoost":
        clf = XGBClassifier(n_estimators=100, eval_metric='logloss')
        clf.fit(X_train_sub, y_train_sub)
        y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]

    elif model_name == "LightGBM":
        clf = LGBMClassifier(n_estimators=100, verbose=-1)
        clf.fit(X_train_sub, y_train_sub)
        y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]

    train_time = time.time() - start_time
    
    # Metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, (y_pred_proba>0.5).astype(int))
    
    return {'Model': model_name, 'AUC': auc, 'Accuracy': acc, 'Time(s)': train_time}

def main():
    print("="*60)
    print("RUNNING 7-MODEL BENCHMARK (PHASE 9)")
    print("="*60)
    
    X_train, X_test, y_train, y_test = load_tabular_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models = ["XGBoost", "LightGBM", "FT-Transformer", "TabNet", "KAN (2024)", "Mamba (SSM)", "TFT (Transformer)"]
    results = []
    
    for m in models:
        try:
            res = train_model(m, X_train, y_train, X_test, y_test, device)
            results.append(res)
            print(f"Result: {res}")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Failed {m}: {e}")
            
    df = pd.DataFrame(results).sort_values('AUC', ascending=False)
    print("\n" + "="*60)
    print(df.to_string(index=False))
    df.to_csv('f:/Diabetics Project/results/novel_7_model_benchmark.csv', index=False)

if __name__ == '__main__':
    main()
