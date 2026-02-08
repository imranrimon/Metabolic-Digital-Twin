"""
Simplified Novel Architecture Benchmark vs Current SOTA
Memory-efficient version with single-fold validation
"""

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
import time
import gc

# Import models
from models_novel import KANModel
from models_sota import FTTransformerModel
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def load_and_split_data():
    """Load and split diabetes data"""
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

def benchmark_model(model_name, X_train, y_train, X_test, y_test, device):
    """Benchmark a single model"""
    print(f"\nTraining {model_name}...")
    start_time = time.time()
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if model_name == "KAN (2024)":
        # KAN training
        X_train_t = torch.FloatTensor(X_train_scaled[:50000]).to(device)  # Subsample to avoid OOM
        y_train_t = torch.FloatTensor(y_train[:50000]).unsqueeze(-1).to(device)
        X_test_t = torch.FloatTensor(X_test_scaled).to(device)
        
        model = KANModel(input_dim=4, hidden_dims=[64, 32], output_dim=1).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        for epoch in range(30):
            model.train()
            optimizer.zero_grad()
            logits = model(X_train_t)
            loss = criterion(logits, y_train_t)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            y_pred_proba = torch.sigmoid(model(X_test_t)).cpu().numpy().flatten()
        
        # Cleanup
        del X_train_t, y_train_t, X_test_t, model
        torch.cuda.empty_cache()
        gc.collect()
        
    elif model_name == "FT-Transformer (SOTA)":
        # FT-Transformer
        X_train_t = torch.FloatTensor(X_train_scaled[:50000]).to(device)
        y_train_t = torch.FloatTensor(y_train[:50000]).unsqueeze(-1).to(device)
        X_test_t = torch.FloatTensor(X_test_scaled).to(device)
        cat_train = torch.zeros(X_train_t.shape[0], 4, dtype=torch.long).to(device)
        cat_test = torch.zeros(X_test_t.shape[0], 4, dtype=torch.long).to(device)
        
        model = FTTransformerModel(n_num_features=4, cat_cardinalities=[2, 2, 2, 2]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        for epoch in range(30):
            model.train()
            optimizer.zero_grad()
            logits = model(X_train_t, cat_train)
            loss = criterion(logits, y_train_t)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            y_pred_proba = torch.sigmoid(model(X_test_t, cat_test)).cpu().numpy().flatten()
        
        # Cleanup
        del X_train_t, y_train_t, X_test_t, cat_train, cat_test, model
        torch.cuda.empty_cache()
        gc.collect()
        
    elif model_name == "LightGBM":
        clf = LGBMClassifier(n_estimators=100, max_depth=5, random_state=42, verbose=-1)
        clf.fit(X_train_scaled, y_train)
        y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
        
    elif model_name == "XGBoost":
        clf = XGBClassifier(n_estimators=100, max_depth=5, random_state=42, eval_metric='logloss')
        clf.fit(X_train_scaled, y_train, verbose=False)
        y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
    
    train_time = time.time() - start_time
    
    # Metrics
    y_pred = (y_pred_proba > 0.5).astype(int)
    auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    
    print(f"  AUC-ROC: {auc:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | Time: {train_time:.2f}s")
    
    return {
        'Model': model_name,
        'AUC-ROC': auc,
        'Accuracy': acc,
        'F1-Score': f1,
        'Precision': prec,
        'Recall': rec,
        'Train Time (s)': train_time
    }

def main():
    print("\n" + "="*80)
    print("NOVEL ARCHITECTURE BENCHMARK vs CURRENT SOTA")
    print("Phase 9: Comparing KAN (2024) Against Established Methods")
    print("="*80 + "\n")
    
    # Load data
    X_train, X_test, y_train, y_test = load_and_split_data()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Train samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}\n")
    
    # Models to benchmark
    models = [
        "KAN (2024)",
        "FT-Transformer (SOTA)",
        "LightGBM",
        "XGBoost"
    ]
    
    results = []
    
    for model_name in models:
        result = benchmark_model(model_name, X_train, y_train, X_test, y_test, device)
        results.append(result)
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80 + "\n")
    print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    
    # Save
    df.to_csv('f:/Diabetics Project/results/novel_architecture_benchmark.csv', index=False)
    print(f"\n{'='*80}")
    print("Results saved to: results/novel_architecture_benchmark.csv")
    print("="*80 + "\n")
    
    # Highlight best model
    best_model = df.loc[df['AUC-ROC'].idxmax()]
    print(f"\n🏆 BEST MODEL: {best_model['Model']}")
    print(f"   AUC-ROC: {best_model['AUC-ROC']:.4f}")
    print(f"   Accuracy: {best_model['Accuracy']:.4f}")
    print(f"   F1-Score: {best_model['F1-Score']:.4f}\n")

if __name__ == "__main__":
    main()
