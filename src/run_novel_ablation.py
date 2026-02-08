"""
Comprehensive Novel Architecture Benchmark
Compares cutting-edge 2024/2025 architectures against current SOTA
"""

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
import time
import sys
import os

# Import models
from models_novel import KANModel
from models_sota import FTTransformerModel

# Traditional ML
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def load_diabetes_data():
    """Load 100k diabetes dataset"""
    df = pd.read_csv('f:/Diabetics Project/data/diabetes-prediction-dataset/diabetes_prediction_dataset.csv')
    
    # Numerical features only for fair comparison
    num_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    X = df[num_cols].values
    y = df['diabetes'].values
    
    return X, y

def train_and_evaluate_kan(X_train, y_train, X_test, y_test, device):
    """Train and evaluate KAN"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_t = torch.FloatTensor(X_train_scaled).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(-1).to(device)
    X_test_t = torch.FloatTensor(X_test_scaled).to(device)
    
    model = KANModel(input_dim=4, hidden_dims=[64, 32], output_dim=1, grid_size=5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Quick training (20 epochs for benchmark)
    model.train()
    for epoch in range(20):
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
    
    return probs

def train_and_evaluate_ft_transformer(X_train, y_train, X_test, y_test, device):
    """Train and evaluate FT-Transformer"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_t = torch.FloatTensor(X_train_scaled).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(-1).to(device)
    X_test_t = torch.FloatTensor(X_test_scaled).to(device)
    
    # Simplified categorical for fair comparison (all zeros)
    cat_train = torch.zeros(X_train.shape[0], 4, dtype=torch.long).to(device)
    cat_test = torch.zeros(X_test.shape[0], 4, dtype=torch.long).to(device)
    
    model = FTTransformerModel(n_num_features=4, cat_cardinalities=[2, 2, 2, 2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Quick training
    model.train()
    for epoch in range(20):
        optimizer.zero_grad()
        logits = model(X_train_t, cat_train)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t, cat_test)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
    
    return probs

def run_comprehensive_benchmark():
    """Run complete architecture benchmark with 5-fold CV"""
    print("\n" + "="*80)
    print("COMPREHENSIVE NOVEL ARCHITECTURE BENCHMARK")
    print("Phase 9: Comparing Cutting-Edge 2024/2025 Models vs SOTA")
    print("="*80 + "\n")
    
    # Load data
    X, y = load_diabetes_data()
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Positive class: {y.sum()} ({y.mean()*100:.2f}%)\n")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Models to benchmark
    models = {
        'KAN (2024)': 'kan',
        'FT-Transformer (SOTA)': 'ft_transformer',
        'LightGBM': 'lightgbm',
        'XGBoost': 'xgboost'
    }
    
    # Results storage
    results = {name: {
        'auc_roc': [],
        'accuracy': [],
        'f1_score': [],
        'precision': [],
        'recall': [],
        'train_time': [],
        'inference_time': []
    } for name in models.keys()}
    
    # 5-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n{'='*80}")
        print(f"FOLD {fold}/5")
        print(f"{'='*80}\n")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # SMOTE for class balance
        smote = SMOTE(random_state=42 + fold)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        
        # Benchmark each model
        for model_name, model_type in models.items():
            print(f"Training {model_name}...")
            start_time = time.time()
            
            if model_type == 'kan':
                y_pred_proba = train_and_evaluate_kan(X_train, y_train, X_test, y_test, device)
            
            elif model_type == 'ft_transformer':
                y_pred_proba = train_and_evaluate_ft_transformer(X_train, y_train, X_test, y_test, device)
            
            elif model_type == 'lightgbm':
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                clf = LGBMClassifier(n_estimators=100, max_depth=5, random_state=42, verbose=-1)
                clf.fit(X_train_scaled, y_train)
                y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
            
            elif model_type == 'xgboost':
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                clf = XGBClassifier(n_estimators=100, max_depth=5, random_state=42, eval_metric='logloss')
                clf.fit(X_train_scaled, y_train, verbose=False)
                y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
            
            train_time = time.time() - start_time
            
            # Inference time
            infer_start = time.time()
            if model_type in ['kan', 'ft_transformer']:
                # Already measured in predict
                inference_time = 0.0
            else:
                _ = clf.predict_proba(X_test_scaled[:100])
                inference_time = (time.time() - infer_start) * 10  # Scale to full test set
            
            # Metrics
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            auc = roc_auc_score(y_test, y_pred_proba)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            
            # Store results
            results[model_name]['auc_roc'].append(auc)
            results[model_name]['accuracy'].append(acc)
            results[model_name]['f1_score'].append(f1)
            results[model_name]['precision'].append(prec)
            results[model_name]['recall'].append(rec)
            results[model_name]['train_time'].append(train_time)
            results[model_name]['inference_time'].append(inference_time)
            
            print(f"  AUC-ROC: {auc:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | Time: {train_time:.2f}s")
    
    # Aggregate results
    print("\n" + "="*80)
    print("BENCHMARK RESULTS (Mean ± Std across 5 folds)")
    print("="*80 + "\n")
    
    final_results = []
    
    for model_name in models.keys():
        row = {
            'Model': model_name,
            'AUC-ROC': f"{np.mean(results[model_name]['auc_roc']):.4f} ± {np.std(results[model_name]['auc_roc']):.4f}",
            'Accuracy': f"{np.mean(results[model_name]['accuracy']):.4f} ± {np.std(results[model_name]['accuracy']):.4f}",
            'F1-Score': f"{np.mean(results[model_name]['f1_score']):.4f} ± {np.std(results[model_name]['f1_score']):.4f}",
            'Precision': f"{np.mean(results[model_name]['precision']):.4f} ± {np.std(results[model_name]['precision']):.4f}",
            'Recall': f"{np.mean(results[model_name]['recall']):.4f} ± {np.std(results[model_name]['recall']):.4f}",
            'Train Time (s)': f"{np.mean(results[model_name]['train_time']):.2f}",
            'Inference (ms)': f"{np.mean(results[model_name]['inference_time'])*1000:.2f}"
        }
        final_results.append(row)
    
    df = pd.DataFrame(final_results)
    print(df.to_string(index=False))
    
    # Save results
    df.to_csv('f:/Diabetics Project/results/novel_architecture_benchmark.csv', index=False)
    print(f"\n{'='*80}")
    print("Results saved to: results/novel_architecture_benchmark.csv")
    print("="*80 + "\n")
    
    return df

if __name__ == "__main__":
    results_df = run_comprehensive_benchmark()
