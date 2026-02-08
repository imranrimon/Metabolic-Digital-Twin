"""
EchoCeptionNet Benchmark
Testing the Novel Hybrid Architecture on Class Imbalance
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, classification_report
from imblearn.over_sampling import SMOTE
import time
import sys

# Import EchoCeption
from models_novel import EchoCeptionNet, FocalLoss

def load_data():
    df = pd.read_csv('f:/Diabetics Project/data/diabetes-prediction-dataset/diabetes_prediction_dataset.csv')
    
    # Simple preprocessing
    df = pd.get_dummies(df, drop_first=True)
    X = df.drop('diabetes', axis=1).values
    y = df['diabetes'].values
    
    # 80/20 Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test

def train_echoception(X_train, y_train, X_test, y_test, use_focal=True):
    print(f"\nTraining EchoCeptionNet (Focal Loss={use_focal})...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Tensors
    X_t = torch.FloatTensor(X_train).to(device)
    y_t = torch.FloatTensor(y_train).unsqueeze(-1).to(device)
    X_v = torch.FloatTensor(X_test).to(device)
    
    # Model
    model = EchoCeptionNet(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Loss Function
    if use_focal:
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    else:
        criterion = nn.BCEWithLogitsLoss()
        
    # Training Loop
    start_time = time.time()
    epochs = 30
    batch_size = 1024
    
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(X_t.size(0))
        epoch_loss = 0
        for i in range(0, X_t.size(0), batch_size):
            idx = perm[i:i+batch_size]
            optimizer.zero_grad()
            out = model(X_t[idx])
            loss = criterion(out, y_t[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        # print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")
        
    train_time = time.time() - start_time
    
    # Eval
    model.eval()
    with torch.no_grad():
        logits = model(X_v)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        
    auc = roc_auc_score(y_test, probs)
    preds = (probs > 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    print(f"  Time: {train_time:.2f}s | AUC: {auc:.4f} | F1: {f1:.4f}")
    return {
        'Model': 'EchoCeptionNet' + (' (Focal)' if use_focal else ''),
        'AUC': auc,
        'Accuracy': acc,
        'F1-Score': f1,
        'Time': train_time
    }

def main():
    print("="*60)
    print("BENCHMARKING ECHO-CEPTION NET (PHASE 11)")
    print("="*60)
    
    X_train, X_test, y_train, y_test = load_data()
    
    # 1. Train EchoCeption with Focal Loss (Imbalance Specialist)
    res_focal = train_echoception(X_train, y_train, X_test, y_test, use_focal=True)
    
    # 2. Train EchoCeption with Standard BCE (Control)
    res_bce = train_echoception(X_train, y_train, X_test, y_test, use_focal=False)
    
    # Load SOTA results for comparison
    try:
        sota_df = pd.read_csv('f:/Diabetics Project/results/grandmaster_benchmark.csv')
        best_auc = sota_df['AUC'].max()
    except:
        best_auc = 0.9790
        
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print(f"SOTA AUC Reference: {best_auc:.4f}")
    print("="*60)
    
    results = pd.DataFrame([res_focal, res_bce])
    print(results.to_string(index=False))
    
    results.to_csv('f:/Diabetics Project/results/echoception_benchmark.csv', index=False)

if __name__ == "__main__":
    main()
