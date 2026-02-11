
"""
Hyper-Scale Optimization Script (RTX 5070Ti Edition)
Target: >98.0% AUC
Architecture: EchoCeption-XL (4096-neuron Reservoir + Self-Attention)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import time
import os
import sys

# Grandmaster Features
sys.path.append(os.path.dirname(__file__))
from grandmaster_features import apply_grandmaster_features

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Launching Hyper-Scale Train on: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# --- EchoCeption-XL Definition ---

class EchoCeptionXL(nn.Module):
    def __init__(self, input_dim, reservoir_dim=4096, output_dim=1):
        super().__init__()
        
        # 0. Pre-Reservoir Self-Attention (New!)
        # Allows the model to weigh input features dynamically before projection
        self.input_proj = nn.Linear(input_dim, 64)
        self.input_attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        
        # 1. Massive Reservoir (Fixed, Random)
        self.reservoir_dim = reservoir_dim
        self.input_to_res = nn.Linear(64, reservoir_dim, bias=False)
        self.res_recurrent = nn.Linear(reservoir_dim, reservoir_dim, bias=False)
        
        with torch.no_grad():
            nn.init.uniform_(self.input_to_res.weight, -0.1, 0.1)
            nn.init.uniform_(self.res_recurrent.weight, -0.1, 0.1)
            # Spectral Radius Control
            try:
                # Fast approximation for huge matrices
                v = torch.randn(reservoir_dim, 1)
                for _ in range(5): v = self.res_recurrent.weight @ v
                spectral_radius = torch.norm(v) / torch.norm(torch.randn(reservoir_dim, 1)) # Rough Estimate
                self.res_recurrent.weight.data *= 0.9 / (spectral_radius + 1e-5)
            except: pass
            
        # Freeze Reservoir
        self.input_to_res.weight.requires_grad = False
        self.res_recurrent.weight.requires_grad = False
        
        # 2. Multi-Scale Inception Head (Wider for XL)
        # Branches: 1x, 3x, 5x, 7x equivalent depth
        dim = reservoir_dim
        self.b1 = nn.Linear(dim, dim//4)
        
        self.b2 = nn.Sequential(
            nn.Linear(dim, dim//4), nn.ReLU(),
            nn.Linear(dim//4, dim//4)
        )
        
        self.b3 = nn.Sequential(
            nn.Linear(dim, dim//4), nn.ReLU(),
            nn.Linear(dim//4, dim//4), nn.ReLU(),
            nn.Linear(dim//4, dim//4)
        )
        
        self.perceiver = nn.Linear(dim, dim//4) # Pooling shortcut
        
        self.concat_dim = (dim//4) * 4
        
        # 3. Deep Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.concat_dim),
            nn.Dropout(0.3),
            nn.Linear(self.concat_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, output_dim)
        )
        
    def forward(self, x):
        # x: (B, input_dim)
        
        # Attention Pre-processing
        emb = self.input_proj(x).unsqueeze(1) # (B, 1, 64)
        attn_out, _ = self.input_attn(emb, emb, emb)
        x_attn = attn_out.squeeze(1) # (B, 64)
        
        # Reservoir Projection
        # h = tanh(Wx + Wh)
        # We do 1 step of recurrence for "Deep" effect
        res = torch.tanh(self.input_to_res(x_attn))
        res = torch.tanh(self.input_to_res(x_attn) + self.res_recurrent(res))
        
        # Inception
        o1 = self.b1(res)
        o2 = self.b2(res)
        o3 = self.b3(res)
        o4 = self.perceiver(res)
        
        concat = torch.cat([o1, o2, o3, o4], dim=1)
        
        return self.classifier(concat)

# --- Data Loading ---

def load_data():
    df = pd.read_csv('f:/Diabetics Project/data/diabetes-prediction-dataset/diabetes_prediction_dataset.csv')
    
    # Apply Grandmaster Features
    df = apply_grandmaster_features(df)
    
    # Select numerical cols (Grandmaster features are numerical)
    # Filter only numeric
    df_num = df.select_dtypes(include=[np.number])
    X = df_num.drop(columns=['diabetes'], errors='ignore').values
    y = df['diabetes'].values
    
    return X, y

# --- Training Loop ---

def train_hyperscale():
    X, y = load_data()
    print(f"Data Loaded: {X.shape}")
    
    # SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"SMOTE Resampled: {X_res.shape}")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)
    
    # 90/10 Split for maximum training data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_res, test_size=0.1, random_state=42)
    
    # Tensors
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test).unsqueeze(1))
    
    # Huge Batch Size for 5070Ti
    train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=4096)
    
    # Model
    model = EchoCeptionXL(input_dim=X.shape[1], reservoir_dim=8192).to(DEVICE) # 8k Neurons!
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4) # Slightly higher decay
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, steps_per_epoch=len(train_loader), epochs=50)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training
    best_auc = 0.0
    print("\nStarting 50 Epoch Hyperscale Run...")
    
    for epoch in range(50):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
        # Eval
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for bx, by in test_loader:
                bx = bx.to(DEVICE)
                pred = torch.sigmoid(model(bx))
                all_preds.append(pred.cpu().numpy())
                all_targets.append(by.numpy())
                
        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_targets)
        auc = roc_auc_score(y_true, y_pred)
        
        if auc > best_auc:
            best_auc = auc
            # Save to f:/Diabetics Project/models/
            torch.save(model.state_dict(), "models/echoception_xl_5070ti.pth")
            print(f"Epoch {epoch+1}: AUC = {auc:.5f} (New Best!)")
        else:
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}: AUC = {auc:.5f}")
                
    print(f"\n🏆 Final Hyperscale AUC: {best_auc:.5f}")
    if best_auc > 0.98:
        print("🚨 MISSION ACCOMPLISHED: BROKE 98% BARRIER! 🚨")
    else:
        print("Close, but not quite 98%. Try Ensemble next.")

if __name__ == "__main__":
    train_hyperscale()
