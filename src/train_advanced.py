import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
import numpy as np

from preprocess import get_processed_data
from shanghai_preprocess import get_shanghai_sequences
from models import AttentionResNetRisk, STAttentionLSTM
from imblearn.over_sampling import SMOTE

def train_advanced_risk():
    print("\n--- Training Advanced Attention-ResNet Risk Model ---")
    dataset_name = '100k'
    X_train, X_test, y_train, y_test, _ = get_processed_data(dataset_name)
    
    # SMOTE for balance
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    
    # Convert to Tensors
    X_train_t = torch.FloatTensor(X_train_sm)
    # Ensure y is numeric and convert to numpy if it's a Series
    y_train_np = y_train_sm.values if hasattr(y_train_sm, 'values') else np.array(y_train_sm)
    y_train_t = torch.FloatTensor(y_train_np).unsqueeze(-1)
    
    X_test_t = torch.FloatTensor(X_test)
    y_test_np = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
    y_test_t = torch.FloatTensor(y_test_np).unsqueeze(-1)
    
    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttentionResNetRisk(input_dim=X_train.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    
    epochs = 30
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            with autocast():
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
            
    # Quick Eval
    model.eval()
    with torch.no_grad():
        test_preds = torch.sigmoid(model(X_test_t.to(device)))
        test_preds_bin = (test_preds > 0.5).float()
        acc = (test_preds_bin == y_test_t.to(device)).sum().item() / len(y_test_t)
        print(f"Risk Model Acc: {acc:.4f}")
        
    torch.save(model.state_dict(), 'f:/Diabetics Project/attention_resnet_risk.pth')
    return model

def train_advanced_forecaster():
    print("\n--- Training Advanced ST-Attention LSTM Forecaster ---")
    X, y, _ = get_shanghai_sequences(seq_length=12, pred_length=1)
    X = torch.FloatTensor(X).unsqueeze(-1)
    y = torch.FloatTensor(y)
    
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    train_ds, _ = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = STAttentionLSTM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    epochs = 30
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.6f}")
            
    torch.save(model.state_dict(), 'f:/Diabetics Project/st_attention_lstm.pth')
    return model

if __name__ == "__main__":
    risk_m = train_advanced_risk()
    forecast_m = train_advanced_forecaster()
    print("Advanced training complete.")
