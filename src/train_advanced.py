import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from preprocess import get_processed_data
from shanghai_preprocess import get_shanghai_sequences
from models import AttentionResNetRisk, STAttentionLSTM
from imblearn.over_sampling import SMOTE
from training_utils import ValidationCheckpoint, load_model_state, progress, split_dataset, update_progress


RISK_CHECKPOINT_PATH = 'f:/Diabetics Project/attention_resnet_risk.pth'
FORECAST_CHECKPOINT_PATH = 'f:/Diabetics Project/st_attention_lstm.pth'

def train_advanced_risk():
    print("\n--- Training Advanced Attention-ResNet Risk Model ---")
    dataset_name = '100k'
    X_train, X_test, y_train, y_test, _ = get_processed_data(dataset_name)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.15,
        random_state=42,
        stratify=y_train,
    )
    
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
    X_val_t = torch.FloatTensor(X_val)
    y_val_np = y_val.values if hasattr(y_val, 'values') else np.array(y_val)
    
    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttentionResNetRisk(input_dim=X_train.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    checkpoint = ValidationCheckpoint(RISK_CHECKPOINT_PATH, metric_name="val_auc", mode="max")
    
    epochs = 30
    model.train()
    epoch_bar = progress(range(1, epochs + 1), desc="AttentionResNet epochs")
    for epoch in epoch_bar:
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

        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(X_val_t.to(device))).cpu().numpy().flatten()
        val_auc = roc_auc_score(y_val_np, val_probs)
        avg_loss = epoch_loss / len(train_loader)
        checkpoint.update(model, epoch, val_auc, extra_metadata={"train_loss": avg_loss})
        update_progress(epoch_bar, train_loss=avg_loss, val_auc=val_auc)
        model.train()
            
    # Quick Eval
    load_model_state(model, RISK_CHECKPOINT_PATH, map_location=device)
    model.eval()
    with torch.no_grad():
        test_preds = torch.sigmoid(model(X_test_t.to(device)))
        test_preds_bin = (test_preds > 0.5).float()
        acc = (test_preds_bin == y_test_t.to(device)).sum().item() / len(y_test_t)
        print(f"Best validation AUC: {checkpoint.best_metric:.4f} at epoch {checkpoint.best_epoch}")
        print(f"Risk Model Acc: {acc:.4f}")
        
    print(f"Risk model saved to {RISK_CHECKPOINT_PATH}")
    return model

def train_advanced_forecaster():
    print("\n--- Training Advanced ST-Attention LSTM Forecaster ---")
    X, y, _ = get_shanghai_sequences(seq_length=12, pred_length=1)
    X = torch.FloatTensor(X).unsqueeze(-1)
    y = torch.FloatTensor(y)
    
    dataset = TensorDataset(X, y)
    train_ds, val_ds, test_ds = split_dataset(dataset, train_fraction=0.7, val_fraction=0.15)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = STAttentionLSTM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    checkpoint = ValidationCheckpoint(FORECAST_CHECKPOINT_PATH, metric_name="val_mse", mode="min")
    
    epochs = 30
    model.train()
    epoch_bar = progress(range(1, epochs + 1), desc="STAttentionLSTM epochs")
    for epoch in epoch_bar:
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        val_mse = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                val_mse += criterion(outputs, batch_y).item()

        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_mse = val_mse / len(val_loader)
        checkpoint.update(model, epoch, avg_val_mse, extra_metadata={"train_loss": avg_train_loss})
        update_progress(epoch_bar, train_loss=avg_train_loss, val_mse=avg_val_mse)
        model.train()

    load_model_state(model, FORECAST_CHECKPOINT_PATH, map_location=device)
    model.eval()
    test_mse = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            test_mse += criterion(outputs, batch_y).item()

    print(f"Best validation MSE: {checkpoint.best_metric:.6f} at epoch {checkpoint.best_epoch}")
    print(f"Final Test MSE: {test_mse/len(test_loader):.6f}")
    print(f"Forecaster saved to {FORECAST_CHECKPOINT_PATH}")
    return model

if __name__ == "__main__":
    risk_m = train_advanced_risk()
    forecast_m = train_advanced_forecaster()
    print("Advanced training complete.")
