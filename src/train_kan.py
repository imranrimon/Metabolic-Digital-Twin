import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import time

from metabolic_twin.config import DIABETES_100K_DATA_PATH, KAN_CHECKPOINT_PATH
from models_novel import KANModel
from training_utils import (
    ValidationCheckpoint,
    load_model_state,
    progress,
    stratified_train_val_test_split,
    update_progress,
)


CHECKPOINT_PATH = KAN_CHECKPOINT_PATH

def load_data():
    """Load and preprocess 100k diabetes dataset"""
    df = pd.read_csv(DIABETES_100K_DATA_PATH)
    
    # Select numerical features
    num_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    
    # Handle categorical if needed (for now, focusing on numerical only)
    X = df[num_cols].values
    y = df['diabetes'].values

    train_arrays, y_train, val_arrays, y_val, test_arrays, y_test = stratified_train_val_test_split(
        X,
        labels=y,
        val_size=0.15,
        test_size=0.15,
        random_state=42,
    )

    X_train = train_arrays[0]
    X_val = val_arrays[0]
    X_test = test_arrays[0]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    return (
        torch.FloatTensor(X_train), torch.FloatTensor(y_train),
        torch.FloatTensor(X_val), torch.FloatTensor(y_val),
        torch.FloatTensor(X_test), torch.FloatTensor(y_test)
    )

def train_kan():
    """Train KAN model on diabetes dataset"""
    print("\n" + "="*60)
    print("Training KAN (Kolmogorov-Arnold Network)")
    print("Dataset: 100k Diabetes Prediction")
    print("="*60 + "\n")
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    # Create datasets
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = KANModel(
        input_dim=4,  # age, bmi, HbA1c, glucose
        hidden_dims=[64, 32],
        output_dim=1,
        grid_size=5,
        dropout=0.1
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    checkpoint = ValidationCheckpoint(CHECKPOINT_PATH, metric_name="val_auc", mode="max")
    
    # Training loop
    epochs = 30
    start_time = time.time()
    
    epoch_bar = progress(range(1, epochs + 1), desc="KAN epochs")
    for epoch in epoch_bar:
        # Train
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).unsqueeze(-1)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validate
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.extend(probs.flatten())
                val_labels.extend(batch_y.numpy())
        
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        
        # Metrics
        val_auc = roc_auc_score(val_labels, val_preds)
        binary_preds = (val_preds > 0.5).astype(int)
        val_acc = accuracy_score(val_labels, binary_preds)
        val_f1 = f1_score(val_labels, binary_preds)
        
        # Learning rate scheduling
        scheduler.step(avg_train_loss)
        
        checkpoint.update(
            model,
            epoch,
            val_auc,
            extra_metadata={
                "train_loss": avg_train_loss,
                "learning_rate": optimizer.param_groups[0]['lr'],
            },
        )
        update_progress(
            epoch_bar,
            train_loss=avg_train_loss,
            val_auc=val_auc,
            val_f1=val_f1,
        )
    
    training_time = time.time() - start_time
    
    # Final evaluation
    load_model_state(model, CHECKPOINT_PATH, map_location=device)
    model.eval()
    
    all_preds = []
    all_labels = []
    inference_times = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            
            # Measure inference time
            start = time.time()
            logits = model(batch_x)
            inference_times.append(time.time() - start)
            
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs.flatten())
            all_labels.extend(batch_y.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    binary_preds = (all_preds > 0.5).astype(int)
    
    # Final metrics
    final_auc = roc_auc_score(all_labels, all_preds)
    final_acc = accuracy_score(all_labels, binary_preds)
    final_f1 = f1_score(all_labels, binary_preds)
    avg_inference = np.mean(inference_times) * 1000  # ms
    
    print("\n" + "="*60)
    print("KAN Training Complete!")
    print("="*60)
    print(f"Best Validation AUC-ROC: {checkpoint.best_metric:.4f} (epoch {checkpoint.best_epoch})")
    print(f"Best Test AUC-ROC: {final_auc:.4f}")
    print(f"Test Accuracy: {final_acc:.4f}")
    print(f"Test F1-Score: {final_f1:.4f}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Training Time: {training_time:.2f}s")
    print(f"Avg Inference Time: {avg_inference:.2f}ms/batch")
    print(f"Model saved to: {CHECKPOINT_PATH}")
    print("="*60 + "\n")
    
    return {
        'model': 'KAN',
        'auc_roc': final_auc,
        'accuracy': final_acc,
        'f1_score': final_f1,
        'parameters': total_params,
        'train_time': training_time,
        'inference_time': avg_inference
    }

if __name__ == "__main__":
    results = train_kan()
    print("\nResults:", results)
