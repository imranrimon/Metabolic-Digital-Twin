import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import time

from models_novel import KANModel

def load_data():
    """Load and preprocess 100k diabetes dataset"""
    df = pd.read_csv('f:/Diabetics Project/data/diabetes-prediction-dataset/diabetes_prediction_dataset.csv')
    
    # Select numerical features
    num_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    
    # Handle categorical if needed (for now, focusing on numerical only)
    X = df[num_cols].values
    y = df['diabetes'].values
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # SMOTE for class balance
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    return (
        torch.FloatTensor(X_train), torch.FloatTensor(y_train),
        torch.FloatTensor(X_test), torch.FloatTensor(y_test)
    )

def train_kan():
    """Train KAN model on diabetes dataset"""
    print("\n" + "="*60)
    print("Training KAN (Kolmogorov-Arnold Network)")
    print("Dataset: 100k Diabetes Prediction")
    print("="*60 + "\n")
    
    # Load data
    X_train, y_train, X_test, y_test = load_data()
    
    # Create datasets
    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
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
    
    # Training loop
    epochs = 30
    best_auc = 0.0
    start_time = time.time()
    
    for epoch in range(epochs):
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
        
        # Evaluate
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs.flatten())
                all_labels.extend(batch_y.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Metrics
        auc = roc_auc_score(all_labels, all_preds)
        binary_preds = (all_preds > 0.5).astype(int)
        acc = accuracy_score(all_labels, binary_preds)
        f1 = f1_score(all_labels, binary_preds)
        
        # Learning rate scheduling
        scheduler.step(avg_train_loss)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Test AUC-ROC: {auc:.4f}")
            print(f"  Test Accuracy: {acc:.4f}")
            print(f"  Test F1: {f1:.4f}")
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), 'f:/Diabetics Project/kan_diabetes.pth')
    
    training_time = time.time() - start_time
    
    # Final evaluation
    model.load_state_dict(torch.load('f:/Diabetics Project/kan_diabetes.pth'))
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
    print(f"Best Test AUC-ROC: {final_auc:.4f}")
    print(f"Test Accuracy: {final_acc:.4f}")
    print(f"Test F1-Score: {final_f1:.4f}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Training Time: {training_time:.2f}s")
    print(f"Avg Inference Time: {avg_inference:.2f}ms/batch")
    print(f"Model saved to: kan_diabetes.pth")
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
