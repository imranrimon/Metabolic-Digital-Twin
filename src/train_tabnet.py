import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import time

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    print("WARNING: pytorch-tabnet not available, using fallback implementation")

def load_data():
    """Load and preprocess 100k diabetes dataset"""
    df = pd.read_csv('f:/Diabetics Project/data/diabetes-prediction-dataset/diabetes_prediction_dataset.csv')
    
    # Select features
    num_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    
    X = df[num_cols].values
    y = df['diabetes'].values
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_test, y_test

def train_tabnet():
    """Train TabNet model on diabetes dataset"""
    print("\n" + "="*60)
    print("Training TabNet (Attentive Interpretable Tabular Learning)")
    print("Dataset: 100k Diabetes Prediction")
    print("="*60 + "\n")
    
    if not TABNET_AVAILABLE:
        print("ERROR: pytorch-tabnet library required but not installed")
        print("Please run: pip install pytorch-tabnet")
        return None
    
    # Load data
    X_train, y_train, X_test, y_test = load_data()
    
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Train samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    # Initialize TabNet
    start_time = time.time()
    
    clf = TabNetClassifier(
        n_d=8,                    # Width of decision prediction layer
        n_a=8,                    # Width of attention embedding
        n_steps=3,                # Number of sequential decision steps
        gamma=1.3,                # Feature reuse coefficient
        n_independent=2,          # Number of independent GLU layers
        n_shared=2,               # Number of shared GLU layers
        epsilon=1e-15,
        momentum=0.02,
        lambda_sparse=1e-3,       # Sparsity regularization
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size":10, "gamma":0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='entmax',       # Attention masking
        verbose=1,
        device_name='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Train
    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_name=['test'],
        eval_metric=['auc', 'accuracy'],
        max_epochs=100,
        patience=15,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )
    
    training_time = time.time() - start_time
    
    # Prediction and metrics
    inference_times = []
    
    # Measure inference time
    batch_size = 512
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i:i+batch_size]
        start = time.time()
        _ = clf.predict_proba(batch)
        inference_times.append(time.time() - start)
    
    # Final predictions
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Metrics
    final_auc = roc_auc_score(y_test, y_pred_proba)
    final_acc = accuracy_score(y_test, y_pred)
    final_f1 = f1_score(y_test, y_pred)
    avg_inference = np.mean(inference_times) * 1000  # ms
    
    # Feature importance
    feature_importance = clf.feature_importances_
    feature_names = ['Age', 'BMI', 'HbA1c', 'Glucose']
    
    print("\n" + "="*60)
    print("TabNet Training Complete!")
    print("="*60)
    print(f"Best Test AUC-ROC: {final_auc:.4f}")
    print(f"Test Accuracy: {final_acc:.4f}")
    print(f"Test F1-Score: {final_f1:.4f}")
    print(f"Training Time: {training_time:.2f}s")
    print(f"Avg Inference Time: {avg_inference:.2f}ms/batch")
    print("\nFeature Importance:")
    for name, importance in zip(feature_names, feature_importance):
        print(f"  {name}: {importance:.4f}")
    print(f"\nModel saved to: tabnet_diabetes")
    print("="*60 + "\n")
    
    # Save model
    clf.save_model('f:/Diabetics Project/tabnet_diabetes')
    
    return {
        'model': 'TabNet',
        'auc_roc': final_auc,
        'accuracy': final_acc,
        'f1_score': final_f1,
        'train_time': training_time,
        'inference_time': avg_inference,
        'feature_importance': dict(zip(feature_names, feature_importance.tolist()))
    }

if __name__ == "__main__":
    results = train_tabnet()
    if results:
        print("\nResults:", results)
