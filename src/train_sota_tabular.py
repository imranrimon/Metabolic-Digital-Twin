import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from models_sota import FTTransformerModel
from models import AttentionResNetRisk

def load_data_split():
    df = pd.read_csv('f:/Diabetics Project/data/diabetes-prediction-dataset/diabetes_prediction_dataset.csv')
    
    # Categorical handling for FT-Transformer
    cat_cols = ['gender', 'smoking_history', 'hypertension', 'heart_disease']
    num_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    
    le = LabelEncoder()
    cat_cards = []
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
        cat_cards.append(len(le.classes_))
        
    X_num = df[num_cols].values
    X_cat = df[cat_cols].values
    y = df['diabetes'].values
    
    scaler = StandardScaler()
    X_num = scaler.fit_transform(X_num)
    
    # Split
    X_num_train, X_num_test, X_cat_train, X_cat_test, y_train, y_test = train_test_split(
        X_num, X_cat, y, test_size=0.2, random_state=42
    )
    
    # SMOTE (only on training)
    smote = SMOTE(random_state=42)
    # Combine for SMOTE
    X_combined_train = np.hstack([X_num_train, X_cat_train])
    X_res, y_res = smote.fit_resample(X_combined_train, y_train)
    
    # Split back
    X_num_res = X_res[:, :len(num_cols)]
    X_cat_res = X_res[:, len(num_cols):].astype(int)
    
    return (torch.FloatTensor(X_num_res), torch.LongTensor(X_cat_res), torch.FloatTensor(y_res),
            torch.FloatTensor(X_num_test), torch.LongTensor(X_cat_test), torch.FloatTensor(y_test),
            cat_cards)

def train_ft_transformer():
    print("\n--- Training FT-Transformer (SOTA Tabular) ---")
    X_num_tr, X_cat_tr, y_tr, X_num_te, X_cat_te, y_te, cat_cards = load_data_split()
    
    train_ds = TensorDataset(X_num_tr, X_cat_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FTTransformerModel(n_num_features=X_num_tr.shape[1], cat_cardinalities=cat_cards).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    epochs = 15
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for b_num, b_cat, b_y in train_loader:
            b_num, b_cat, b_y = b_num.to(device), b_cat.to(device), b_y.to(device).unsqueeze(-1)
            optimizer.zero_grad()
            logits = model(b_num, b_cat)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
        
    # Eval
    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(X_num_te.to(device), X_cat_te.to(device))).cpu().numpy().flatten()
        binary_preds = (preds > 0.5).astype(int)
        acc = (binary_preds == y_te.numpy()).mean()
        print(f"FT-Transformer Test Accuracy: {acc:.4f}")

    torch.save(model.state_dict(), 'f:/Diabetics Project/ft_transformer_risk.pth')
    print("SOTA Model saved.")

if __name__ == "__main__":
    train_ft_transformer()
