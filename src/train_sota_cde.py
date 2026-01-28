import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

from models_sota import NeuralCDEModel
from cde_preprocess import prepare_cde_data

def train_cde():
    print("\n--- Training Neural CDE (SOTA Continuous Temporal) ---")
    data_path = "f:/Diabetics Project/data/shanghai_total.csv"
    if not os.path.exists(data_path):
         # Try to generate it if missing
         print("Shanghai total not found. Run shanghai_preprocess.py first.")
         return

    coeffs, X = prepare_cde_data(data_path)
    if coeffs is None:
        print("Failed to prepare CDE data.")
        return
        
    # Simple forecasting task: predict last value from first window_size-1
    # Actually, let's just train to reconstruct or predict next step
    y = torch.FloatTensor(X[:, -1]).unsqueeze(-1)
    dataset = TensorDataset(coeffs, y)
    
    train_size = int(0.8 * len(dataset))
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # input_channels=2 (time, glucose)
    model = NeuralCDEModel(input_channels=2, hidden_channels=32, output_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for b_coeffs, b_y in train_loader:
            b_coeffs, b_y = b_coeffs.to(device), b_y.to(device)
            optimizer.zero_grad()
            preds = model(b_coeffs)
            loss = criterion(preds, b_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
            
    # Save
    torch.save(model.state_dict(), 'f:/Diabetics Project/neural_cde_glucose.pth')
    print("Neural CDE Model saved.")

if __name__ == "__main__":
    train_cde()
