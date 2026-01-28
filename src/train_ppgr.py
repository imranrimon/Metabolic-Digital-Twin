import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from cgmacros_preprocess import CGMacrosPreprocessor
from ppgr_model import PPGRModel

def train_ppgr():
    print("\n--- Training Personalized Glycemic Response (PPGR) Model ---")
    preprocessor = CGMacrosPreprocessor("f:/Diabetics Project/data/cgmacros/data_volume/CGMacros")
    X, y = preprocessor.load_all_ppgr(num_participants=45) # Use majority of participants
    
    if X is None or len(X) == 0:
        print("Failed to extract PPGR data.")
        return
        
    X_t = torch.FloatTensor(X)
    y_t = torch.FloatTensor(y).unsqueeze(-1)
    
    dataset = TensorDataset(X_t, y_t)
    train_size = int(0.8 * len(dataset))
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PPGRModel(input_dim=X.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print(f"Dataset size: {len(X)} samples. Training on {device}...")
    
    epochs = 40
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, MSE Loss: {epoch_loss/len(train_loader):.4f}")
            
    # Final Eval
    model.eval()
    with torch.no_grad():
        test_mse = 0
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            preds = model(batch_x)
            test_mse += criterion(preds, batch_y).item()
        print(f"Final Test MSE: {test_mse/len(test_loader):.4f}")
        
    torch.save(model.state_dict(), 'f:/Diabetics Project/ppgr_model.pth')
    print("PPGR Model saved.")

if __name__ == "__main__":
    train_ppgr()
