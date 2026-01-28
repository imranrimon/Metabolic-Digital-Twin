import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from shanghai_preprocess import get_shanghai_sequences
import numpy as np

class GlucoseLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=1):
        super(GlucoseLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def train_forecaster():
    # 1. Load Data
    X, y, scaler = get_shanghai_sequences(seq_length=12, pred_length=1) # 3 hours history
    X = torch.FloatTensor(X).unsqueeze(-1)
    y = torch.FloatTensor(y)
    
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 2. Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GlucoseLSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Training Loop
    epochs = 20
    print(f"Training on {device}...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.6f}")
            
    # 4. Evaluation
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            test_loss += criterion(outputs, batch_y).item()
            
    print(f"Final Test MSE: {test_loss/len(test_loader):.6f}")
    
    # Save model
    torch.save(model.state_dict(), 'f:/Diabetics Project/glucose_lstm.pth')
    print("Model saved to glucose_lstm.pth")

if __name__ == "__main__":
    # Ensure torch is installed
    train_forecaster()
