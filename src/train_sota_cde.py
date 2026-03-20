import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

from metabolic_twin.config import NEURAL_CDE_CHECKPOINT_PATH, SHANGHAI_TOTAL_DATA_PATH
from models_sota import NeuralCDEModel
from cde_preprocess import prepare_cde_data
from training_utils import ValidationCheckpoint, load_model_state, progress, split_dataset, update_progress


CHECKPOINT_PATH = NEURAL_CDE_CHECKPOINT_PATH

def train_cde():
    print("\n--- Training Neural CDE (SOTA Continuous Temporal) ---")
    data_path = SHANGHAI_TOTAL_DATA_PATH
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
    
    train_ds, val_ds, test_ds = split_dataset(dataset, train_fraction=0.7, val_fraction=0.15)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # input_channels=2 (time, glucose)
    model = NeuralCDEModel(input_channels=2, hidden_channels=32, output_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    checkpoint = ValidationCheckpoint(CHECKPOINT_PATH, metric_name="val_mse", mode="min")
    
    epochs = 20
    epoch_bar = progress(range(1, epochs + 1), desc="NeuralCDE epochs")
    for epoch in epoch_bar:
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

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for b_coeffs, b_y in val_loader:
                b_coeffs, b_y = b_coeffs.to(device), b_y.to(device)
                preds = model(b_coeffs)
                val_loss += criterion(preds, b_y).item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        checkpoint.update(
            model,
            epoch,
            avg_val_loss,
            extra_metadata={"train_loss": avg_train_loss},
        )
        update_progress(epoch_bar, train_loss=avg_train_loss, val_loss=avg_val_loss)

    load_model_state(model, CHECKPOINT_PATH, map_location=device)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for b_coeffs, b_y in test_loader:
            b_coeffs, b_y = b_coeffs.to(device), b_y.to(device)
            preds = model(b_coeffs)
            test_loss += criterion(preds, b_y).item()

    print(f"Best validation MSE: {checkpoint.best_metric:.4f} at epoch {checkpoint.best_epoch}")
    print(f"Final Test MSE: {test_loss/len(test_loader):.4f}")
    print(f"Neural CDE model saved to {CHECKPOINT_PATH}.")

if __name__ == "__main__":
    train_cde()
