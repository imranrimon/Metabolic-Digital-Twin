import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from cgmacros_preprocess import CGMacrosPreprocessor
from ppgr_model import PPGRModel
from training_utils import ValidationCheckpoint, load_model_state, progress, split_dataset, update_progress


CHECKPOINT_PATH = "f:/Diabetics Project/ppgr_model.pth"

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
    train_ds, val_ds, test_ds = split_dataset(dataset, train_fraction=0.7, val_fraction=0.15)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PPGRModel(input_dim=X.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    checkpoint = ValidationCheckpoint(CHECKPOINT_PATH, metric_name="val_mse", mode="min")
    
    print(f"Dataset size: {len(X)} samples. Training on {device}...")
    
    epochs = 40
    epoch_bar = progress(range(1, epochs + 1), desc="PPGR epochs")
    for epoch in epoch_bar:
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

        model.eval()
        val_mse = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)
                val_mse += criterion(preds, batch_y).item()

        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_mse = val_mse / len(val_loader)
        checkpoint.update(
            model,
            epoch,
            avg_val_mse,
            extra_metadata={"train_loss": avg_train_loss},
        )
        update_progress(epoch_bar, train_loss=avg_train_loss, val_mse=avg_val_mse)
            
    # Final Eval
    load_model_state(model, CHECKPOINT_PATH, map_location=device)
    model.eval()
    with torch.no_grad():
        test_mse = 0
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            preds = model(batch_x)
            test_mse += criterion(preds, batch_y).item()
        print(f"Best validation MSE: {checkpoint.best_metric:.4f} at epoch {checkpoint.best_epoch}")
        print(f"Final Test MSE: {test_mse/len(test_loader):.4f}")

    print(f"PPGR model saved to {CHECKPOINT_PATH}.")

if __name__ == "__main__":
    train_ppgr()
