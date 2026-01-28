import torch
import torchcde
import pandas as pd
import numpy as np
import os

def prepare_cde_data(filepath, window_size=60, stride=15):
    """
    Converts CGM CSV data into Neural CDE coefficients.
    """
    df = pd.read_csv(filepath)
    # Assume Shanghai format: glucose column
    glucose = df['glucose'].values
    
    # Create windows
    windows = []
    for i in range(0, len(glucose) - window_size, stride):
        win = glucose[i:i + window_size]
        if not np.isnan(win).any():
            windows.append(win)
            
    if not windows:
        return None
        
    # Scale
    X = np.array(windows)
    X = (X - 100) / 50.0 # Simple normalization
    
    # Add time channel
    times = np.linspace(0, window_size - 1, window_size)
    times = np.tile(times, (X.shape[0], 1))
    
    # Shape: (batch, length, channels) -> (batch, window_size, 2)
    # channel 0: time, channel 1: glucose
    cde_input = np.stack([times, X], axis=-1)
    
    # Convert to torch tensor
    cde_input_t = torch.FloatTensor(cde_input)
    
    # Natural cubic spline interpolation
    coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(cde_input_t)
    
    return coeffs, X

if __name__ == "__main__":
    # Test on one file
    path = "f:/Diabetics Project/data/shanghai_cgm/shanghai_total.csv" # Check path
    if os.path.exists(path):
        coeffs, val = prepare_cde_data(path)
        print(f"Neural CDE Coefficients generated. Shape: {coeffs.shape}")
    else:
        print("Shanghai total path not found. Ensure shanghai_preprocess.py was run.")
