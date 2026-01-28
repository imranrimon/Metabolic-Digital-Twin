import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

def load_shanghai_cgm(data_dirs):
    all_data = []
    for directory in data_dirs:
        print(f"Loading files from {directory}...")
        for filename in os.listdir(directory):
            if filename.endswith(".xlsx") or filename.endswith(".xls"):
                file_path = os.path.join(directory, filename)
                try:
                    if filename.endswith('.xls'):
                        df = pd.read_excel(file_path, engine='xlrd')
                    else:
                        df = pd.read_excel(file_path, engine='openpyxl')
                    
                    if 'Date' in df.columns and 'CGM (mg / dl)' in df.columns:
                        temp_df = df[['Date', 'CGM (mg / dl)']].copy()
                        temp_df = temp_df.rename(columns={'CGM (mg / dl)': 'glucose'})
                        temp_df = temp_df.dropna(subset=['glucose'])
                        # Store patient/session ID from filename
                        temp_df['patient_id'] = filename.split('_')[0]
                        all_data.append(temp_df)
                except Exception as e:
                    print(f"Skipping {filename}: {e}")
    
    return pd.concat(all_data, ignore_index=True)

def create_sequences(data, seq_length, pred_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length - pred_length):
        x = data[i : (i + seq_length)]
        y = data[(i + seq_length) : (i + seq_length + pred_length)]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def get_shanghai_sequences(seq_length=8, pred_length=1):
    """
    seq_length: past intervals (e.g., 8 * 15m = 2 hours)
    pred_length: future intervals (e.g., 1 * 15m = 15 min)
    """
    base_dir = "f:/Diabetics Project/data/shanghai_dataset"
    dirs = [
        os.path.join(base_dir, "Shanghai_T1DM"),
        os.path.join(base_dir, "Shanghai_T2DM")
    ]
    
    df = load_shanghai_cgm(dirs)
    df.to_csv("f:/Diabetics Project/data/shanghai_total.csv", index=False)
    print("Saved shanghai_total.csv")
    
    # Sort and scale
    scaler = MinMaxScaler()
    df['glucose_scaled'] = scaler.fit_transform(df[['glucose']])
    
    # Simple approach: Create sequences per patient session
    all_x, all_y = [], []
    for patient in df['patient_id'].unique():
        patient_data = df[df['patient_id'] == patient].sort_values('Date')
        if len(patient_data) > seq_length + pred_length:
            x, y = create_sequences(patient_data['glucose_scaled'].values, seq_length, pred_length)
            all_x.append(x)
            all_y.append(y)
            
    return np.concatenate(all_x), np.concatenate(all_y), scaler

if __name__ == "__main__":
    X, y, scaler = get_shanghai_sequences()
    print(f"Total sequences: {X.shape}")
    print(f"Input shape: {X.shape}, Output shape: {y.shape}")
