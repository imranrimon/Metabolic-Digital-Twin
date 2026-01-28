import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

class CGMacrosPreprocessor:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.scaler = StandardScaler()
        
    def preprocess_participant(self, participant_id):
        # f:/Diabetics Project/data/cgmacros/data_volume/CGMacros/CGMacros-001/CGMacros-001.csv
        file_path = os.path.join(self.base_dir, f"CGMacros-{participant_id:03d}", f"CGMacros-{participant_id:03d}.csv")
        if not os.path.exists(file_path):
            return None
            
        df = pd.read_csv(file_path)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # Select key columns
        # Libre GL and Dexcom GL are often both present, we'll favor Libre or take mean
        df['glucose'] = df['Libre GL'].fillna(df['Dexcom GL'])
        
        # Interpolate missing glucose (1-minute intervals)
        df['glucose'] = df['glucose'].interpolate(method='linear')
        
        # Activity features: HR, Calories (Activity), METs
        activity_cols = ['HR', 'Calories (Activity)', 'METs']
        for col in activity_cols:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill').fillna(0)
            
        # Nutrition: Meal Type, Calories, Carbs, Protein, Fat, Fiber
        nutri_cols = ['Calories', 'Carbs', 'Protein', 'Fat', 'Fiber']
        for col in nutri_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
            
        return df

    def extract_ppgr_windows(self, df, window_minutes=120):
        """
        Extracts windows starting from 'Meal Type' events.
        """
        meal_indices = df[df['Meal Type'].notna()].index
        X, y = [], []
        
        for idx in meal_indices:
            # Post-meal window
            end_idx = idx + window_minutes
            if end_idx >= len(df):
                continue
                
            window_df = df.iloc[idx:end_idx]
            
            # Features: Macronutrients at start + Activity during window
            # Label: Glucose curve (PPGR) or Max Glucose change
            
            # Static Features (Nutrients)
            nutrients = df.iloc[idx][['Calories', 'Carbs', 'Protein', 'Fat', 'Fiber']].values
            
            # Dynamic Features (Activity) - Mean values during window (Fixed length: 3)
            activity_vals = []
            for c in ['HR', 'Calories (Activity)', 'METs']:
                if c in window_df.columns:
                    activity_vals.append(window_df[c].mean())
                else:
                    activity_vals.append(0.0)
            activity = np.array(activity_vals)
            
            # Baseline glucose (at meal start)
            baseline_g = df.iloc[idx]['glucose']
            
            # Target: Max glucose change or full curve
            max_g = window_df['glucose'].max()
            glucose_change = max_g - baseline_g
            
            feat_vector = np.concatenate([nutrients, activity, [baseline_g]])
            X.append(feat_vector)
            y.append(glucose_change)
            
        return np.array(X), np.array(y)

    def load_all_ppgr(self, num_participants=49):
        all_X, all_y = [], []
        for i in range(1, num_participants + 1):
            df = self.preprocess_participant(i)
            if df is not None:
                X, y = self.extract_ppgr_windows(df)
                if len(X) > 0:
                    all_X.append(X)
                    all_y.append(y)
        
        if not all_X:
            return None, None
            
        X_final = np.concatenate(all_X)
        y_final = np.concatenate(all_y)
        
        # Final scaling
        X_scaled = self.scaler.fit_transform(X_final)
        
        return X_scaled, y_final

if __name__ == "__main__":
    preprocessor = CGMacrosPreprocessor("f:/Diabetics Project/data/cgmacros/data_volume/CGMacros")
    X, y = preprocessor.load_all_ppgr(10) # Test with first 10
    print(f"Extracted {len(X)} PPGR samples. Feature shape: {X.shape}")
