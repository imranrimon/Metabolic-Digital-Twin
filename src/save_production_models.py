"""
Save Production Models
Trains and serializes the best performing models for the API.
Focus: Optimized XGBoost (Rank 2, 97.9% AUC) - Robust and Fast.
"""

import pandas as pd
import joblib
import os
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from grandmaster_features import apply_grandmaster_features

def main():
    print("Training Production Models...")
    
    # 1. Load Data
    df = pd.read_csv('f:/Diabetics Project/data/diabetes-prediction-dataset/diabetes_prediction_dataset.csv')
    
    # Preprocessing for API consistency
    # The API will receive raw data, so we should emulate the pipeline
    # However, XGBoost handles encodings? No, we need to one-hot encode.
    # We'll build a pipeline that handles this? 
    # Actually, simpler to just start from the processed state for training,
    # and replicate preprocessing in API.
    
    df_processed = pd.get_dummies(df, drop_first=True)
    df_rich = apply_grandmaster_features(df_processed)
    
    X = df_rich.drop('diabetes', axis=1)
    y = df_rich['diabetes']
    
    # 2. Load Hyperparams
    try:
        params = joblib.load('f:/Diabetics Project/src/best_hyperparams.pkl')
        xgb_params = params['xgboost']
        print(f"Loaded XGB params: {xgb_params}")
    except:
        print("Using default XGB params")
        xgb_params = {'n_estimators': 500, 'learning_rate': 0.05}
        
    # 3. Train Full Model
    print("Fitting XGBoost on full dataset...")
    # We include scaler in the pipeline for safety? 
    # XGBoost is invariant to scaling largely, but interactions might not be?
    # Let's verify: interaction = age * bmi.
    # Providing raw features to XGBoost is fine.
    
    model = XGBClassifier(**xgb_params, n_jobs=-1, random_state=42, eval_metric='logloss')
    model.fit(X, y)
    
    # 4. Save
    os.makedirs('f:/Diabetics Project/models', exist_ok=True)
    model.save_model('f:/Diabetics Project/models/production_xgboost.json')
    joblib.dump(list(X.columns), 'f:/Diabetics Project/models/production_features.pkl')
    
    print("Saved: models/production_xgboost.json")
    print("Saved: models/production_features.pkl")

if __name__ == "__main__":
    main()
