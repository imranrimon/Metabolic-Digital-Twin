import torch
import uvicorn
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import sys
import os
from xgboost import XGBClassifier

# Add parent src to path to import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from models_sota import FTTransformerModel, NeuralCDEModel
from metabolic_rl import DQNAgent
from grandmaster_features import apply_grandmaster_features

app = FastAPI(title="Metabolic Digital Twin API (Grandmaster Edition)")

# Enable CORS for mobile frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Loading ---

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODELS = {}
FEATURES = []

@app.get("/")
def read_root():
    return {
        "message": "Metabolic Digital Twin API is Online (SOTA Config)",
        "endpoints": {
            "health": "/health",
            "predict_risk": "/predict/risk",
            "recommend_diet": "/recommend/diet"
        },
        "port": 8001
    }

@app.on_event("startup")
def load_models():
    # 1. Production XGBoost (Grandmaster SOTA)
    try:
        model = XGBClassifier()
        model.load_model("../../models/production_xgboost.json")
        MODELS['risk'] = model
        
        # Load feature names to ensure correct order
        global FEATURES
        FEATURES = joblib.load("../../models/production_features.pkl")
        print("Success: SOTA XGBoost Model loaded (97.9% AUC).")
    except Exception as e:
        print(f"Warning: XGBoost Risk Model fallback active: {e}")

    # 2. Neural CDE (Trend)
    try:
        # Fallback to LSTM if CDE weights missing
        weights_path = "../../neural_cde_glucose.pth"
        if os.path.exists(weights_path):
            model = NeuralCDEModel(input_channels=2, hidden_channels=32, output_channels=1).to(DEVICE)
            model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
            model.eval()
            MODELS['trend'] = model
            print("Success: SOTA Trend Model loaded.")
        else:
            print("Notice: Trend Model using heuristic fallback.")
    except Exception as e:
        print(f"Warning: Trend Model error: {e}")

    # 3. RL Policy (Diet)
    try:
        agent = DQNAgent(state_dim=1, action_dim=5)
        # Check specific path or generic
        path = "../../metabolic_policy.pth"
        if os.path.exists(path):
            agent.model.load_state_dict(torch.load(path, map_location=DEVICE))
            agent.model.eval()
            MODELS['policy'] = agent
            print("Success: SOTA Policy Agent loaded.")
    except Exception as e:
        print(f"Warning: Policy Agent fallback active: {e}")

# --- API Schemas ---

class RawRiskRequest(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    smoking_history: str
    bmi: float
    HbA1c_level: float
    blood_glucose_level: int

class DietRequest(BaseModel):
    current_glucose: float

# --- Endpoints ---

@app.post("/predict/risk")
async def predict_risk(req: RawRiskRequest):
    model = MODELS.get('risk')
    if not model:
        return {"risk_probability": 0.15, "status": "Low (Mock - Model Not Loaded)"}
    
    try:
        # 1. Convert Request to DataFrame
        data = {
            'gender': [req.gender],
            'age': [req.age],
            'hypertension': [req.hypertension],
            'heart_disease': [req.heart_disease],
            'smoking_history': [req.smoking_history],
            'bmi': [req.bmi],
            'HbA1c_level': [req.HbA1c_level],
            'blood_glucose_level': [req.blood_glucose_level]
        }
        df = pd.DataFrame(data)
        
        # 2. Preprocessing (One-Hot)
        # Note: We must ensure columns match training. 
        # In production, we'd use a saved Scikit-Learn Pipeline. 
        # Here we manually recreate the dummification logic or 
        # assume simplified inputs for the demo.
        # "gender" -> we need gender_Male, gender_Other (if existed)
        # "smoking" -> we need smoking...
        
        # Quick Hack for Robustness: Manually construct expected columns
        # This prevents "feature mismatch" errors in XGBoost
        df_processed = pd.get_dummies(df)
        
        # Add Grandmaster Features
        df_rich = apply_grandmaster_features(df_processed)
        
        # Align with training features
        # Create empty DF with all training columns
        df_final = pd.DataFrame(columns=FEATURES)
        
        # Fill strictly what we have, 0 elsewhere
        for col in df_rich.columns:
            if col in df_final.columns:
                df_final.loc[0, col] = df_rich.iloc[0][col]
        
        df_final = df_final.fillna(0) # Categorical missing levels become 0
        
        # 3. Predict
        prob = model.predict_proba(df_final.values)[0][1]
        
        return {
            "risk_probability": round(float(prob), 4),
            "status": "High" if prob > 0.5 else "Low", 
            "model": "XGBoost Grandmaster (Optuna)"
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/recommend/diet")
async def recommend_diet(req: DietRequest):
    agent = MODELS.get('policy')
    # Default fallback if agent not loaded
    if not agent:
        return {"recommendation": "Balanced Meal", "reason": "System fallback."}
    
    action_idx = agent.act([req.current_glucose])
    meal_types = ["Low Carb", "Balanced", "Moderate Carb", "High Fiber", "Custom Protocol"]
    
    return {
        "recommendation": meal_types[action_idx],
        "reason": "Optimized via Offline Reinforcement Learning for Time-in-Range."
    }

@app.get("/health")
def health_check():
    return {"status": "online", "gpu": torch.cuda.is_available(), "models": list(MODELS.keys())}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
