import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import sys
import os

# Add parent src to path to import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from models_sota import FTTransformerModel, NeuralCDEModel
from metabolic_rl import DQNAgent

app = FastAPI(title="Metabolic Digital Twin API")

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

@app.get("/")
def read_root():
    return {
        "message": "Metabolic Digital Twin API is Online",
        "endpoints": {
            "health": "/health",
            "predict_risk": "/predict/risk",
            "recommend_diet": "/recommend/diet"
        },
        "port": 8001
    }

@app.on_event("startup")
def load_models():
    # 1. FT-Transformer (Risk)
    try:
        model = FTTransformerModel(n_num_features=4, cat_cardinalities=[3, 6, 2, 2]).to(DEVICE)
        model.load_state_dict(torch.load("../../ft_transformer_risk.pth", map_location=DEVICE))
        model.eval()
        MODELS['risk'] = model
        print("Success: SOTA Risk Model loaded.")
    except Exception as e:
        print(f"Warning: Risk Model fallback active: {e}")

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
        agent.model.load_state_dict(torch.load("../../metabolic_policy.pth", map_location=DEVICE))
        agent.model.eval()
        MODELS['policy'] = agent
        print("Success: SOTA Policy Agent loaded.")
    except Exception as e:
        print(f"Warning: Policy Agent fallback active: {e}")

# --- API Schemas ---

class RiskRequest(BaseModel):
    num_features: List[float]
    cat_features: List[int]

class ForecastRequest(BaseModel):
    current_glucose: float

class DietRequest(BaseModel):
    current_glucose: float

# --- Endpoints ---

@app.post("/predict/risk")
async def predict_risk(req: RiskRequest):
    model = MODELS.get('risk')
    if not model:
        # Static demo fallback
        return {"risk_probability": 0.15, "status": "Low (Mock)"}
    
    with torch.no_grad():
        x_n = torch.FloatTensor(req.num_features).unsqueeze(0).to(DEVICE)
        x_c = torch.LongTensor(req.cat_features).unsqueeze(0).to(DEVICE)
        prob = torch.sigmoid(model(x_n, x_c)).item()
        
    return {"risk_probability": round(prob, 4), "status": "High" if prob > 0.5 else "Low"}

@app.post("/recommend/diet")
async def recommend_diet(req: DietRequest):
    agent = MODELS.get('policy')
    if not agent:
        return {"recommendation": "Balanced Meal", "reason": "Heuristic fallback active."}
    
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
