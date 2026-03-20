
import sys
import os
import torch
import uvicorn
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from xgboost import XGBClassifier

BACKEND_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, "../.."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Add parent src to path to import models
sys.path.append(SRC_DIR)

from conformal import load_conformal_classifier
from models_sota import NeuralCDEModel
from recommender import DietRecommender
from risk_pipeline import prepare_risk_inference_features

app = FastAPI(title="Metabolic Digital Twin API (Prototype)")

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
CATEGORY_LEVELS = {}
CONFORMAL = None
CONFORMAL_SUMMARY = {}
CONFORMAL_METHOD = None

# Initialize Recommender
recommender = DietRecommender(os.path.join(SRC_DIR, "food_db.json"))

@app.get("/")
def read_root():
    return {
        "message": "Metabolic Digital Twin API is online (prototype config)",
        "endpoints": {
            "health": "/health",
            "predict_risk": "/predict/risk",
            "recommend_diet": "/recommend/diet"
        },
        "port": 8001
    }

@app.on_event("startup")
def load_models():
    global FEATURES, CATEGORY_LEVELS, CONFORMAL, CONFORMAL_SUMMARY, CONFORMAL_METHOD

    # 1. Production XGBoost risk model
    try:
        model = XGBClassifier()
        model.load_model(os.path.join(MODELS_DIR, "production_xgboost.json"))
        MODELS['risk'] = model
        
        # Load feature names to ensure correct order
        preprocess_path = os.path.join(MODELS_DIR, "production_preprocess.pkl")
        if os.path.exists(preprocess_path):
            preprocess_artifact = joblib.load(preprocess_path)
            FEATURES = preprocess_artifact["feature_columns"]
            CATEGORY_LEVELS = preprocess_artifact.get("category_levels", {})
        else:
            FEATURES = joblib.load(os.path.join(MODELS_DIR, "production_features.pkl"))
            CATEGORY_LEVELS = {}
        print("Success: production XGBoost risk model loaded.")

        conformal_path = os.path.join(MODELS_DIR, "production_conformal.pkl")
        if os.path.exists(conformal_path):
            conformal_artifact = joblib.load(conformal_path)
            CONFORMAL = load_conformal_classifier(conformal_artifact)
            CONFORMAL_SUMMARY = conformal_artifact.get("summary", {})
            CONFORMAL_METHOD = conformal_artifact.get("method", CONFORMAL_SUMMARY.get("method"))
            print("Success: conformal calibration artifact loaded.")
        else:
            print("Notice: conformal calibration artifact not found.")
    except Exception as e:
        print(f"Warning: XGBoost Risk Model fallback active: {e}")

    # 2. Neural CDE trend model
    try:
        # Fallback to LSTM if CDE weights missing
        weights_path = os.path.join(PROJECT_ROOT, "neural_cde_glucose.pth")
        if os.path.exists(weights_path):
            model = NeuralCDEModel(input_channels=2, hidden_channels=32, output_channels=1).to(DEVICE)
            model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
            model.eval()
            MODELS['trend'] = model
            print("Success: trend model loaded.")
        else:
            print("Notice: Trend Model using heuristic fallback.")
    except Exception as e:
        print(f"Warning: Trend Model error: {e}")

    # 3. RL Policy (Diet)
    try:
        from metabolic_rl import DQNAgent
        agent = DQNAgent(state_dim=1, action_dim=5)
        # Check specific path or generic
        path = os.path.join(PROJECT_ROOT, "metabolic_policy.pth")
        if os.path.exists(path):
            agent.model.load_state_dict(torch.load(path, map_location=DEVICE))
            agent.model.eval()
            MODELS['policy'] = agent
            print("Success: policy agent loaded.")
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
    age: Optional[float] = 45
    bmi: Optional[float] = 25
    gender: Optional[int] = 1 # 1 Male, 0 Female

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

        # 2. Preprocess and align with the saved production schema
        df_final = prepare_risk_inference_features(df, FEATURES, CATEGORY_LEVELS)

        # 3. Predict
        probs = model.predict_proba(df_final.values)[0]
        prob = float(probs[1])

        response = {
            "risk_probability": round(float(prob), 4),
            "status": "High" if prob > 0.5 else "Low", 
            "model": "XGBoost risk model"
        }

        if CONFORMAL is not None:
            conformal_details = CONFORMAL.predict_details_from_probabilities([probs])[0]
            response["conformal"] = {
                "method": CONFORMAL_METHOD,
                "prediction_set": conformal_details["prediction_set"],
                "status": conformal_details["status"],
                "p_values": {
                    label: round(float(value), 4)
                    for label, value in conformal_details["p_values"].items()
                },
                "scores": {
                    label: round(float(value), 4)
                    for label, value in conformal_details.get("scores", {}).items()
                },
                "alpha": round(float(CONFORMAL.alpha), 4),
                "target_coverage": round(float(1 - CONFORMAL.alpha), 4),
                "summary": {
                    key: value if isinstance(value, str) else round(float(value), 4)
                    for key, value in CONFORMAL_SUMMARY.items()
                    if isinstance(value, (int, float, str))
                },
            }
            if "ranked_labels" in conformal_details:
                response["conformal"]["ranked_labels"] = conformal_details["ranked_labels"]
            if "threshold" in conformal_details:
                response["conformal"]["threshold"] = round(float(conformal_details["threshold"]), 4)

        return response
    except Exception as e:
        return {"error": str(e)}

@app.post("/recommend/diet")
async def recommend_diet(req: DietRequest):
    # 1. RL Policy Selection (Meal Type)
    agent = MODELS.get('policy')
    meal_type_rec = "Balanced"
    if agent:
        action_idx = agent.act([req.current_glucose])
        meal_types = ["Low Carb", "Balanced", "Moderate Carb", "High Fiber", "Custom Protocol"]
        meal_type_rec = meal_types[action_idx]
    
    return {
        "strategy": meal_type_rec,
        "reason": f"RL Agent selected '{meal_type_rec}' based on glucose {req.current_glucose}mg/dL."
    }

@app.post("/recommend/meals")
async def recommend_meals(req: DietRequest):
    # Content-Based Food Recommendation
    risk_level = 0.8 if req.current_glucose > 140 else 0.2
    
    plan = recommender.recommend_meals(
        age=req.age, 
        bmi=req.bmi, 
        risk_level=risk_level, 
        gender=req.gender
    )
    
    return {
        "caloric_target": plan['daily_caloric_target'],
        "meals": plan['suggested_meals'],
        "macros": { # Estimated distribution based on standard diabetic diet
            "carbs": "45%",
            "protein": "25%",
            "fat": "30%"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "gpu": torch.cuda.is_available(),
        "models": list(MODELS.keys()),
        "conformal_available": CONFORMAL is not None,
        "conformal_method": CONFORMAL_METHOD,
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
