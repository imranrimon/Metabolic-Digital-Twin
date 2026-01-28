import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from models import AttentionResNetRisk, STAttentionLSTM
from ppgr_model import PPGRModel
from recommender import DietRecommender
from preprocess import load_and_preprocess_100k
from sklearn.preprocessing import StandardScaler

class AdvancedDiabetesSystem:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.recommender = DietRecommender('f:/Diabetics Project/src/food_db.json')
        
        # Initialize Models
        self.risk_model = None
        self.forecast_model = None
        self.ppgr_model = None
        
    def load_models(self, risk_path, forecast_path, ppgr_path):
        # We need to know InputDim. For 100k it's 8.
        self.risk_model = AttentionResNetRisk(input_dim=8).to(self.device)
        self.risk_model.load_state_dict(torch.load(risk_path))
        self.risk_model.eval()
        
        self.forecast_model = STAttentionLSTM().to(self.device)
        self.forecast_model.load_state_dict(torch.load(forecast_path))
        self.forecast_model.eval()
        
        # PPGR model takes 9 inputs: Calories, Carbs, Protein, Fat, Fiber, HR, Cal_Act, METs, BaselineG
        self.ppgr_model = PPGRModel(input_dim=9).to(self.device)
        self.ppgr_model.load_state_dict(torch.load(ppgr_path))
        self.ppgr_model.eval()
        
    def run_analysis(self, user_tabular_data, user_glucose_sequence):
        """
        user_tabular_data: dict with health indicators
        user_glucose_sequence: list of 12 recent glucose readings (3 hours @ 15m)
        """
        # 1. Risk Prediction
        # Mocking scaler and encoding here for the demo
        feat_cols = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
        tab_df = pd.DataFrame([user_tabular_data])[feat_cols]
        tab_tensor = torch.FloatTensor(tab_df.values).to(self.device)
        
        with torch.no_grad():
            risk_logit = self.risk_model(tab_tensor)
            risk_prob = torch.sigmoid(risk_logit).item()
            
        # 2. Glucose Forecasting
        # Mocking normalization (actual system should use training scaler)
        seq_tensor = torch.FloatTensor(user_glucose_sequence).view(1, -1, 1).to(self.device)
        # Assuming seq is already normalized [0, 1]
        with torch.no_grad():
            forecast_val = self.forecast_model(seq_tensor).item()
            
        # 3. Diet Recommendation
        recommendation = self.recommender.recommend_meals(
            age=user_tabular_data['age'],
            bmi=user_tabular_data['bmi'],
            risk_level=risk_prob,
            gender=user_tabular_data['gender']
        )
        
        return {
            "prediction_type": "Multimodal GPU Pipeline (Phase 4)",
            "diabetes_risk_probability": round(float(risk_prob), 4),
            "status": "High Risk" if risk_prob > 0.5 else "Low Risk",
            "glucose_forecast_next_15m": round(forecast_val, 2),
            "diet_plan": recommendation
        }
        
    def predict_meal_impact(self, meal_macros, current_activity_level, baseline_glucose):
        """
        Predicts how a specific meal will spike blood glucose.
        meal_macros: [Calories, Carbs, Protein, Fat, Fiber]
        current_activity_level: [HR, Calories_Act, METs]
        """
        feat_vector = np.concatenate([meal_macros, current_activity_level, [baseline_glucose]])
        feat_tensor = torch.FloatTensor(feat_vector).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            spike_pred = self.ppgr_model(feat_tensor).item()
            
        return {
            "predicted_glucose_spike": round(spike_pred, 2),
            "estimated_peak_glucose": round(baseline_glucose + spike_pred, 2)
        }

if __name__ == "__main__":
    system = AdvancedDiabetesSystem()
    
    # Paths to trained models
    risk_p = 'f:/Diabetics Project/attention_resnet_risk.pth'
    fore_p = 'f:/Diabetics Project/st_attention_lstm.pth'
    ppgr_p = 'f:/Diabetics Project/ppgr_model.pth'
    
    try:
        system.load_models(risk_p, fore_p, ppgr_p)
        
        # Sample User
        tab_input = {
            "gender": 1, "age": 55, "hypertension": 1, "heart_disease": 0, 
            "smoking_history": 4, "bmi": 32.0, "HbA1c_level": 7.5, "blood_glucose_level": 180
        }
        # Normalized glucose sequence
        glucose_seq = [0.65] * 12 
        
        result = system.run_analysis(tab_input, glucose_seq)
        
        # PREDICT MEAL IMPACT
        # Example: Eating a high-carb meal (50g carbs) while sedentary
        meal = [500, 60, 20, 15, 5]
        activity = [70, 1.2, 1.0] # HR, CalAct, METs
        impact = system.predict_meal_impact(meal, activity, 110) # 110 baseline
        
        print("\n--- Advanced Integrated System Analysis ---")
        print(json.dumps(result, indent=2))
        print("\n--- Predictive Meal Impact Analysis ---")
        print(json.dumps(impact, indent=2))
        
    except Exception as e:
        print(f"System not fully ready: {e}. Ensure models are trained.")
