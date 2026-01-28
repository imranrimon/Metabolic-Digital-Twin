import joblib
import os
import pandas as pd
import numpy as np
from preprocess import load_and_preprocess_100k
from recommender import DietRecommender
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

class DiabetesPredictionSystem:
    def __init__(self):
        self.data_dir = "f:/Diabetics Project/data"
        self.recommender = DietRecommender('f:/Diabetics Project/src/food_db.json')
        self.model = None
        self.scaler = StandardScaler()
        
    def train_final_model(self):
        print("Training final integration model (XGBoost)...")
        from preprocess import get_processed_data
        from imblearn.over_sampling import SMOTE
        
        X_train, X_test, y_train, y_test, self.cols = get_processed_data('100k')
        smote = SMOTE(random_state=42)
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
        
        self.model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')
        self.model.fit(X_train_sm, y_train_sm)
        print("Final model trained.")

    def run_inference(self, user_data):
        """
        user_data: dict with keys matching the 100k dataset features
        """
        if self.model is None:
            self.train_final_model()
            
        # Convert user_data to DataFrame
        df = pd.DataFrame([user_data])
        
        # Preprocess (similar to load_and_preprocess_100k)
        # Note: In a real app, we'd persist the LabelEncoder and Scaler
        # For this demo, we'll assume the user provides pre-encoded categorical values
        
        # Feature order must match the model's expected order
        # Feature list for 100k: gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level
        
        features = df.values # Simple mockup
        risk_prob = self.model.predict_proba(features)[0][1]
        
        # Get recommendations
        recommendation = self.recommender.recommend_meals(
            age=user_data['age'],
            bmi=user_data['bmi'],
            risk_level=risk_prob,
            gender=user_data['gender']
        )
        
        return {
            "diabetes_risk_probability": round(float(risk_prob), 4),
            "status": "High Risk" if risk_prob > 0.5 else "Low Risk",
            "diet_plan": recommendation
        }

if __name__ == "__main__":
    system = DiabetesPredictionSystem()
    
    # Mock user input (Male, 55, Has Hypertension, Non-smoker, BMI 32, HbA1c 7.5, Glucose 180)
    # Encoded: gender=1, age=55, hypertension=1, heart_disease=0, smoking_history=4 (never), bmi=32, hba1c=7.5, glucose=180
    user_input = {
        "gender": 1,
        "age": 55,
        "hypertension": 1,
        "heart_disease": 0,
        "smoking_history": 4, 
        "bmi": 32.0,
        "HbA1c_level": 7.5,
        "blood_glucose_level": 180
    }
    
    result = system.run_inference(user_input)
    print("\n--- System Prediction & Diet Plan ---")
    import json
    print(json.dumps(result, indent=2))
