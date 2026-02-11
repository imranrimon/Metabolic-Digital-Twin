
import requests
import json
import time

def test_api():
    base_url = "http://localhost:8001"
    
    # 1. Test Risk Prediction
    print("Testing Risk Prediction...")
    try:
        risk_res = requests.post(f"{base_url}/predict/risk", json={
            "gender": "Female", "age": 55.0, "hypertension": 1, 
            "heart_disease": 0, "smoking_history": "past", 
            "bmi": 32.0, "HbA1c_level": 7.0, "blood_glucose_level": 200
        })
        print("Risk Response:", json.dumps(risk_res.json(), indent=2))
    except Exception as e: print(e)

    # 2. Test Diet Strategy (Legacy/Wrapper)
    print("\nTesting Diet Strategy...")
    try:
        diet_res = requests.post(f"{base_url}/recommend/diet", json={
            "current_glucose": 150.0,
            "age": 45, "bmi": 28.0, "gender": 1
        })
        print("Strategy:", diet_res.json().get('strategy'))
    except Exception as e: print(e)

    # 3. Test Dedicated Meal Endpoint
    print("\nTesting Dedicated Meal Endpoint...")
    try:
        meal_res = requests.post(f"{base_url}/recommend/meals", json={
            "current_glucose": 150.0,
            "age": 45, "bmi": 28.0, "gender": 1
        })
        data = meal_res.json()
        print("Caloric Target:", data.get('caloric_target'))
        print("Macros:", data.get('macros'))
        meals = data.get('meals', {})
        print(f"Meal Counts: Breakfast={len(meals.get('Breakfast', []))}, Lunch={len(meals.get('Lunch', []))}")
        
        if len(meals.get('Breakfast', [])) > 0 and 'macros' in data:
            print("SUCCESS: Dedicated meal endpoint works!")
        else:
            print("WARNING: Meal plan empty or macros missing.")
            
    except Exception as e: print(e)

if __name__ == "__main__":
    test_api()
