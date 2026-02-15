
"""
Script to visualize the Recommender System output.
Demonstrates:
1. RL Policy Selection (Meal Type)
2. Food Knowledge Base Integration (Nutrition)
3. Structured Meal Plan Generation
"""

import sys
import os
import json
import pandas as pd

# Add src to path
sys.path.append(os.path.dirname(__file__))
from recommender import DietRecommender

def visualize_recommender():
    print("\n" + "="*60)
    print(" METABOLIC DIGITAL TWIN: AI DIETITIAN DEMO (RTX 8000)")
    print("="*60)
    
    # 1. Initialize Recommender
    try:
        # Check if db exists
        db_path = 'src/food_db.json'
        if not os.path.exists(db_path):
             # Fallback to absolute path or relative
             db_path = 'f:/Diabetics Project/src/food_db.json'
             
        recommender = DietRecommender(db_path)
        print("[+] Recommender System Initialized")
    except Exception as e:
        print(f"[!] Error initializing recommender: {e}")
        return

    # 2. Interactive User Profile
    print("\n" + "-"*30)
    print(" ENTER PATIENT METRICS")
    print("-" * 30)
    
    try:
        glucose = float(input("Current Glucose (mg/dL) [e.g. 160]: ") or 160)
        weight = float(input("Weight (kg) [e.g. 85]: ") or 85)
        height = float(input("Height (cm) [e.g. 175]: ") or 175)
        age = int(input("Age [e.g. 45]: ") or 45)
        gender_input = input("Gender (m/f) [default m]: ").lower()
        gender = 'male' if gender_input.startswith('m') or gender_input == '' else 'female'
    except ValueError:
        print("[!] Invalid input. Using default values.")
        glucose, weight, height, age, gender = 160, 85, 175, 45, 'male'

    # BMI Calc
    height_m = height / 100
    bmi = weight / (height_m ** 2)

    user_state = {
        'glucose': glucose,
        'weight': weight,
        'height': height,
        'age': age,
        'activity_level': 'sedentary',
        'gender': gender,
        'bmi': round(bmi, 1)
    }
    
    print("\n[Patient Profile]")
    print(json.dumps(user_state, indent=2))
    
    # 3. Generate Recommendation
    print("\n[AI Generating Plan...]")
    
    # Use the same logic as the API
    # 1. Calorie Calculation
    bmr = 10 * user_state['weight'] + 6.25 * user_state['height'] - 5 * user_state['age'] + 5
    tdee = bmr * 1.2 # Sedentary
    target_calories = int(tdee - 500) # Deficit for weight loss
    
    # 2. Meal Plan Generation
    risk_level = 0.8 if user_state['glucose'] > 140 else 0.2
    
    # recommend_meals(self, age, bmi, risk_level, gender, weight=70, height=170)
    plan = recommender.recommend_meals(
        age=user_state['age'],
        bmi=user_state['bmi'],
        risk_level=risk_level,
        gender=1 if user_state['gender'] == 'male' else 0,
        weight=user_state['weight'],
        height=user_state['height']
    )
    
    # 4. Visualize Output
    strategy = "Low Carb" if risk_level > 0.5 else "Balanced"
    print(f"\n[Strategy Selected]: {strategy} (Due to Glucose {user_state['glucose']} mg/dL)")
    print(f"[Target Calories]: {target_calories} kcal/day")
    
    # Display Meals
    print("\n" + "-"*60)
    print(f"{'Meal':<12} | {'Food Item':<30} | {'Cal':<6} | {'GI':<4}")
    print("-"*60)
    
    meals = plan.get('suggested_meals', {})
    
    for meal_name in ['Breakfast', 'Lunch', 'Dinner', 'Snack']:
        items = meals.get(meal_name, [])
        if not items:
            print(f"{meal_name:<12} | {'No recommendation':<30} | {'-':<6} | {'-':<4}")
            continue
            
        for i, item in enumerate(items[:2]):
            name = item['food'][:28] # Truncate
            cal = str(item.get('calories', '?'))
            gi = str(item.get('gi', '?'))
            label = meal_name if i == 0 else ""
            print(f"{label:<12} | {name:<30} | {cal:<6} | {gi:<4}")
            
    print("-"*60)
    
    print("\n[Macro Distribution]")
    print(f"Carbs: {plan.get('macros', {}).get('carbs', '?')}")
    print(f"Protein: {plan.get('macros', {}).get('protein', '?')}")
    print(f"Fats: {plan.get('macros', {}).get('fats', '?')}")

if __name__ == "__main__":
    visualize_recommender()
