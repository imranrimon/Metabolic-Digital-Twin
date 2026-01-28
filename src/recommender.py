import json
import os

class DietRecommender:
    def __init__(self, db_path):
        with open(db_path, 'r') as f:
            self.food_db = json.load(f)
            
    def calculate_calories(self, age, weight, height, gender, activity_level=1.2):
        # Harris-Benedict Equation for BMR
        if gender == 1: # Male
            bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
        else: # Female
            bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
        return bmr * activity_level

    def recommend_meals(self, age, bmi, risk_level, gender, weight=70, height=170):
        daily_cals = self.calculate_calories(age, weight, height, gender)
        
        # Adjustment based on BMI and Risk
        if bmi > 25 or risk_level > 0.5:
            daily_cals *= 0.85 # Deficit for weight management
            
        recommendations = {
            "Breakfast": [],
            "Lunch": [],
            "Dinner": [],
            "Snack": []
        }
        
        # Simple content-based filtering (logic: lower GI is better if risk is high)
        sorted_food = sorted(self.food_db, key=lambda x: x['gi'])
        
        for item in sorted_food:
            cat = item['category']
            if cat in recommendations and len(recommendations[cat]) < 2:
                recommendations[cat].append(item)
                
        return {
            "daily_caloric_target": round(daily_cals),
            "suggested_meals": recommendations
        }

if __name__ == "__main__":
    recommender = DietRecommender('f:/Diabetics Project/src/food_db.json')
    # Test for a high-risk obese male
    sample_rec = recommender.recommend_meals(age=45, bmi=30, risk_level=0.8, gender=1)
    print(json.dumps(sample_rec, indent=2))
