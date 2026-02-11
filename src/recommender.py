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
            
        # Target distribution (approx)
        targets = {
            "Breakfast": daily_cals * 0.25,
            "Lunch": daily_cals * 0.35,
            "Dinner": daily_cals * 0.30,
            "Snack": daily_cals * 0.10
        }
        
        recommendations = {
            "Breakfast": [],
            "Lunch": [],
            "Dinner": [],
            "Snack": []
        }
        
        # Helper to categorize based on name (since 'category' is generic)
        def get_meal_type(name):
            name = name.lower()
            if any(x in name for x in ['oat', 'egg', 'yogurt', 'milk', 'toast', 'berry', 'apple', 'banana', 'coffee']):
                return 'Breakfast'
            if any(x in name for x in ['chicken', 'beef', 'fish', 'salmon', 'rice', 'pasta', 'salad', 'soup', 'sandwich', 'burger', 'steak']):
                return 'Lunch/Dinner'
            if any(x in name for x in ['nut', 'almond', 'walnut', 'fruit', 'bar', 'chip', 'cookie', 'corn']):
                return 'Snack'
            return 'Lunch/Dinner' # Default to main meal
            
        # Sort by GI (lower first for diabetics)
        sorted_food = sorted(self.food_db, key=lambda x: x['gi'])
        
        for item in sorted_food:
            m_type = get_meal_type(item['food'])
            
            if m_type == 'Breakfast':
                if len(recommendations['Breakfast']) < 2:
                    recommendations['Breakfast'].append(item)
            elif m_type == 'Lunch/Dinner':
                if len(recommendations['Lunch']) < 2:
                    recommendations['Lunch'].append(item)
                elif len(recommendations['Dinner']) < 2:
                    recommendations['Dinner'].append(item)
            elif m_type == 'Snack':
                if len(recommendations['Snack']) < 2:
                    recommendations['Snack'].append(item)
                    
        return {
            "daily_caloric_target": round(daily_cals),
            "suggested_meals": recommendations
        }

if __name__ == "__main__":
    recommender = DietRecommender('f:/Diabetics Project/src/food_db.json')
    # Test for a high-risk obese male
    sample_rec = recommender.recommend_meals(age=45, bmi=30, risk_level=0.8, gender=1)
    print(json.dumps(sample_rec, indent=2))
