"""Backward-compatible wrapper for the organized package layout."""

from metabolic_twin.recommendation.heuristic import *  # noqa: F401,F403


if __name__ == "__main__":
    recommender = DietRecommender("f:/Diabetics Project/src/food_db.json")
    sample_rec = recommender.recommend_meals(age=45, bmi=30, risk_level=0.8, gender=1)
    import json

    print(json.dumps(sample_rec, indent=2))

