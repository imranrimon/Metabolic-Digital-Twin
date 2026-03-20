from dataclasses import dataclass
from itertools import combinations

import numpy as np

from metabolic_twin.recommendation.heuristic import DietRecommender
from metabolic_twin.recommendation.rl import MetabolicEnv


MEAL_ORDER = ["Breakfast", "Lunch", "Dinner", "Snack"]
MEAL_SHARE = {
    "Breakfast": 0.25,
    "Lunch": 0.35,
    "Dinner": 0.30,
    "Snack": 0.10,
}
MEAL_SLOTS = 2
FEATURE_KEYS = ["calories", "carbs", "protein", "fat", "gi"]


def infer_meal_type(name):
    name = name.lower()
    if any(token in name for token in ["oat", "egg", "yogurt", "milk", "toast", "berry", "apple", "banana", "coffee"]):
        return "Breakfast"
    if any(
        token in name
        for token in ["chicken", "beef", "fish", "salmon", "rice", "pasta", "salad", "soup", "sandwich", "burger", "steak"]
    ):
        return "Lunch/Dinner"
    if any(token in name for token in ["nut", "almond", "walnut", "fruit", "bar", "chip", "cookie", "corn"]):
        return "Snack"
    return "Lunch/Dinner"


def meal_action_from_items(items):
    carbs = sum(item["carbs"] for item in items)
    protein = sum(item["protein"] for item in items)
    fat = sum(item["fat"] for item in items)
    return [
        float(np.clip(carbs / 60.0, 0.0, 1.2)),
        float(np.clip(protein / 35.0, 0.0, 1.0)),
        float(np.clip(fat / 25.0, 0.0, 1.0)),
    ]


@dataclass
class NextGenPlannerConfig:
    use_graph: bool = True
    use_state_rerank: bool = True
    beam_width: int = 26
    risk_adjustment_strength: float = 1.0


class GuidelineGraphTwinRecommender(DietRecommender):
    def __init__(self, db_path, config=None):
        super().__init__(db_path)
        self.config = config or NextGenPlannerConfig()
        self.food_db = self._deduplicate_food_db(self.food_db)
        self.meal_types = [infer_meal_type(item["food"]) for item in self.food_db]
        self.feature_matrix = self._build_feature_matrix(self.food_db)
        self.normalized_feature_matrix = self._normalize_rows(self.feature_matrix)
        self.similarity_matrix = self.normalized_feature_matrix @ self.normalized_feature_matrix.T

    def _deduplicate_food_db(self, food_db):
        seen = set()
        unique_items = []
        for item in food_db:
            key = (
                item["food"].strip().lower(),
                round(float(item["calories"]), 2),
                round(float(item["carbs"]), 2),
                round(float(item["protein"]), 2),
                round(float(item["fat"]), 2),
                round(float(item["gi"]), 2),
            )
            if key in seen:
                continue
            seen.add(key)
            unique_items.append(item)
        return unique_items

    def _build_feature_matrix(self, food_db):
        matrix = np.array([[float(item[key]) for key in FEATURE_KEYS] for item in food_db], dtype=np.float32)
        means = matrix.mean(axis=0, keepdims=True)
        stds = matrix.std(axis=0, keepdims=True)
        stds[stds < 1e-6] = 1.0
        return (matrix - means) / stds

    def _normalize_rows(self, matrix):
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1.0
        return matrix / norms

    def _daily_calorie_target(self, age, bmi, risk_level, gender, weight, height):
        daily_cals = self.calculate_calories(age=age, weight=weight, height=height, gender=gender)
        bmi_excess = max(bmi - 24.0, 0.0) / 16.0
        risk_excess = max(risk_level - 0.35, 0.0)
        adjustment = 1.0 - self.config.risk_adjustment_strength * ((0.11 * bmi_excess) + (0.18 * risk_excess))
        adjustment = float(np.clip(adjustment, 0.76, 1.02))
        return float(daily_cals * adjustment)

    def _macro_profile(self, risk_level, glucose):
        severity = max(float(risk_level), np.clip((float(glucose) - 95.0) / 110.0, 0.0, 1.0))
        if severity >= 0.72:
            return {"carb_ratio": 0.16, "protein_ratio": 0.29, "fat_ratio": 0.55, "gi_ceiling": 35.0}
        if severity >= 0.45:
            return {"carb_ratio": 0.22, "protein_ratio": 0.29, "fat_ratio": 0.49, "gi_ceiling": 42.0}
        return {"carb_ratio": 0.28, "protein_ratio": 0.27, "fat_ratio": 0.45, "gi_ceiling": 50.0}

    def _meal_targets(self, age, bmi, risk_level, glucose, gender, weight, height):
        daily_cals = self._daily_calorie_target(age, bmi, risk_level, gender, weight, height)
        profile = self._macro_profile(risk_level, glucose)
        meal_targets = {}
        meal_gi_offset = {"Breakfast": -4.0, "Lunch": 0.0, "Dinner": -2.0, "Snack": -8.0}
        carb_scale = {"Breakfast": 0.90, "Lunch": 1.00, "Dinner": 0.90, "Snack": 0.55}
        protein_scale = {"Breakfast": 1.05, "Lunch": 1.00, "Dinner": 1.00, "Snack": 0.85}
        fat_scale = {"Breakfast": 1.00, "Lunch": 1.00, "Dinner": 1.00, "Snack": 1.10}

        for meal_name in MEAL_ORDER:
            meal_cals = daily_cals * MEAL_SHARE[meal_name]
            base_carbs = (meal_cals * profile["carb_ratio"]) / 4.0
            base_protein = (meal_cals * profile["protein_ratio"]) / 4.0
            base_fat = (meal_cals * profile["fat_ratio"]) / 9.0
            meal_targets[meal_name] = {
                "calories": meal_cals,
                "carbs": base_carbs * carb_scale[meal_name],
                "protein": base_protein * protein_scale[meal_name],
                "fat": base_fat * fat_scale[meal_name],
                "gi_ceiling": max(profile["gi_ceiling"] + meal_gi_offset[meal_name], 18.0),
            }
        return daily_cals, meal_targets

    def _candidate_indices(self, meal_name, target):
        if meal_name in {"Lunch", "Dinner"}:
            allowed = {"Lunch/Dinner"}
        else:
            allowed = {meal_name}

        target_per_item_cals = target["calories"] / MEAL_SLOTS
        candidate_indices = [
            index
            for index, item in enumerate(self.food_db)
            if self.meal_types[index] in allowed
            and item["calories"] >= max(25.0, 0.18 * target_per_item_cals)
            and item["calories"] <= 2.6 * target_per_item_cals
            and item["gi"] <= target["gi_ceiling"] + 16.0
        ]

        if len(candidate_indices) < 12:
            candidate_indices = [
                index
                for index, item in enumerate(self.food_db)
                if self.meal_types[index] in allowed and item["calories"] <= 3.2 * target_per_item_cals
            ]
        return candidate_indices

    def _base_item_score(self, item, target):
        per_item_cals = target["calories"] / MEAL_SLOTS
        per_item_carbs = target["carbs"] / MEAL_SLOTS
        per_item_protein = target["protein"] / MEAL_SLOTS
        calorie_penalty = abs(item["calories"] - per_item_cals) / max(per_item_cals, 1.0)
        carb_penalty = abs(item["carbs"] - per_item_carbs) / max(per_item_carbs + 4.0, 4.0)
        protein_shortfall = max(per_item_protein - item["protein"], 0.0) / max(per_item_protein, 1.0)
        gi_penalty = max(item["gi"] - target["gi_ceiling"], 0.0) / max(target["gi_ceiling"], 1.0)
        return calorie_penalty + carb_penalty + (1.15 * protein_shortfall) + (1.35 * gi_penalty)

    def _pair_metrics(self, pair_indices):
        totals = {key: 0.0 for key in FEATURE_KEYS}
        for index in pair_indices:
            item = self.food_db[index]
            for key in FEATURE_KEYS:
                totals[key] += float(item[key])
        return totals

    def _graph_penalty(self, pair_indices, used_indices):
        if not self.config.use_graph:
            return 0.0

        pair_similarity = float(self.similarity_matrix[pair_indices[0], pair_indices[1]])
        reuse_penalty = 0.0
        if used_indices:
            reuse_penalty = float(np.mean(self.similarity_matrix[list(pair_indices), :][:, list(used_indices)]))
        return 0.05 * max(pair_similarity, 0.0) + (0.08 * max(reuse_penalty, 0.0))

    def _state_penalty(self, current_glucose, pair_indices):
        if not self.config.use_state_rerank:
            return 0.0

        env = MetabolicEnv()
        env.state = float(current_glucose)
        items = [self.food_db[index] for index in pair_indices]
        next_state, reward, _, _ = env.step(meal_action_from_items(items))
        next_glucose = float(next_state[0])
        glucose_penalty = abs(next_glucose - 116.0) / 24.0
        range_penalty = max(next_glucose - 140.0, 0.0) / 12.0
        return glucose_penalty + range_penalty - (reward / 15.0)

    def _pair_score(self, pair_indices, target, current_glucose, used_indices):
        totals = self._pair_metrics(pair_indices)
        calorie_error = abs(totals["calories"] - target["calories"]) / max(target["calories"], 1.0)
        carb_error = abs(totals["carbs"] - target["carbs"]) / max(target["carbs"] + 6.0, 6.0)
        protein_shortfall = max(target["protein"] - totals["protein"], 0.0) / max(target["protein"], 1.0)
        fat_error = abs(totals["fat"] - target["fat"]) / max(target["fat"] + 4.0, 4.0)
        gi_penalty = max((totals["gi"] / MEAL_SLOTS) - target["gi_ceiling"], 0.0) / max(target["gi_ceiling"], 1.0)
        graph_penalty = self._graph_penalty(pair_indices, used_indices)
        state_penalty = self._state_penalty(current_glucose, pair_indices)
        return (
            (1.1 * calorie_error)
            + (1.2 * carb_error)
            + (1.35 * protein_shortfall)
            + (0.55 * fat_error)
            + (1.85 * gi_penalty)
            + graph_penalty
            + state_penalty
        )

    def _select_meal_pair(self, meal_name, target, current_glucose, used_indices):
        candidate_indices = self._candidate_indices(meal_name, target)
        candidate_indices = sorted(candidate_indices, key=lambda idx: self._base_item_score(self.food_db[idx], target))
        candidate_indices = candidate_indices[: self.config.beam_width]

        best_pair = None
        best_score = float("inf")
        for pair_indices in combinations(candidate_indices, 2):
            if any(index in used_indices for index in pair_indices):
                continue
            score = self._pair_score(pair_indices, target, current_glucose, used_indices)
            if score < best_score:
                best_score = score
                best_pair = pair_indices

        if best_pair is None:
            filtered = [index for index in candidate_indices if index not in used_indices]
            if len(filtered) >= 2:
                best_pair = tuple(filtered[:2])
            elif filtered:
                best_pair = (filtered[0], filtered[0])
            else:
                best_pair = (candidate_indices[0], candidate_indices[1])

        return list(best_pair), float(best_score)

    def recommend_meals(self, age, bmi, risk_level, gender, weight=70, height=170, glucose=110.0):
        daily_cals, meal_targets = self._meal_targets(age, bmi, risk_level, glucose, gender, weight, height)
        recommendations = {meal: [] for meal in MEAL_ORDER}
        strategy_trace = []
        rationale = {}
        used_indices = set()
        env = MetabolicEnv()
        env.state = float(glucose)

        for meal_name in MEAL_ORDER:
            pair_indices, pair_score = self._select_meal_pair(
                meal_name=meal_name,
                target=meal_targets[meal_name],
                current_glucose=env.state,
                used_indices=used_indices,
            )
            items = [self.food_db[index] for index in pair_indices]
            recommendations[meal_name] = items
            used_indices.update(pair_indices)
            env.step(meal_action_from_items(items))
            strategy_trace.append(
                f"{meal_name}: carb<={meal_targets[meal_name]['carbs']:.1f}g, gi<={meal_targets[meal_name]['gi_ceiling']:.0f}"
            )
            rationale[meal_name] = {
                "target_calories": round(meal_targets[meal_name]["calories"], 1),
                "target_carbs": round(meal_targets[meal_name]["carbs"], 1),
                "target_protein": round(meal_targets[meal_name]["protein"], 1),
                "target_fat": round(meal_targets[meal_name]["fat"], 1),
                "target_gi_ceiling": round(meal_targets[meal_name]["gi_ceiling"], 1),
                "pair_score": round(pair_score, 4),
            }

        return {
            "daily_caloric_target": round(daily_cals),
            "meal_targets": meal_targets,
            "suggested_meals": recommendations,
            "strategy_trace": strategy_trace,
            "rationale": rationale,
        }
