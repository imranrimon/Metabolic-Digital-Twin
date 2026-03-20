import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metabolic_rl import CHECKPOINT_PATH, DQNAgent, MetabolicEnv, train_metabolic_rl
from recommender import DietRecommender
from training_utils import load_model_state, progress


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FOOD_DB_PATH = PROJECT_ROOT / "src" / "food_db.json"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
ABLATION_PATH = RESULTS_DIR / "recommender_ablation.csv"
CASE_STUDIES_PATH = RESULTS_DIR / "recommender_case_studies.csv"
PLOT_PATH = PLOTS_DIR / "recommender_ablation_tradeoffs.png"

MEAL_ORDER = ["Breakfast", "Lunch", "Dinner", "Snack"]
MEAL_SHARE = {
    "Breakfast": 0.25,
    "Lunch": 0.35,
    "Dinner": 0.30,
    "Snack": 0.10,
}
MEAL_SLOTS = 2

ACTION_PROFILES = {
    0: {"label": "very_low_carb", "target_carbs": 8.0},
    1: {"label": "low_carb", "target_carbs": 14.0},
    2: {"label": "balanced", "target_carbs": 22.0},
    3: {"label": "moderate_carb", "target_carbs": 30.0},
    4: {"label": "high_carb", "target_carbs": 38.0},
}

VARIANT_ORDER = [
    "Full heuristic",
    "No risk adjustment",
    "No GI prioritization",
    "No meal-type routing",
    "RL policy only",
    "Hybrid RL + heuristic",
]


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


def estimate_height_cm(gender, age):
    base_height = 176.0 if gender == 1 else 163.0
    if age >= 60:
        base_height -= 2.0
    elif age >= 45:
        base_height -= 1.0
    return base_height


def compute_risk_level(age, bmi, glucose):
    age_term = max(age - 25.0, 0.0) / 45.0
    bmi_term = max(bmi - 20.0, 0.0) / 15.0
    glucose_term = max(glucose - 90.0, 0.0) / 120.0
    risk = 0.05 + (0.20 * age_term) + (0.35 * bmi_term) + (0.45 * glucose_term)
    return float(np.clip(risk, 0.05, 0.95))


def build_synthetic_cohort():
    cases = []
    case_index = 1
    for gender in [0, 1]:
        for age in [32, 48, 64]:
            for bmi in [22.0, 28.0, 34.0]:
                for glucose in [105.0, 145.0, 185.0]:
                    height_cm = estimate_height_cm(gender, age)
                    weight_kg = bmi * (height_cm / 100.0) ** 2
                    cases.append(
                        {
                            "case_id": f"C{case_index:03d}",
                            "case_label": f"{'Male' if gender == 1 else 'Female'} age {age}, BMI {bmi:.0f}, glucose {glucose:.0f}",
                            "gender": gender,
                            "age": age,
                            "bmi": float(bmi),
                            "glucose": float(glucose),
                            "height_cm": float(round(height_cm, 1)),
                            "weight_kg": float(round(weight_kg, 1)),
                            "risk_level": compute_risk_level(age, bmi, glucose),
                        }
                    )
                    case_index += 1
    return cases


def build_case_studies():
    studies = []
    profiles = [
        ("CS1", "Low risk / normal BMI", 0, 32, 22.0, 102.0),
        ("CS2", "Moderate risk / overweight", 1, 49, 29.0, 148.0),
        ("CS3", "High risk / obese", 1, 66, 35.0, 188.0),
    ]
    for case_id, label, gender, age, bmi, glucose in profiles:
        height_cm = estimate_height_cm(gender, age)
        weight_kg = bmi * (height_cm / 100.0) ** 2
        studies.append(
            {
                "case_id": case_id,
                "case_label": label,
                "gender": gender,
                "age": age,
                "bmi": float(bmi),
                "glucose": float(glucose),
                "height_cm": float(round(height_cm, 1)),
                "weight_kg": float(round(weight_kg, 1)),
                "risk_level": compute_risk_level(age, bmi, glucose),
            }
        )
    return studies


def build_meal_targets(total_calories):
    return {meal: total_calories * share for meal, share in MEAL_SHARE.items()}


def should_apply_risk_adjustment(case):
    return case["bmi"] > 25.0 or case["risk_level"] > 0.5


def compute_daily_calories(recommender, case, use_risk_adjustment=True):
    calories = recommender.calculate_calories(
        age=case["age"],
        weight=case["weight_kg"],
        height=case["height_cm"],
        gender=case["gender"],
    )
    if use_risk_adjustment and should_apply_risk_adjustment(case):
        calories *= 0.85
    return float(calories)


def empty_plan():
    return {meal: [] for meal in MEAL_ORDER}


def candidate_pool(food_db, meal_name, use_meal_routing=True):
    if not use_meal_routing:
        return list(food_db)

    if meal_name in {"Lunch", "Dinner"}:
        allowed = {"Lunch/Dinner"}
    else:
        allowed = {meal_name}

    matches = [item for item in food_db if infer_meal_type(item["food"]) in allowed]
    if matches:
        return matches
    return list(food_db)


def sort_food_db(food_db, use_gi_priority=True):
    if use_gi_priority:
        return sorted(food_db, key=lambda item: (item["gi"], item["calories"], item["food"].lower()))
    return sorted(food_db, key=lambda item: item["food"].lower())


def plan_complete(recommendations):
    return all(len(recommendations[meal]) >= MEAL_SLOTS for meal in MEAL_ORDER)


def heuristic_plan(food_db, daily_cals, use_gi_priority=True, use_meal_routing=True):
    recommendations = empty_plan()
    used_foods = set()
    meal_targets = build_meal_targets(daily_cals)
    for meal_name in MEAL_ORDER:
        recommendations[meal_name] = select_heuristic_items(
            food_db=food_db,
            meal_name=meal_name,
            meal_target=meal_targets[meal_name],
            used_foods=used_foods,
            use_gi_priority=use_gi_priority,
            use_meal_routing=use_meal_routing,
        )

    return {
        "daily_caloric_target": round(daily_cals),
        "meal_targets": meal_targets,
        "suggested_meals": recommendations,
        "strategy_trace": [],
    }


def meal_action_from_items(items):
    carbs = sum(item["carbs"] for item in items)
    protein = sum(item["protein"] for item in items)
    fat = sum(item["fat"] for item in items)
    return [
        float(np.clip(carbs / 60.0, 0.0, 1.2)),
        float(np.clip(protein / 35.0, 0.0, 1.0)),
        float(np.clip(fat / 25.0, 0.0, 1.0)),
    ]


def score_strategy_item(item, per_item_calories, action_profile, use_gi_bonus):
    target_carbs = action_profile["target_carbs"]
    calorie_penalty = abs(item["calories"] - per_item_calories) / max(per_item_calories, 1.0)
    carb_penalty = abs(item["carbs"] - target_carbs) / max(target_carbs, 1.0)
    fat_penalty = max(item["fat"] - 20.0, 0.0) / 20.0
    protein_bonus = item["protein"] / 30.0
    gi_penalty = item["gi"] / 100.0 if use_gi_bonus else 0.0
    return calorie_penalty + carb_penalty + fat_penalty + gi_penalty - (0.20 * protein_bonus)


def score_heuristic_item(item, per_item_calories, meal_name, use_gi_priority):
    calorie_penalty = abs(item["calories"] - per_item_calories) / max(per_item_calories, 1.0)
    low_calorie_penalty = max((0.45 * per_item_calories) - item["calories"], 0.0) / max(per_item_calories, 1.0)
    high_calorie_penalty = max(item["calories"] - (1.75 * per_item_calories), 0.0) / max(per_item_calories, 1.0)
    gi_penalty = item["gi"] / 100.0 if use_gi_priority else 0.0
    fat_penalty = max(item["fat"] - 22.0, 0.0) / 22.0
    protein_bonus = item["protein"] / (25.0 if meal_name != "Snack" else 18.0)
    return calorie_penalty + low_calorie_penalty + high_calorie_penalty + gi_penalty + fat_penalty - (0.15 * protein_bonus)


def select_heuristic_items(food_db, meal_name, meal_target, used_foods, use_gi_priority, use_meal_routing):
    per_item_calories = meal_target / MEAL_SLOTS
    candidates = candidate_pool(food_db, meal_name, use_meal_routing=use_meal_routing)
    ranked = sorted(
        candidates,
        key=lambda item: (
            score_heuristic_item(item, per_item_calories, meal_name, use_gi_priority),
            item["gi"] if use_gi_priority else 0.0,
            item["food"].lower(),
        ),
    )
    picks = []
    for item in ranked:
        if item["food"] in used_foods:
            continue
        picks.append(item)
        used_foods.add(item["food"])
        if len(picks) >= MEAL_SLOTS:
            break

    if len(picks) < MEAL_SLOTS:
        fallback = sort_food_db(food_db, use_gi_priority=use_gi_priority)
        for item in fallback:
            if item["food"] in used_foods:
                continue
            picks.append(item)
            used_foods.add(item["food"])
            if len(picks) >= MEAL_SLOTS:
                break
    return picks


def select_strategy_items(food_db, meal_name, meal_target, action_profile, used_foods, use_gi_bonus):
    per_item_calories = meal_target / MEAL_SLOTS
    candidates = candidate_pool(food_db, meal_name, use_meal_routing=True)
    ranked = sorted(
        candidates,
        key=lambda item: (
            score_strategy_item(item, per_item_calories, action_profile, use_gi_bonus),
            item["gi"],
            item["food"].lower(),
        ),
    )
    picks = []
    for item in ranked:
        if item["food"] in used_foods:
            continue
        picks.append(item)
        used_foods.add(item["food"])
        if len(picks) >= MEAL_SLOTS:
            break

    if len(picks) < MEAL_SLOTS:
        fallback = sort_food_db(food_db, use_gi_priority=use_gi_bonus)
        for item in fallback:
            if item["food"] in used_foods:
                continue
            picks.append(item)
            used_foods.add(item["food"])
            if len(picks) >= MEAL_SLOTS:
                break
    return picks


def get_rl_agent(force_train=False):
    policy_path = Path(CHECKPOINT_PATH)
    if force_train or not policy_path.exists():
        print("Training RL policy checkpoint for recommender ablation...")
        train_metabolic_rl()

    agent = DQNAgent(state_dim=1, action_dim=len(ACTION_PROFILES))
    load_model_state(agent.model, CHECKPOINT_PATH, map_location="cpu")
    agent.epsilon = 0.0
    agent.model.eval()
    return agent


def rl_plan(food_db, recommender, case, agent, use_gi_bonus):
    daily_cals = compute_daily_calories(recommender, case, use_risk_adjustment=True)
    meal_targets = build_meal_targets(daily_cals)
    recommendations = empty_plan()
    strategy_trace = []
    used_foods = set()
    env = MetabolicEnv()
    env.state = float(case["glucose"])

    for meal_name in MEAL_ORDER:
        action_index = agent.act(np.array([env.state], dtype=np.float32), greedy=True)
        action_profile = ACTION_PROFILES[action_index]
        strategy_trace.append(action_profile["label"])
        items = select_strategy_items(
            food_db=food_db,
            meal_name=meal_name,
            meal_target=meal_targets[meal_name],
            action_profile=action_profile,
            used_foods=used_foods,
            use_gi_bonus=use_gi_bonus,
        )
        recommendations[meal_name] = items
        env.step(meal_action_from_items(items))

    return {
        "daily_caloric_target": round(daily_cals),
        "meal_targets": meal_targets,
        "suggested_meals": recommendations,
        "strategy_trace": strategy_trace,
    }


def stringify_meal(items):
    if not items:
        return "No recommendation"
    return "; ".join(item["food"] for item in items)


def meal_type_matches(meal_name, item):
    inferred = infer_meal_type(item["food"])
    if meal_name in {"Lunch", "Dinner"}:
        return inferred == "Lunch/Dinner"
    return inferred == meal_name


def evaluate_plan(case, variant, plan):
    all_items = [item for meal in MEAL_ORDER for item in plan["suggested_meals"].get(meal, [])]
    gi_values = [item["gi"] for item in all_items]
    total_calories = sum(item["calories"] for item in all_items)
    total_carbs = sum(item["carbs"] for item in all_items)
    total_protein = sum(item["protein"] for item in all_items)
    total_fat = sum(item["fat"] for item in all_items)
    filled_slots = sum(len(plan["suggested_meals"].get(meal, [])) for meal in MEAL_ORDER)
    unique_foods = len({item["food"] for item in all_items})
    consistency_matches = sum(
        1
        for meal_name in MEAL_ORDER
        for item in plan["suggested_meals"].get(meal_name, [])
        if meal_type_matches(meal_name, item)
    )

    env = MetabolicEnv()
    env.state = float(case["glucose"])
    glucose_trace = [env.state]
    total_reward = 0.0
    in_range_count = 0
    for meal_name in MEAL_ORDER:
        meal_items = plan["suggested_meals"].get(meal_name, [])
        next_state, reward, _, _ = env.step(meal_action_from_items(meal_items))
        glucose_value = float(next_state[0])
        glucose_trace.append(glucose_value)
        total_reward += reward
        if 70.0 <= glucose_value <= 140.0:
            in_range_count += 1

    row = {
        "variant": variant,
        "case_id": case["case_id"],
        "case_label": case["case_label"],
        "age": case["age"],
        "gender": "Male" if case["gender"] == 1 else "Female",
        "bmi": case["bmi"],
        "glucose": case["glucose"],
        "risk_level": round(case["risk_level"], 4),
        "daily_caloric_target": plan["daily_caloric_target"],
        "plan_calories": total_calories,
        "calorie_target_error": abs(total_calories - plan["daily_caloric_target"]),
        "mean_gi": float(np.mean(gi_values)) if gi_values else np.nan,
        "max_gi": float(np.max(gi_values)) if gi_values else np.nan,
        "meal_coverage": filled_slots / float(len(MEAL_ORDER) * MEAL_SLOTS),
        "meal_type_consistency": consistency_matches / float(filled_slots) if filled_slots else 0.0,
        "diversity": unique_foods / float(filled_slots) if filled_slots else 0.0,
        "total_carbs": total_carbs,
        "total_protein": total_protein,
        "total_fat": total_fat,
        "sim_reward": total_reward,
        "time_in_range": in_range_count / float(len(MEAL_ORDER)),
        "final_glucose": glucose_trace[-1],
        "peak_glucose": max(glucose_trace),
        "strategy_trace": " | ".join(plan.get("strategy_trace", [])),
        "breakfast": stringify_meal(plan["suggested_meals"].get("Breakfast", [])),
        "lunch": stringify_meal(plan["suggested_meals"].get("Lunch", [])),
        "dinner": stringify_meal(plan["suggested_meals"].get("Dinner", [])),
        "snack": stringify_meal(plan["suggested_meals"].get("Snack", [])),
    }
    return row


def summarize_results(detail_df):
    summary = (
        detail_df.groupby("variant", as_index=False)
        .agg(
            cases=("case_id", "count"),
            mean_gi=("mean_gi", "mean"),
            max_gi=("max_gi", "mean"),
            calorie_target_error=("calorie_target_error", "mean"),
            meal_coverage=("meal_coverage", "mean"),
            meal_type_consistency=("meal_type_consistency", "mean"),
            diversity=("diversity", "mean"),
            total_carbs=("total_carbs", "mean"),
            total_protein=("total_protein", "mean"),
            sim_reward=("sim_reward", "mean"),
            time_in_range=("time_in_range", "mean"),
            final_glucose=("final_glucose", "mean"),
        )
        .copy()
    )
    summary["variant"] = pd.Categorical(summary["variant"], categories=VARIANT_ORDER, ordered=True)
    summary = summary.sort_values("variant").reset_index(drop=True)
    return summary


def plot_summary(summary_df):
    plot_df = summary_df.copy()
    plot_df["variant"] = plot_df["variant"].astype(str)

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.suptitle("Recommendation-Layer Ablation Tradeoffs", fontsize=15)

    colors = ["#2A6F97", "#5FA8D3", "#9CC5A1", "#F4A261", "#E76F51", "#6D597A"]

    axes[0].barh(plot_df["variant"], plot_df["mean_gi"], color=colors)
    axes[0].invert_yaxis()
    axes[0].set_title("Mean GI (lower is better)")
    axes[0].set_xlabel("Mean GI")

    axes[1].barh(plot_df["variant"], plot_df["calorie_target_error"], color=colors)
    axes[1].invert_yaxis()
    axes[1].set_title("Calorie Target Error (lower is better)")
    axes[1].set_xlabel("Mean absolute error")

    axes[2].barh(plot_df["variant"], plot_df["sim_reward"], color=colors)
    axes[2].invert_yaxis()
    axes[2].set_title("Simulated Reward (higher is better)")
    axes[2].set_xlabel("Average reward")

    for axis in axes:
        axis.grid(alpha=0.2, axis="x")

    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


def print_summary(summary_df):
    display_df = summary_df.copy()
    numeric_cols = [col for col in display_df.columns if col != "variant" and col != "cases"]
    display_df[numeric_cols] = display_df[numeric_cols].round(4)
    print("\nRecommendation-layer ablation summary")
    print(display_df.to_string(index=False))


def run_variant(recommender, food_db, agent, case, variant):
    if variant == "Full heuristic":
        daily_cals = compute_daily_calories(recommender, case, use_risk_adjustment=True)
        return heuristic_plan(food_db, daily_cals, use_gi_priority=True, use_meal_routing=True)

    if variant == "No risk adjustment":
        daily_cals = compute_daily_calories(recommender, case, use_risk_adjustment=False)
        return heuristic_plan(food_db, daily_cals, use_gi_priority=True, use_meal_routing=True)

    if variant == "No GI prioritization":
        daily_cals = compute_daily_calories(recommender, case, use_risk_adjustment=True)
        return heuristic_plan(food_db, daily_cals, use_gi_priority=False, use_meal_routing=True)

    if variant == "No meal-type routing":
        daily_cals = compute_daily_calories(recommender, case, use_risk_adjustment=True)
        return heuristic_plan(food_db, daily_cals, use_gi_priority=True, use_meal_routing=False)

    if variant == "RL policy only":
        return rl_plan(food_db, recommender, case, agent, use_gi_bonus=False)

    if variant == "Hybrid RL + heuristic":
        return rl_plan(food_db, recommender, case, agent, use_gi_bonus=True)

    raise ValueError(f"Unknown variant: {variant}")


def run_recommender_ablation(force_train_policy=False, skip_plot=False):
    recommender = DietRecommender(str(FOOD_DB_PATH))
    food_db = recommender.food_db
    agent = get_rl_agent(force_train=force_train_policy)

    cases = build_synthetic_cohort()
    detail_rows = []

    total_iterations = len(cases) * len(VARIANT_ORDER)
    case_bar = progress(range(total_iterations), desc="Recommender ablation")
    progress_index = 0
    for case in cases:
        for variant in VARIANT_ORDER:
            plan = run_variant(recommender, food_db, agent, case, variant)
            detail_rows.append(evaluate_plan(case, variant, plan))
            progress_index += 1
            if hasattr(case_bar, "update"):
                case_bar.update(1)
    if hasattr(case_bar, "close"):
        case_bar.close()

    detail_df = pd.DataFrame(detail_rows)
    summary_df = summarize_results(detail_df)

    case_rows = []
    for case in build_case_studies():
        for variant in VARIANT_ORDER:
            plan = run_variant(recommender, food_db, agent, case, variant)
            case_rows.append(evaluate_plan(case, variant, plan))
    case_df = pd.DataFrame(case_rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(ABLATION_PATH, index=False)
    case_df.to_csv(CASE_STUDIES_PATH, index=False)

    if not skip_plot:
        plot_summary(summary_df)

    print_summary(summary_df)
    print(f"\nSaved ablation summary to {ABLATION_PATH}")
    print(f"Saved case studies to {CASE_STUDIES_PATH}")
    if not skip_plot:
        print(f"Saved ablation figure to {PLOT_PATH}")

    return summary_df, case_df


def parse_args():
    parser = argparse.ArgumentParser(description="Run offline recommendation-layer ablation for the metabolic digital twin.")
    parser.add_argument(
        "--force-train-policy",
        action="store_true",
        help="Retrain the RL policy before evaluating RL-based recommendation variants.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip generation of the summary plot.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_recommender_ablation(force_train_policy=args.force_train_policy, skip_plot=args.skip_plot)
