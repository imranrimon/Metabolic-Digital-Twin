import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from recommender import DietRecommender
from recommender_nextgen import GuidelineGraphTwinRecommender, NextGenPlannerConfig
from run_recommender_ablation import (
    FOOD_DB_PATH,
    build_case_studies,
    build_synthetic_cohort,
    compute_daily_calories,
    evaluate_plan,
    get_rl_agent,
    heuristic_plan,
    rl_plan,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
ABLATION_PATH = RESULTS_DIR / "recommender_nextgen_ablation.csv"
CASE_STUDIES_PATH = RESULTS_DIR / "recommender_nextgen_case_studies.csv"
PLOT_PATH = PLOTS_DIR / "recommender_nextgen_tradeoffs.png"

VARIANT_ORDER = [
    "Full heuristic",
    "Guideline planner",
    "Guideline + graph",
    "Guideline + twin",
    "Graph-Twin planner",
    "RL policy only",
    "Hybrid RL + heuristic",
]


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
    colors = ["#1D3557", "#457B9D", "#2A9D8F", "#8AB17D", "#E9C46A", "#F4A261", "#E76F51"]

    fig, axes = plt.subplots(1, 4, figsize=(21, 6))
    fig.suptitle("Next-Generation Recommendation Ablation", fontsize=15)

    axes[0].barh(plot_df["variant"], plot_df["mean_gi"], color=colors)
    axes[0].invert_yaxis()
    axes[0].set_title("Mean GI")
    axes[0].set_xlabel("Lower is better")

    axes[1].barh(plot_df["variant"], plot_df["calorie_target_error"], color=colors)
    axes[1].invert_yaxis()
    axes[1].set_title("Calorie Error")
    axes[1].set_xlabel("Lower is better")

    axes[2].barh(plot_df["variant"], plot_df["sim_reward"], color=colors)
    axes[2].invert_yaxis()
    axes[2].set_title("Simulated Reward")
    axes[2].set_xlabel("Higher is better")

    axes[3].barh(plot_df["variant"], plot_df["meal_type_consistency"], color=colors)
    axes[3].invert_yaxis()
    axes[3].set_title("Meal-Type Consistency")
    axes[3].set_xlabel("Higher is better")
    axes[3].set_xlim(0.0, 1.05)

    for axis in axes:
        axis.grid(alpha=0.2, axis="x")

    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


def print_summary(summary_df):
    display_df = summary_df.copy()
    numeric_cols = [col for col in display_df.columns if col not in {"variant", "cases"}]
    display_df[numeric_cols] = display_df[numeric_cols].round(4)
    print("\nNext-generation recommendation ablation summary")
    print(display_df.to_string(index=False))


def run_variant(heuristic_recommender, nextgen_planners, rl_agent, case, variant):
    if variant == "Full heuristic":
        daily_cals = compute_daily_calories(heuristic_recommender, case, use_risk_adjustment=True)
        return heuristic_plan(heuristic_recommender.food_db, daily_cals, use_gi_priority=True, use_meal_routing=True)

    if variant == "Guideline planner":
        planner = nextgen_planners["Guideline planner"]
        return planner.recommend_meals(
            age=case["age"],
            bmi=case["bmi"],
            risk_level=case["risk_level"],
            gender=case["gender"],
            weight=case["weight_kg"],
            height=case["height_cm"],
            glucose=case["glucose"],
        )

    if variant == "Guideline + graph":
        planner = nextgen_planners["Guideline + graph"]
        return planner.recommend_meals(
            age=case["age"],
            bmi=case["bmi"],
            risk_level=case["risk_level"],
            gender=case["gender"],
            weight=case["weight_kg"],
            height=case["height_cm"],
            glucose=case["glucose"],
        )

    if variant == "Guideline + twin":
        planner = nextgen_planners["Guideline + twin"]
        return planner.recommend_meals(
            age=case["age"],
            bmi=case["bmi"],
            risk_level=case["risk_level"],
            gender=case["gender"],
            weight=case["weight_kg"],
            height=case["height_cm"],
            glucose=case["glucose"],
        )

    if variant == "Graph-Twin planner":
        planner = nextgen_planners["Graph-Twin planner"]
        return planner.recommend_meals(
            age=case["age"],
            bmi=case["bmi"],
            risk_level=case["risk_level"],
            gender=case["gender"],
            weight=case["weight_kg"],
            height=case["height_cm"],
            glucose=case["glucose"],
        )

    if variant == "RL policy only":
        return rl_plan(heuristic_recommender.food_db, heuristic_recommender, case, rl_agent, use_gi_bonus=False)

    if variant == "Hybrid RL + heuristic":
        return rl_plan(heuristic_recommender.food_db, heuristic_recommender, case, rl_agent, use_gi_bonus=True)

    raise ValueError(f"Unknown variant: {variant}")


def build_planners():
    return {
        "Guideline planner": GuidelineGraphTwinRecommender(
            str(FOOD_DB_PATH),
            config=NextGenPlannerConfig(use_graph=False, use_state_rerank=False),
        ),
        "Guideline + graph": GuidelineGraphTwinRecommender(
            str(FOOD_DB_PATH),
            config=NextGenPlannerConfig(use_graph=True, use_state_rerank=False),
        ),
        "Guideline + twin": GuidelineGraphTwinRecommender(
            str(FOOD_DB_PATH),
            config=NextGenPlannerConfig(use_graph=False, use_state_rerank=True),
        ),
        "Graph-Twin planner": GuidelineGraphTwinRecommender(
            str(FOOD_DB_PATH),
            config=NextGenPlannerConfig(use_graph=True, use_state_rerank=True),
        ),
    }


def run_recommender_nextgen_ablation(force_train_policy=False, skip_plot=False):
    heuristic_recommender = DietRecommender(str(FOOD_DB_PATH))
    nextgen_planners = build_planners()
    rl_agent = get_rl_agent(force_train=force_train_policy)

    detail_rows = []
    for case in build_synthetic_cohort():
        for variant in VARIANT_ORDER:
            plan = run_variant(heuristic_recommender, nextgen_planners, rl_agent, case, variant)
            detail_rows.append(evaluate_plan(case, variant, plan))

    detail_df = pd.DataFrame(detail_rows)
    summary_df = summarize_results(detail_df)

    case_rows = []
    for case in build_case_studies():
        for variant in VARIANT_ORDER:
            plan = run_variant(heuristic_recommender, nextgen_planners, rl_agent, case, variant)
            case_rows.append(evaluate_plan(case, variant, plan))
    case_df = pd.DataFrame(case_rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(ABLATION_PATH, index=False)
    case_df.to_csv(CASE_STUDIES_PATH, index=False)
    if not skip_plot:
        plot_summary(summary_df)

    print_summary(summary_df)
    print(f"\nSaved next-generation ablation summary to {ABLATION_PATH}")
    print(f"Saved next-generation case studies to {CASE_STUDIES_PATH}")
    if not skip_plot:
        print(f"Saved next-generation ablation figure to {PLOT_PATH}")

    return summary_df, case_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a cutting-edge-inspired recommendation ablation with guideline, graph, and digital-twin components."
    )
    parser.add_argument(
        "--force-train-policy",
        action="store_true",
        help="Retrain the RL policy before running RL-based recommendation variants.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip generation of the summary plot.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_recommender_nextgen_ablation(force_train_policy=args.force_train_policy, skip_plot=args.skip_plot)
