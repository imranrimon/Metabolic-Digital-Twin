from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
METRICS_PATH = RESULTS_DIR / "metrics_summary.csv"

TOP_100K_AUC_PATH = PLOTS_DIR / "top_100k_model_auc.png"
HBA1C_ABLATION_PATH = PLOTS_DIR / "hba1c_ablation_auc_drop.png"


def load_metrics():
    return pd.read_csv(METRICS_PATH)


def plot_top_100k_auc(metrics_df):
    subset = metrics_df[
        (metrics_df["dataset"] == "100k")
        & (metrics_df["ablation"] == "Full_Features")
        & (
            metrics_df["model"].isin(
                [
                    "LightGBM",
                    "XGBoost",
                    "Stacked Ensemble (RF+XGB)",
                    "Random Forest",
                    "Attention-ResNet",
                ]
            )
        )
    ].copy()

    label_map = {
        "LightGBM": "LightGBM",
        "XGBoost": "XGBoost",
        "Stacked Ensemble (RF+XGB)": "Stacked Ensemble",
        "Random Forest": "Random Forest",
        "Attention-ResNet": "Attention-ResNet",
    }
    order = ["LightGBM", "XGBoost", "Stacked Ensemble", "Random Forest", "Attention-ResNet"]
    subset["label"] = subset["model"].map(label_map)
    subset["label"] = pd.Categorical(subset["label"], categories=order, ordered=True)
    subset = subset.sort_values("label")

    plt.figure(figsize=(8.2, 4.8))
    colors = ["#1f5aa6", "#3d7dd8", "#70a1d7", "#9db4c0", "#d9a441"]
    bars = plt.bar(subset["label"], subset["auc"], color=colors)
    plt.ylim(0.94, 0.985)
    plt.ylabel("AUC")
    plt.title("Top Models on the 100k Diabetes Prediction Benchmark")
    plt.grid(axis="y", alpha=0.2)
    plt.xticks(rotation=15, ha="right")

    for bar, value in zip(bars, subset["auc"]):
        plt.text(bar.get_x() + bar.get_width() / 2.0, value + 0.0006, f"{value:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(TOP_100K_AUC_PATH, dpi=220, bbox_inches="tight")
    plt.close()


def plot_hba1c_ablation(metrics_df):
    full_df = metrics_df[
        (metrics_df["dataset"] == "100k")
        & (metrics_df["ablation"] == "Full_Features")
        & (
            metrics_df["model"].isin(
                [
                    "LightGBM",
                    "XGBoost",
                    "Stacked Ensemble (RF+XGB)",
                    "Random Forest",
                    "Attention-ResNet",
                ]
            )
        )
    ][["model", "auc"]].rename(columns={"auc": "full_auc"})

    no_hba1c_df = metrics_df[
        (metrics_df["dataset"] == "100k")
        & (metrics_df["ablation"] == "No_HbA1c")
        & (
            metrics_df["model"].isin(
                [
                    "LightGBM",
                    "XGBoost",
                    "Stacked Ensemble (RF+XGB)",
                    "Random Forest",
                    "Attention-ResNet",
                ]
            )
        )
    ][["model", "auc"]].rename(columns={"auc": "no_hba1c_auc"})

    merged = full_df.merge(no_hba1c_df, on="model")
    merged["delta"] = merged["full_auc"] - merged["no_hba1c_auc"]

    label_map = {
        "LightGBM": "LightGBM",
        "XGBoost": "XGBoost",
        "Stacked Ensemble (RF+XGB)": "Stacked Ensemble",
        "Random Forest": "Random Forest",
        "Attention-ResNet": "Attention-ResNet",
    }
    order = ["Attention-ResNet", "Random Forest", "Stacked Ensemble", "XGBoost", "LightGBM"]
    merged["label"] = merged["model"].map(label_map)
    merged["label"] = pd.Categorical(merged["label"], categories=order, ordered=True)
    merged = merged.sort_values("label")

    plt.figure(figsize=(8.2, 4.8))
    colors = ["#b44d3a", "#c86b5a", "#d98f76", "#e8b99b", "#f0d8c4"]
    bars = plt.barh(merged["label"], merged["delta"], color=colors)
    plt.xlabel("AUC drop after removing HbA1c")
    plt.title("Clinical Dependence on HbA1c Across the Strongest 100k Models")
    plt.grid(axis="x", alpha=0.2)

    for bar, value in zip(bars, merged["delta"]):
        plt.text(value + 0.0007, bar.get_y() + bar.get_height() / 2.0, f"{value:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(HBA1C_ABLATION_PATH, dpi=220, bbox_inches="tight")
    plt.close()


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df = load_metrics()
    plot_top_100k_auc(metrics_df)
    plot_hba1c_ablation(metrics_df)
    print(f"Saved {TOP_100K_AUC_PATH}")
    print(f"Saved {HBA1C_ABLATION_PATH}")


if __name__ == "__main__":
    main()
