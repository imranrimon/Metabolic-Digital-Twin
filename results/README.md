# Results Directory Guide

This folder mixes primary benchmark outputs and exploratory experiments. Use the following labels when preparing the paper.

## Primary Results for the Paper

| File | Status | Use in paper |
|---|---|---|
| `metrics_summary.csv` | Primary | Main comparable benchmark for Pima and 100k |
| `echoception_benchmark.csv` | Primary | Dedicated analysis of the proposed EchoCeptionNet |
| `grandmaster_benchmark.csv` | Secondary but useful | Optimized out-of-fold stacking analysis |
| `tabular_foundation_benchmark.csv` | Secondary but useful | Dedicated TabICLv2 baseline comparison against the strongest tabular baselines |
| `bootstrap_confidence_intervals.csv` | Secondary but useful | Journal-grade uncertainty table for the production XGBoost analysis |
| `calibration_summary_xgboost_100k.csv` | Secondary but useful | Calibration, Brier, and ECE summary for the production XGBoost model |

## Secondary Supporting Results

| File | Status | Use in paper |
|---|---|---|
| `novel_architecture_benchmark.csv` | Secondary | Supporting comparison against selected modern tabular models |
| `comprehensive_ablation_results.csv` | Secondary | Backup benchmark summary |
| `cgm_foundation_benchmark.csv` | Prototype supporting result | Temporal-holdout CGM comparison for the systems/prototype section |
| `recommender_ablation.csv` | Prototype supporting result | Offline recommendation-layer ablation across heuristic, dropout, RL-only, and hybrid variants |
| `recommender_case_studies.csv` | Prototype supporting result | Three fixed patient case studies for the recommendation-layer prototype |
| `recommender_nextgen_ablation.csv` | Prototype supporting result | Cutting-edge-inspired ablation for the guideline, graph, and digital-twin recommendation planner family |
| `recommender_nextgen_case_studies.csv` | Prototype supporting result | Fixed patient case studies for the next-generation recommendation planner family |
| `subgroup_analysis_xgboost_100k.csv` | Secondary | Sex and age subgroup robustness table |
| `calibration_bins_xgboost_100k.csv` | Secondary | Bin-level data behind the calibration figure |
| `plots/recommender_ablation_tradeoffs.png` | Prototype supporting figure | Compact visualization of GI, calorie-error, and reward tradeoffs in the recommendation layer |
| `plots/recommender_nextgen_tradeoffs.png` | Prototype supporting figure | Tradeoff view for the next-generation recommendation planner ablation |
| `plots/` | Supporting | Select only a few high-value figures for the paper |

## Exploratory Results

These files should not be used as headline scientific claims until they are revalidated under a consistent protocol.

| File | Why exploratory |
|---|---|
| `hyperscale_benchmark.csv` | The current training script uses a leakage-prone preprocessing order |
| `graph_benchmark.csv` | Single-number exploratory graph experiment without the same benchmark depth |
| `novel_7_model_benchmark.csv` | Mixes protocols and includes synthetic temporal assumptions |

## Recommended Citation Order Inside the Repo

When writing the paper or the README, cite result files in this order:

1. `metrics_summary.csv`
2. `echoception_benchmark.csv`
3. `tabular_foundation_benchmark.csv`
4. `grandmaster_benchmark.csv`
5. everything else only as supporting or exploratory evidence
