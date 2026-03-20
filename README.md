# Diabetes Project: Risk Benchmark and Metabolic Digital Twin Prototype

This repository combines three related work streams:

1. diabetes risk prediction on tabular clinical data,
2. prototype glucose-forecasting and diet-recommendation modules,
3. a FastAPI/mobile demo that exposes the models.

The most paper-ready part of the project is the tabular risk benchmark. The forecasting, reinforcement-learning, and dashboard pieces are useful prototype components, but they are not yet validated strongly enough to carry the main scientific claim on their own.

## Codebase Layout

The repository now uses a cleaner split between reusable package code and runnable scripts:

- `src/metabolic_twin/`: reusable package modules grouped by domain
- `src/metabolic_twin/config/`: centralized repo-relative paths for datasets, checkpoints, and outputs
- `src/`: experiment and entrypoint scripts, plus backward-compatible wrappers for older imports
- `results/`: paper outputs, plots, and inspection artifacts
- `models/`: production and benchmark model metadata/artifacts
- `docs/`: paper framing, walkthroughs, and structure notes

See `docs/codebase_structure.md` for the detailed layout.

## Current Paper-Ready Position

| Area | Strongest supported finding | Primary evidence |
|---|---|---|
| Risk prediction on the 100k dataset | LightGBM reached AUC 0.9786 and XGBoost reached AUC 0.9773 under a consistent train/test protocol | `results/metrics_summary.csv` |
| Tabular foundation-model baseline | TabICLv2 reached AUC 0.9804 on the 100k dataset and outperformed LightGBM/XGBoost in the dedicated FM comparison | `results/tabular_foundation_benchmark.csv` |
| Risk prediction on Pima | Random Forest reached AUC 0.8155 and remained the strongest small-data baseline | `results/metrics_summary.csv` |
| Novel architecture | EchoCeptionNet reached AUC 0.9773, while the new hyperspherical EchoCeptionSphereNet reached AUC 0.9770 with the best F1 at 0.8030 | `results/echoception_benchmark.csv` |
| CGM prototype benchmark | ST-Attention LSTM currently leads the temporal holdout benchmark, with TabICLForecaster added as a foundation-model baseline | `results/cgm_foundation_benchmark.csv` |
| Recommendation-layer ablation | The full heuristic remains the strongest plausible recommendation variant, while no-routing variants reveal the need to report meal plausibility alongside metabolic proxies | `results/recommender_ablation.csv` |
| Next-generation recommendation ablation | A cutting-edge-inspired guideline, graph, and digital-twin planner family improves calorie targeting and post-meal glucose proxies versus the simple heuristic, with the state-aware variant currently strongest | `results/recommender_nextgen_ablation.csv` |
| Clinical ablation | Removing HbA1c reduced AUC by 0.0457 to 0.0743 across the top 100k models | `results/metrics_summary.csv` |
| Optimized stacking | Tuned XGBoost slightly outperformed the stacking meta-learner in out-of-fold evaluation | `results/grandmaster_benchmark.csv` |

## What Should Not Be the Main Paper Claim Yet

- `results/hyperscale_benchmark.csv` contains a 0.9908 AUC headline number, but the current hyperscale script applies SMOTE and scaling before the train/test split, so it should be treated as exploratory until re-run with a leakage-safe protocol.
- `src/train_sota_cde.py` and `src/cde_preprocess.py` provide a prototype continuous-time forecasting pipeline, but the current training target is not yet a clean future forecasting target.
- `src/metabolic_rl.py` trains a policy in a simulated environment rather than on real patient decision trajectories, so it is better framed as a proof-of-concept controller.
- `mobile_app/backend/api.py` is a demo integration layer; it is not a validated clinical deployment.

## Goal-to-Task Map

| Goal | Main scripts and assets | Assessment |
|---|---|---|
| Benchmark diabetes risk prediction | `src/preprocess.py`, `src/run_all_experiments.py`, `src/train_stacking.py`, `src/run_echoception_benchmark.py`, `results/metrics_summary.csv` | Clear, coherent, and currently the strongest contribution |
| Explore a novel neural tabular model | `src/models_novel.py`, `src/run_echoception_benchmark.py` | Competitive and interesting, but should be framed as a strong exploratory contribution rather than a proven new SOTA |
| Add foundation-model baselines | `src/run_tabular_fm_benchmark.py`, `src/run_timeseries_fm_benchmark.py` | Clear comparison layer that strengthens the paper, as long as it remains a baseline section rather than the main novelty claim |
| Prototype glucose forecasting | `src/run_timeseries_fm_benchmark.py`, `src/cde_preprocess.py`, `src/train_sota_cde.py`, `src/shanghai_preprocess.py` | Now has a coherent temporal-holdout benchmark, but still reads best as a prototype extension rather than a headline clinical result |
| Prototype dietary recommendation and control | `src/recommender.py`, `src/metabolic_rl.py`, `src/show_recommendations.py`, `src/run_recommender_ablation.py` | Now has a coherent offline ablation and case-study evaluation, but it is still a prototype recommendation layer rather than a clinically validated policy system |
| Demonstrate end-to-end integration | `mobile_app/backend/api.py`, `mobile_app/frontend/` | Useful demo and systems contribution, best presented as a prototype interface |

## Recommended Paper Story

Make the paper center on:

- a careful benchmark of classical and neural models for diabetes risk prediction,
- the clinically meaningful HbA1c ablation,
- the proposed EchoCeptionNet as a competitive novel architecture,
- the broader digital-twin stack as a prototype systems extension.

Do not center the paper on global "SOTA" claims unless the repository is first cleaned up to include:

- leakage-safe reruns of every headline experiment,
- one unified evaluation protocol across all compared models,
- proper temporal holdout forecasting for the CGM module,
- a clearer distinction between validated results and exploratory prototypes.

## Where Newer Ideas Fit

Three strong next-step directions fit this repository well:

- conformal prediction on top of the best risk model to provide calibrated risk sets and abstentions, which now uses APS-style adaptive prediction sets in the production XGBoost API flow,
- hyperspherical classification for the neural branch of the tabular pipeline, which is now implemented as EchoCeptionSphereNet and can be analyzed as a head-level architectural upgrade,
- foundation models for tabular and medical time-series extensions, now added as dedicated comparison baselines rather than as the first headline claim.

## Repository Guide

- `project_paper.tex`: current paper draft aligned to the supported results.
- `docs/walkthrough.md`: project audit, maturity review, and recommended framing.
- `docs/paper_results_analysis.md`: a concrete results-section design for the paper.
- `results/README.md`: which result files are primary, secondary, or exploratory.
- `src/run_all_experiments.py`: unified comparable benchmark for Pima and 100k.
- `src/run_echoception_benchmark.py`: dedicated benchmark for the proposed model.
- `src/run_tabular_fm_benchmark.py`: TabICLv2 vs LightGBM/XGBoost baseline comparison on the 100k dataset.
- `src/run_timeseries_fm_benchmark.py`: temporal-holdout CGM benchmark with persistence, neural baselines, and TabICLForecaster.
- `src/run_recommender_ablation.py`: offline recommendation-layer ablation with heuristic, dropout, RL-only, and hybrid variants.
- `src/recommender_nextgen.py`: guideline-constrained, nutrient-graph, metabolic-state-aware meal planner.
- `src/run_recommender_nextgen_ablation.py`: cutting-edge-inspired recommender ablation for the next-generation planner family.
- `src/run_top_tier_validation.py`: bootstrap CIs, subgroup analysis, and calibration outputs for the production XGBoost risk model.
- `src/train_stacking.py`: optimized out-of-fold stacking experiment.
- `mobile_app/backend/api.py`: prototype deployment layer.

## Reproducing the Core Results

Run the core, paper-facing experiments:

```bash
python src/run_all_experiments.py
python src/run_echoception_benchmark.py
python src/run_tabular_fm_benchmark.py
python src/run_top_tier_validation.py
python src/train_stacking.py
```

Use the following only as exploratory research scripts until they are revalidated:

```bash
python src/train_hyperscale.py
python src/train_sota_cde.py
python src/metabolic_rl.py
```

Optional prototype benchmark:

```bash
python src/run_timeseries_fm_benchmark.py
python src/run_recommender_ablation.py
python src/run_recommender_nextgen_ablation.py
```
