# Top-Tier Journal Results Analysis Blueprint

This document redesigns the results section as a journal-grade analysis plan rather than a simple benchmark summary. The key difference is that a top-tier journal expects a hierarchy of evidence:

1. a clearly defined primary endpoint,
2. strong baseline comparisons,
3. uncertainty and calibration analysis,
4. clinically interpretable analyses,
5. robustness and error analysis,
6. a disciplined separation between validated findings and prototype extensions.

For this repository, that means the paper should be centered on the tabular diabetes risk benchmark, and everything else should support that claim rather than compete with it.

## 1. Primary Claim Hierarchy

The paper should make exactly one primary empirical claim:

- the repository provides a strong and well-controlled benchmark for diabetes risk prediction on tabular data, and the proposed EchoCeption family remains competitive within that benchmark.

The secondary claims should be:

- HbA1c ablation demonstrates clinical dependence on a key biomarker,
- APS conformal prediction adds calibrated uncertainty reporting to the production risk model,
- foundation-model baselines were tested and contextualized,
- the CGM/recommendation/app stack is a systems prototype rather than a fully validated clinical contribution.

This hierarchy matters. A top-tier journal will tolerate ambitious scope only if the manuscript clearly distinguishes the strongest validated contribution from exploratory extensions.

## 2. Recommended Results Section Structure

Use the following section order in the paper.

### 2.1 Cohort, Splits, and Evaluation Protocol

Open the results section with a short paragraph that restates:

- datasets,
- train/test split policy,
- preprocessing leakage controls,
- class-imbalance handling,
- primary and secondary metrics,
- how uncertainty is reported.

This should be brief, but it must remove any ambiguity about how numbers were generated.

Recommended wording:

"All primary benchmark results were obtained under a leakage-safe train/test protocol, with preprocessing fit on the training partition only and class balancing applied only to the training data. Model ranking was based on AUC as the primary endpoint, with accuracy, precision, recall, and F1 reported as secondary metrics."

If space allows, add a compact reproducibility note:

- fixed random seed,
- exact dataset file,
- exact script that produced each table.

### 2.2 Primary Discrimination Benchmark

This is the first major table and should be treated as the paper's main quantitative result.

Use `results/metrics_summary.csv` for the standard benchmark table and keep the model set clean.

Recommended 100k table:

| Model | AUC | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| LightGBM | 0.9786 | 0.9710 | 0.9341 | 0.7082 | 0.8056 |
| XGBoost | 0.9773 | 0.9692 | 0.9075 | 0.7100 | 0.7967 |
| Stacked Ensemble | 0.9742 | 0.9636 | 0.8146 | 0.7394 | 0.7752 |
| Random Forest | 0.9660 | 0.9600 | 0.7781 | 0.7406 | 0.7589 |
| Attention-ResNet | 0.9653 | 0.9494 | 0.6950 | 0.7212 | 0.7079 |

Recommended Pima table:

| Model | AUC | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.8155 | 0.7403 | 0.6167 | 0.6852 | 0.6491 |
| LightGBM | 0.8131 | 0.7468 | 0.6316 | 0.6667 | 0.6486 |
| Logistic Regression | 0.8115 | 0.7143 | 0.5806 | 0.6667 | 0.6207 |
| SVM | 0.8115 | 0.7468 | 0.6271 | 0.6852 | 0.6549 |
| Stacked Ensemble | 0.8107 | 0.7403 | 0.6167 | 0.6852 | 0.6491 |

What to say:

- tree ensembles are the strongest and most stable standard baselines,
- the 100k dataset provides the clearest ranking signal,
- the Pima dataset shows tighter margins and greater sensitivity to data scarcity.

This is where the paper earns trust. Keep the claims simple and exact.

### 2.3 Calibration, Uncertainty, and Reliability

This is the first place where the paper can look more like a top-tier journal paper and less like a benchmark leaderboard.

Use the production XGBoost model plus the APS conformal layer as the reliability analysis block.

Currently supported by the repository:

- APS conformal target coverage: 0.9000
- APS empirical coverage: 0.9001
- average set size: 0.9272
- singleton rate: 0.9272
- empty-set rate: 0.0728
- calibration-split AUC for the underlying XGBoost model: 0.9793
- bootstrap 95% CI for that AUC: [0.9756, 0.9823]
- Brier score: 0.0222
- expected calibration error: 0.0013

Recommended table:

| Reliability metric | Value |
|---|---:|
| Target coverage | 0.9000 |
| Empirical coverage | 0.9001 |
| Average prediction-set size | 0.9272 |
| Singleton prediction-set rate | 0.9272 |
| Empty-set rate | 0.0728 |
| Underlying XGBoost AUC | 0.9793 |

Recommended narrative:

"To complement discrimination performance, we evaluated uncertainty using APS-style conformal prediction on the production XGBoost model. The resulting prediction sets achieved empirical coverage aligned with the nominal 90% target while remaining predominantly singleton, indicating that the model can provide calibrated confidence information without collapsing into overly large ambiguous sets."

What a stronger final journal version should still add next:

- threshold-specific PPV/NPV,
- decision-curve analysis,
- abstention analysis for empty sets,
- external validation of calibration.

Implemented artifact files:

- `results/bootstrap_confidence_intervals.csv`
- `results/calibration_summary_xgboost_100k.csv`
- `results/calibration_bins_xgboost_100k.csv`
- `results/plots/calibration_xgboost_100k.png`

### 2.4 Clinical Utility and Operating-Point Analysis

Top-tier journals usually want more than AUC. They want to know how the model behaves when used.

This subsection should be a compact operating-point analysis for the best 100k model.

Include:

- ROC and PR curves,
- one main decision threshold,
- sensitivity/specificity/PPV/NPV at that threshold,
- optional decision-curve analysis if you want stronger translational framing.

Recommended framing:

"The primary ranking metric was AUC, but deployment relevance depends on operating-point behavior. We therefore report thresholded performance for the best 100k model and use conformal sets to identify cases suitable for confident prediction versus abstention."

Important:

- do not bury this in supplements,
- but also do not overload it with many thresholds.

Pick one clinically motivated threshold and one sensitivity-oriented threshold at most.

### 2.5 Clinically Meaningful Feature Dependence

This is the HbA1c ablation section and it should be treated as one of the paper's strongest insights.

Recommended table:

| Model | Full-feature AUC | No-HbA1c AUC | Delta |
|---|---:|---:|---:|
| Attention-ResNet | 0.9653 | 0.8910 | -0.0743 |
| Random Forest | 0.9660 | 0.9011 | -0.0649 |
| Stacked Ensemble | 0.9742 | 0.9227 | -0.0515 |
| XGBoost | 0.9773 | 0.9273 | -0.0500 |
| LightGBM | 0.9786 | 0.9329 | -0.0457 |

Text to emphasize:

- the effect is consistent across model families,
- the drop is too large to dismiss as noise,
- the benchmark is therefore relying on clinically meaningful metabolic signal.

This section is much more valuable than another leaderboard table.

### 2.6 Novelty Validation: EchoCeption Family

This subsection should answer a specific question:

- does the proposed architecture add something scientifically interesting beyond a generic neural baseline?

Use `results/echoception_benchmark.csv`.

Recommended table:

| Variant | Head | AUC | Accuracy | F1 |
|---|---|---:|---:|---:|
| EchoCeptionNet | BCE | 0.9773 | 0.9702 | 0.7909 |
| EchoCeptionSphereNet | Hyperspherical | 0.9770 | 0.9715 | 0.8030 |
| EchoCeptionNet | Focal | 0.9768 | 0.9710 | 0.7975 |

Interpretation:

- BCE gives the best AUC,
- the hyperspherical head gives the best F1,
- the architecture is competitive with the strongest tree models but does not decisively surpass them.

This is the right place to articulate novelty precisely:

- the novelty is architectural,
- the value is competitive performance with a distinct inductive design,
- the contribution is strengthened further by the hyperspherical head analysis.

Do not frame this as global SOTA. Frame it as a validated new architecture that remains close to the best baseline family.

### 2.7 Modern Baselines and Foundation Models

This section should be short and disciplined.

Use it to show that the repository was checked against newer foundation-style baselines, not to turn the paper into a foundation-model paper.

Recommended table:

| Task | Model | Main metric | Result |
|---|---|---|---:|
| 100k risk prediction | TabICLv2 | AUC | 0.9804 |
| 100k risk prediction | LightGBM | AUC | 0.9784 |
| 100k risk prediction | XGBoost | AUC | 0.9754 |
| Shanghai CGM holdout | ST-Attention LSTM | MSE | 42.40 |
| Shanghai CGM holdout | Glucose LSTM | MSE | 89.59 |
| Shanghai CGM holdout | Persistence | MSE | 97.49 |
| Shanghai CGM holdout | TabICLForecaster | MSE | 119.38 |

Interpretation:

- on tabular risk prediction, TabICLv2 is a strong comparison baseline and slightly exceeds the classical baselines in the dedicated run,
- on the CGM prototype benchmark, the repo's ST-Attention LSTM currently outperforms the FM baseline,
- this strengthens the paper's benchmarking depth without changing the main novelty claim.

### 2.8 Error Analysis, Robustness, and Subgroup Analysis

This is the main gap between the current repository and a true top-tier journal submission.

Add this subsection if possible before submission. It should contain:

- subgroup performance by sex,
- subgroup performance by age bands,
- confusion matrix or error buckets for false positives and false negatives,
- sensitivity to random seed or split variation,
- confidence intervals via bootstrap or repeated runs.

Minimum acceptable top-tier upgrade:

- bootstrap 95% CI for AUC on the best 100k model,
- bootstrap CI for the HbA1c ablation delta,
- at least one subgroup table.

This repository now has an initial subgroup artifact:

- `results/subgroup_analysis_xgboost_100k.csv`

If this section is missing, reviewers will likely treat the study as promising but still incomplete.

### 2.9 Prototype Extensions

This subsection must stay last.

Include:

- CGM forecasting branch,
- recommendation layer,
- RL prototype,
- app layer.

Language to use:

"The broader digital-twin stack demonstrates architectural extensibility of the repository, but these components are not yet evaluated with the same rigor as the core tabular benchmark and are therefore best interpreted as prototype extensions."

This section protects the paper from overclaiming.

The repository now includes an offline recommendation-layer ablation that can strengthen this prototype subsection without turning it into a headline claim. Use:

- `results/recommender_ablation.csv`
- `results/recommender_case_studies.csv`
- `results/plots/recommender_ablation_tradeoffs.png`
- `results/recommender_nextgen_ablation.csv`
- `results/recommender_nextgen_case_studies.csv`
- `results/plots/recommender_nextgen_tradeoffs.png`

Recommended prototype table:

| Variant | Mean GI | Calorie error | Simulated reward | Meal-type consistency |
|---|---:|---:|---:|---:|
| Full heuristic | 14.84 | 159.19 | -2.29 | 1.00 |
| No risk adjustment | 15.96 | 140.78 | -3.23 | 1.00 |
| No GI prioritization | 21.81 | 232.54 | -4.22 | 1.00 |
| No meal-type routing | 0.05 | 25.46 | 6.67 | 0.50 |
| RL policy only | 42.93 | 151.80 | -18.59 | 1.00 |
| Hybrid RL + heuristic | 29.10 | 108.44 | -18.16 | 1.00 |

How to interpret this table:

- the full heuristic is the strongest plausible hand-engineered variant,
- removing GI prioritization degrades both the GI profile and the metabolic proxy reward,
- removing meal-type routing appears to improve low-level proxy metrics only because it serves semantically implausible meals, which is why meal-type consistency must be reported,
- the current RL-only and hybrid variants remain weaker than the heuristic recommendation baseline and should be framed as exploratory control extensions.

Recommended figure use:

- include `plots/recommender_ablation_tradeoffs.png` only if space allows,
- otherwise keep the summary table in the prototype subsection and move the case studies to the supplement or appendix.

The repository also now includes a cutting-edge-inspired next-generation planner family that is closer to recent recommendation trends based on guideline constraints, graph-style food relationships, and state-aware reranking. A compact summary from `results/recommender_nextgen_ablation.csv` is:

| Variant | Mean GI | Calorie error | Simulated reward | Final glucose |
|---|---:|---:|---:|---:|
| Full heuristic | 14.84 | 159.19 | -2.29 | 160.62 |
| Guideline planner | 22.65 | 109.61 | -2.65 | 154.32 |
| Guideline + graph | 22.75 | 110.15 | -2.64 | 154.30 |
| Guideline + twin | 21.58 | 117.06 | -2.16 | 152.86 |
| Graph-Twin planner | 22.02 | 117.83 | -2.22 | 153.06 |

Recommended interpretation:

- the strongest next-generation variant is currently the state-aware `Guideline + twin` planner,
- graph regularization is feasible in the repository, but with the current food database it adds little beyond the state-aware reranker,
- compared with the simple heuristic, the new planner family substantially improves calorie-target adherence and lowers final simulated glucose, while sacrificing some of the very low-GI behavior of the hand-tuned heuristic,
- this is best presented as a cutting-edge-inspired systems advance, not as a claim of clinical recommender SOTA.

## 3. Exact Table Order

Use the tables in this order:

1. Main 100k benchmark table
2. Main Pima benchmark table
3. Calibration and APS conformal reliability table
4. HbA1c ablation table
5. EchoCeption family table
6. Foundation-model baseline table
7. Optional subgroup table

That order moves from strongest evidence to more specialized analyses.

## 4. Exact Figure Order

Recommended figures:

1. ROC curves for the top 100k models
2. Precision-recall curves for the top 100k models
3. Calibration plot for the best risk model
4. HbA1c ablation bar chart
5. EchoCeption variant comparison figure
6. System figure showing risk model, uncertainty layer, forecasting module, recommender, and app

If space is tight, keep only figures 1, 3, 4, and 6 in the main paper.

## 5. Statistical Reporting Needed for a Top-Tier Version

The current repository is close on benchmarking depth, but a top-tier journal version should add:

- 95% confidence intervals for all primary metrics,
- DeLong test or bootstrap comparison for top AUC differences,
- calibration error metrics,
- threshold-specific clinical utility metrics,
- subgroup analysis,
- repeated-seed or repeated-split robustness analysis.

If you want one practical minimum upgrade path, do this:

1. add bootstrap CIs for AUC on the top five 100k models,
2. add one calibration figure plus Brier score,
3. add sex and age subgroup analyses for the best model,
4. report APS abstention and singleton-set behavior at the chosen threshold.

## 6. Claim Language to Use

Good:

- "competitive with the strongest tree baselines"
- "clinically meaningful performance dependence on HbA1c"
- "calibrated uncertainty via APS conformal prediction"
- "prototype digital-twin extensions"

Avoid:

- "state of the art" unless you add a much stronger external comparison protocol,
- "clinically validated deployment",
- "closed-loop metabolic control",
- "foundation models outperform all baselines" without repeated-seed confirmation.

## 7. Journal-Grade Results Narrative

If you want the whole results section to sound top-tier, keep the narrative arc simple:

1. establish the benchmark,
2. show the model is reliable, not just accurate,
3. show the signal is clinically meaningful,
4. validate the novelty,
5. place newer baselines in context,
6. close with honest prototype extensions.

That is the version of this project that is most defensible in peer review.

## 8. Immediate Next Steps for This Repo

Before submission, the highest-value additions are:

- bootstrap confidence intervals,
- calibration figure plus Brier/ECE,
- subgroup analysis,
- thresholded clinical utility table,
- one figure for APS conformal set behavior,
- one compact figure for EchoCeption vs EchoCeptionSphereNet.

If those are added, the results section becomes much closer to a genuine top-tier journal presentation.
