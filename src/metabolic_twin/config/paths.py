"""
Centralized project paths for datasets, model artifacts, and result outputs.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
REPORTS_DIR = RESULTS_DIR / "reports"
INSPECTION_DIR = RESULTS_DIR / "inspection"
INSPECTION_LOGS_DIR = INSPECTION_DIR / "logs"
INSPECTION_EDA_DIR = INSPECTION_DIR / "eda"

PIMA_DATA_PATH = DATA_DIR / "pima-indians-diabetes-database" / "diabetes.csv"
DIABETES_100K_DATA_PATH = DATA_DIR / "diabetes-prediction-dataset" / "diabetes_prediction_dataset.csv"
SHANGHAI_DATASET_DIR = DATA_DIR / "shanghai_dataset"
SHANGHAI_T1DM_DIR = SHANGHAI_DATASET_DIR / "Shanghai_T1DM"
SHANGHAI_T2DM_DIR = SHANGHAI_DATASET_DIR / "Shanghai_T2DM"
SHANGHAI_TOTAL_DATA_PATH = DATA_DIR / "shanghai_total.csv"
SHANGHAI_INSPECTION_SAMPLE_PATH = SHANGHAI_T1DM_DIR / "1001_0_20210730.xlsx"
CGMACROS_DATA_DIR = DATA_DIR / "cgmacros" / "data_volume" / "CGMacros"
FOOD_DB_PATH = SRC_DIR / "food_db.json"

BEST_HYPERPARAMS_PATH = SRC_DIR / "best_hyperparams.pkl"
STACKING_META_MODEL_PATH = SRC_DIR / "stacking_meta_model.pkl"

PRODUCTION_XGBOOST_PATH = MODELS_DIR / "production_xgboost.json"
PRODUCTION_FEATURES_PATH = MODELS_DIR / "production_features.pkl"
PRODUCTION_PREPROCESS_PATH = MODELS_DIR / "production_preprocess.pkl"
PRODUCTION_CONFORMAL_PATH = MODELS_DIR / "production_conformal.pkl"

ECHOCEPTION_FOCAL_CHECKPOINT_PATH = MODELS_DIR / "echoception_focal_best.pth"
ECHOCEPTION_BCE_CHECKPOINT_PATH = MODELS_DIR / "echoception_bce_best.pth"
ECHOCEPTION_SPHERE_CHECKPOINT_PATH = MODELS_DIR / "echoception_sphere_best.pth"
GLUCOSE_LSTM_TEMPORAL_HOLDOUT_PATH = MODELS_DIR / "glucose_lstm_temporal_holdout.pth"
ST_ATTENTION_TEMPORAL_HOLDOUT_PATH = MODELS_DIR / "st_attention_lstm_temporal_holdout.pth"
GRAPH_SOTA_CHECKPOINT_PATH = MODELS_DIR / "graph_sota_rtx8000.pth"
HYPERSCALE_CHECKPOINT_PATH = MODELS_DIR / "echoception_xl_5070ti.pth"

ATTENTION_RESNET_RISK_CHECKPOINT_PATH = PROJECT_ROOT / "attention_resnet_risk.pth"
ST_ATTENTION_LSTM_CHECKPOINT_PATH = PROJECT_ROOT / "st_attention_lstm.pth"
PPGR_CHECKPOINT_PATH = PROJECT_ROOT / "ppgr_model.pth"
GLUCOSE_LSTM_CHECKPOINT_PATH = PROJECT_ROOT / "glucose_lstm.pth"
KAN_CHECKPOINT_PATH = PROJECT_ROOT / "kan_diabetes.pth"
NEURAL_CDE_CHECKPOINT_PATH = PROJECT_ROOT / "neural_cde_glucose.pth"
FT_TRANSFORMER_RISK_CHECKPOINT_PATH = PROJECT_ROOT / "ft_transformer_risk.pth"
RL_POLICY_CHECKPOINT_PATH = PROJECT_ROOT / "metabolic_policy.pth"
TABNET_MODEL_PATH = PROJECT_ROOT / "tabnet_diabetes"

METRICS_SUMMARY_PATH = RESULTS_DIR / "metrics_summary.csv"
GRANDMASTER_BENCHMARK_PATH = RESULTS_DIR / "grandmaster_benchmark.csv"
ECHOCEPTION_BENCHMARK_PATH = RESULTS_DIR / "echoception_benchmark.csv"
TABULAR_FM_BENCHMARK_PATH = RESULTS_DIR / "tabular_foundation_benchmark.csv"
CGM_FM_BENCHMARK_PATH = RESULTS_DIR / "cgm_foundation_benchmark.csv"
BOOTSTRAP_CI_PATH = RESULTS_DIR / "bootstrap_confidence_intervals.csv"
SUBGROUP_ANALYSIS_PATH = RESULTS_DIR / "subgroup_analysis_xgboost_100k.csv"
CALIBRATION_SUMMARY_PATH = RESULTS_DIR / "calibration_summary_xgboost_100k.csv"
CALIBRATION_BINS_PATH = RESULTS_DIR / "calibration_bins_xgboost_100k.csv"
CALIBRATION_FIG_PATH = PLOTS_DIR / "calibration_xgboost_100k.png"
RECOMMENDER_ABLATION_PATH = RESULTS_DIR / "recommender_ablation.csv"
RECOMMENDER_NEXTGEN_ABLATION_PATH = RESULTS_DIR / "recommender_nextgen_ablation.csv"
COMPREHENSIVE_ABLATION_RESULTS_PATH = RESULTS_DIR / "comprehensive_ablation_results.csv"
NOVEL_ARCHITECTURE_BENCHMARK_PATH = RESULTS_DIR / "novel_architecture_benchmark.csv"
NOVEL_7_MODEL_BENCHMARK_PATH = RESULTS_DIR / "novel_7_model_benchmark.csv"
TOP_100K_MODEL_AUC_FIG_PATH = PLOTS_DIR / "top_100k_model_auc.png"
HBA1C_ABLATION_FIG_PATH = PLOTS_DIR / "hba1c_ablation_auc_drop.png"

BASELINES_RESULTS_TEMPLATE = "baselines_{dataset_name}.csv"
ENSEMBLE_RESULTS_TEMPLATE = "ensemble_{dataset_name}.csv"
MLP_RESULTS_TEMPLATE = "mlp_{dataset_name}.csv"


def result_path(filename: str) -> Path:
    return RESULTS_DIR / filename
