# 🩺 Metabolic Digital Twin

A state-of-the-art (SOTA) metabolic management system that integrates predictive diagnostics, continuous forecasting, and proactive dietary control. This project transforms raw clinical data into a "Digital Twin" capable of forecasting glucose trends and optimizing dietary policies using Reinforcement Learning.

**Status**: ✅ Complete (Kaggle Grandmaster Edition)
**Top Performance**: **97.90% AUC-ROC** (Stacking Ensemble)

## 🚀 Key Features

### 1. 🧠 Advanced Neural Engines
*   **Kaggle Grandmaster Ensemble**: A stacking meta-learner combining Optimized XGBoost, LightGBM, and FT-Transformer (Deep Learning). **Achieved 97.90% AUC**.
*   **Neural Controlled Differential Equations (CDE)**: Continuous-time glucose forecasting that models metabolic dynamics as irregular time-series.
*   **Interactive Novel Architectures**: Experimental implementations of **KAN (Kolmogorov-Arnold Networks)**, **Mamba (State Space Models)**, and **TabNet** (Interpretable Attention).

### 2. 🤖 AI Dietitian (Reinforcement Learning)
*   **Offline RL Agent**: A DQN-based policy trained on physiological states to recommend optimal macronutrient ratios.
*   **Goal**: Maximize "Time in Range" (TIR) and minimize glycemic variability.

### 3. 📱 Premium Patient Dashboard
*   **Glassmorphic UI**: A high-end, mobile-first Android interface built with Vanilla JS and CSS3.
*   **Real-time Inference**: Powered by a robust **FastAPI** backend serving SOTA model predictions instantly.

## 📂 Project Structure

```text
├── mobile_app/           # Full-stack Patient Dashboard
│   ├── backend/          # FastAPI predictive server
│   └── frontend/         # Glassmorphic UI (HTML/CSS/JS)
├── src/                  # Core Neural Engines & Preprocessing
│   ├── models_novel.py   # KAN, Mamba, TFT, TabNet architectures
│   ├── grandmaster_*.py  # Advanced feature engineering & optimization
│   └── train_stacking.py # Stacking Ensemble pipeline
├── results/              # Benchmark csvs & Performance plots
│   └── grandmaster_benchmark.csv # Final 97.9% AUC results
└── README.md             # Project documentation
```

## 📊 Benchmark Hall of Fame

| Rank | Model Strategy | AUC-ROC | Innovation |
|------|----------------|---------|------------|
| 🥇 | **Stacking Ensemble** | **97.90%** | **Meta-Learning (XGB+LGBM+DL)** |
| 🥈 | XGBoost (Optuna) | 97.89% | Hyperparameter Optimization |
| 🥉 | LightGBM (Optuna) | 97.77% | Interaction Features |
| 4 | FT-Transformer | 97.16% | Deep Learning SOTA |
| 5 | TabNet | 96.14% | Interpretable Attention |
| 6 | KAN (2024) | 94.40% | Learnable Activations |

## 🛠 Tech Stack
- **Deep Learning**: PyTorch, `torchcde`, FT-Transformer, KAN, Mamba
- **Machine Learning**: XGBoost, LightGBM, Scikit-learn, Optuna (AutoML)
- **Backend API**: FastAPI, Uvicorn
- **Frontend**: HTML5, CSS3, Chart.js
- **Hardware**: NVIDIA RTX 5070Ti (CUDA 12.9)

## 🚦 How to Reproduce SOTA Results

### 1. Install Dependencies
```bash
pip install torch torchcde xgboost lightgbm pandas scikit-learn optuna pytorch-tabnet fastapi uvicorn
```

### 2. Run the Grandmaster Pipeline
```bash
# Step 1: Optimize Hyperparameters (Takes ~30 mins)
python src/optimize_hyperparams.py

# Step 2: Train Stacking Ensemble & Generate Benchmark
python src/train_stacking.py
```

### 3. Launch the Mobile App
```bash
cd mobile_app/backend
python api.py
# Then open mobile_app/frontend/index.html
```

---
*Developed by Imran Rimon - Transformative Metabolic Intelligence.*
