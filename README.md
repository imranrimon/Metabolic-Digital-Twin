# 🩺 Metabolic Digital Twin

A state-of-the-art (SOTA) metabolic management system that integrates predictive diagnostics, continuous forecasting, and proactive dietary control. This project transforms raw clinical data into a "Digital Twin" capable of forecasting glucose trends and optimizing dietary policies using Reinforcement Learning.

**Status**: ✅ Complete (Hyper-Scale Edition)
**Top Performance**: **98.45% AUC-ROC** (EchoCeption-XL) 🥇

## 🚀 Key Features

### 1. 🧠 Hyper-Scale Neural Engines
*   **EchoCeption-XL (New SOTA)**: A massive **88M parameter** hybrid model scaling the **Echo State Network** reservoir to **4096 neurons** with Pre-Reservoir **Self-Attention**.
*   **Grandmaster Ensemble**: A stacking meta-learner combining Optimized XGBoost, LightGBM, and FT-Transformer. **97.90% AUC**.
*   **Neural Controlled Differential Equations (CDE)**: Continuous-time glucose forecasting that models metabolic dynamics as an irregular time-series.
*   **Interactive Novel Architectures**: Experimental implementations of **KAN**, **Mamba**, and **TabNet**.

### 2. 🤖 AI Dietitian (Reinforcement Learning)
*   **Offline RL Agent**: A DQN-based policy trained on physiological states to recommend optimal macronutrient ratios.
*   **Smart Meal Planner**: Suggests specific foods (low GI) based on the RL agent's strategy.

### 3. 📱 Premium Patient Dashboard
*   **Glassmorphic UI**: A high-end, mobile-first Android interface built with Vanilla JS and CSS3.
*   **Real-time Inference**: Powered by a robust **FastAPI** backend serving SOTA model predictions instantly.

## 📂 Project Structure

```text
├── mobile_app/           # Full-stack Patient Dashboard
│   ├── backend/          # FastAPI predictive server
│   └── frontend/         # Glassmorphic UI (HTML/CSS/JS)
├── src/                  # Core Neural Engines & Preprocessing
│   ├── train_hyperscale.py # The 98.45% AUC script (EchoCeption-XL)
│   ├── models_novel.py   # KAN, Mamba, TFT, EchoCeption architectures
│   └── grandmaster_*.py  # Advanced feature engineering & optimization
├── results/              # Benchmark csvs & Performance plots
│   └── hyperscale_benchmark.csv # Final 98.45% AUC results
├── project_paper.tex     # IEEE Conference Paper (Ready to Submit)
└── README.md             # Project documentation
```

## 📊 Benchmark Hall of Fame

| Rank | Model Strategy | AUC-ROC | Innovation |
|------|----------------|---------|------------|
| 🥇 | **EchoCeption-XL (5070Ti)** | **98.45%** | **Massive Reservoir (4096 dim) + Attention** |
| 🥈 | Stacking Ensemble | 97.90% | Meta-Learning (XGB+LGBM+DL) |
| 🥉 | XGBoost (Optuna) | 97.89% | Hyperparameter Optimization |
| 4 | EchoCeptionNet (Base) | 97.72% | Novel ESN+Inception Hybrid |
| 5 | FT-Transformer | 97.16% | Deep Learning SOTA |
| 6 | Logistic Regression | 95.40% | Baseline |

## 🛠 Tech Stack
- **Deep Learning**: PyTorch, `torchcde`, Echo State Networks, Self-Attention
- **Machine Learning**: XGBoost, LightGBM, Scikit-learn, Optuna (AutoML)
- **Backend API**: FastAPI, Uvicorn
- **Frontend**: HTML5, CSS3, Chart.js
- **Hardware**: **NVIDIA RTX 5070Ti** (Crucial for EchoCeption-XL training)

## 🚦 How to Reproduce SOTA Results

### 1. Install Dependencies
```bash
pip install torch torchcde xgboost lightgbm pandas scikit-learn optuna pytorch-tabnet fastapi uvicorn
```

### 2. Run the Hyper-Scale Pipeline (Requires GPU with >12GB VRAM)
```bash
# Train the 88M parameter EchoCeption-XL model
python src/train_hyperscale.py
```

### 3. Launch the Mobile App
```bash
cd mobile_app/backend
python api.py
# Then open mobile_app/frontend/index.html
```

---
*Developed by Imran Rimon - Transformative Metabolic Intelligence.*
