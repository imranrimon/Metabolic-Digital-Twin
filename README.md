# 🩺 Metabolic Digital Twin (Grandmaster Edition)

A state-of-the-art (SOTA) metabolic management system that integrates predictive diagnostics, continuous forecasting, and proactive dietary control. This project transforms raw clinical data into a "Digital Twin" capable of forecasting glucose trends and optimizing dietary policies.

**Status**: ✅ Complete (RTX 8000 Hyper-Scale Edition)
**Top Performance**: **0.9908 AUC-ROC** (EchoCeption-XL on RTX 8000) 🥇🏆

## 🚀 Key Features

### 1. 🧠 Hyper-Scale Neural Engines
*   **EchoCeption-XL (New SOTA)**: A massive **88M parameter** hybrid model scaling the **Echo State Network** reservoir to **4096+ neurons** with Pre-Reservoir **Self-Attention**.
    *   **Performance**: **0.9908 AUC** (Beating 97.9% Baseline).
*   **Population Graph Learning**: A Transductive GNN trained on a **100,000-node graph** (RTX 8000 Exclusive).
*   **Grandmaster Ensemble**: Stacking meta-learner combining Optimized XGBoost, LightGBM, and FT-Transformer.

### 2. 🤖 AI Dietitian (Reinforcement Learning)
*   **Offline RL Agent**: A DQN-based policy trained on physiological states to recommend optimal macronutrient ratios.
*   **Interactive Demo**: Run `python src/show_recommendations.py` to input your metrics and get a custom meal plan.

### 3. 📱 Premium Patient Dashboard
*   **Glassmorphic UI**: High-end Android interface built with Vanilla JS/CSS3.
*   **Real-time Inference**: Powered by **FastAPI**.
*   **One-Click Launch**: Run `run_web_app.bat` to start the entire system.

## 📂 Project Structure

```text
├── mobile_app/           # Full-stack Patient Dashboard
│   ├── backend/          # FastAPI predictive server
│   └── frontend/         # Glassmorphic UI
├── src/                  # Core Neural Engines
│   ├── train_hyperscale.py # The 0.9908 AUC script (RTX 8000)
│   ├── train_graph_sota.py # Population Graph Training
│   ├── show_recommendations.py # Interactive Dietitian Demo
│   └── models_novel.py   # KAN, Mamba, TFT, EchoCeption architectures
├── results/              # Benchmark csvs
│   └── hyperscale_benchmark.csv # Proof of 0.9908 AUC
├── project_paper.tex     # IEEE Conference Paper
└── run_web_app.bat       # Easy Launcher
```

## 📊 Benchmark Hall of Fame

| Rank | Model Strategy | AUC-ROC | Innovation |
|------|----------------|---------|------------|
| 🥇 | **EchoCeption-XL (RTX 8000)** | **0.9908** | **Hyper-Scale Reservoir + Attention** |
| 🥈 | EchoCeption-XL (5070Ti) | 98.45% | Large Reservoir (4096 dim) |
| 🥉 | Stacking Ensemble | 97.90% | Meta-Learning |
| 4 | EchoCeptionNet (Base) | 97.72% | Novel ESN+Inception Hybrid |
| 5 | Population Graph | 88.89% | Transductive GNN |

## 🛠 Tech Stack
- **Deep Learning**: PyTorch, `torchcde`, Echo State Networks
- **Hardware**: **NVIDIA RTX 8000** (48GB VRAM) & RTX 5070Ti
- **Backend API**: FastAPI, Uvicorn
- **Frontend**: HTML5, CSS3, Chart.js

## 🚦 How to Reproduce SOTA Results

### 1. Install Dependencies
```bash
pip install torch torchcde xgboost lightgbm pandas scikit-learn optuna pytorch-tabnet fastapi uvicorn
```

### 2. Run the Hyper-Scale Pipeline (Requires High-VRAM GPU)
```bash
# Train the 88M parameter EchoCeption-XL model
python src/train_hyperscale.py
```

### 3. Run the Interactive Dietitian
```bash
python src/show_recommendations.py
```

### 4. Launch the Web App
```bash
run_web_app.bat
```

---
*Developed by Imran Rimon - Transformative Metabolic Intelligence.*
