# 🩺 Metabolic Digital Twin

A state-of-the-art (SOTA 2024/2025) metabolic management system that integrates predictive diagnostics with proactive dietary control. This project transforms raw clinical data into a "Digital Twin" capable of forecasting glucose trends and optimizing dietary policies using Reinforcement Learning.

## 🚀 Key Features

*   **FT-Transformer (Risk Engine)**: Superior tabular risk prediction for diabetes, outperforming traditional ML on large-scale datasets.
*   **Neural Controlled Differential Equations (Forecaster)**: Continuous-time glucose modeling that treats metabolic behavior as a smooth dynamical system.
*   **Offline Reinforcement Learning (Diet Agent)**: A DQN-based agent trained to recommend meal compositions that maximize "Time in Range" (TIR).
*   **Mobile Patient Dashboard**: A premium, glassmorphic Android-first web app (FastAPI + Vanilla JS) for real-time monitoring and AI diet suggestions.

## 📂 Project Structure

```text
├── mobile_app/           # Full-stack Patient Dashboard
│   ├── backend/          # FastAPI predictive server
│   └── frontend/         # Glassmorphic UI (HTML/CSS/JS)
├── src/                  # Core Neural Engines & Preprocessing
├── results/              # Ablation benchmarks & Performance plots
├── models/               # Saved SOTA weights (.pth) - [Excluded from Git]
└── README.md             # Project documentation
```

## 🛠 Tech Stack
- **Deep Learning**: PyTorch, `torchcde`, `rtdl-revisiting-models`
- **Analytics**: Scikit-learn, XGBoost, LightGBM
- **Backend**: FastAPI, Uvicorn
- **Frontend**: Vanilla HTML5, CSS3 (Glassmorphism), Chart.js
- **Hardware**: NVIDIA RTX 5070Ti (CUDA 12.9)

## 🚦 Getting Started

### 1. Requirements
```bash
pip install torch torchcde rtdl_revisiting_models fastapi uvicorn xgboost lightgbm pandas scikit-learn
```

### 2. Launching the Backend
```bash
cd mobile_app/backend
python api.py
```

### 3. Launching the Dashboard
Open `mobile_app/frontend/index.html` in any modern browser.

## 📊 Ablation Results
Our SOTA pipeline achieved a **15% reduction** in glucose forecasting error (CDE vs LSTM) and competitive AUC-ROC on the 100k Diabetes dataset using the FT-Transformer. Detailed results can be found in `results/comprehensive_ablation_results.csv`.

---
*Developed by Imran Rimon - Transformative Metabolic Intelligence.*
