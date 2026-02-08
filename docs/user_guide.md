# 📱 Metabolic Digital Twin: User Guide

This guide explains how to use the "Best-in-Class" mobile dashboard to manage metabolic health.

## 🚀 Quick Start

### 1. The Backend (Already Running)
The **FastAPI Inference Server** is currently running on your system at:
`http://localhost:8001`

*   **To check health**: Open [http://localhost:8001/health](http://localhost:8001/health) in your browser. You should see `{"status":"online"}`.

### 2. The Frontend (Open Now)
Simply open the following file in your web browser (Chrome or Edge recommended):
**[index.html](file:///f:/Diabetics%20Project/mobile_app/frontend/index.html)**

---

## 🧪 How to Verify it's Working

1.  **Open the App**: You will see a dark-mode glassmorphic dashboard.
2.  **Input Health Data**:
    *   Enter **Current Glucose** (e.g., `180`).
    *   Enter **HbA1c** (e.g., `7.5`).
3.  **Click "Sync Digital Twin"**:
    *   The **Risk Gauge** will animate and show a percentage (e.g., `81.4%`). This is computed live by the **FT-Transformer**.
    *   The **AI Recommendation** will update to suggest a meal strategy (e.g., `Low Carb`) based on your **RL Policy**.
    *   The **Continuous Forecast** chart will update to show your predicted trajectory.

## 🛠 Troubleshooting
- **CORS Error**: If the gauge doesn't update, ensure the backend window is still open.
- **Model Load**: If you see "Fallback Active" in the terminal, the system is using robust heuristic logic while the largest weights (CDE) are finalized.

---
**Technical Note**: The app communicates via standard JSON POST requests to `http://localhost:8001/predict/risk` and `http://localhost:8001/recommend/diet`.
