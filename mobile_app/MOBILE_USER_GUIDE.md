# Metabolic Digital Twin Prototype: Mobile Guide

This dashboard is a prototype companion to the research code in the repository. It is useful for demonstrations, but it should not be described as a validated clinical product.

## Running the Demo

### Backend

Start the API from the project root:

```bash
python mobile_app/backend/api.py
```

Then verify the service:

`http://localhost:8001/health`

### Frontend

Open:

`mobile_app/frontend/index.html`

in a browser.

## What the Demo Uses

- Risk prediction: production XGBoost model when loaded successfully
- Conformal uncertainty: APS-style prediction sets, per-label p-values, and ranked labels when the calibration artifact is available
- Meal strategy selection: policy model when available
- Meal suggestions: recommender backed by the food database

Some advanced modules may use fallback behavior if their weights are missing.
The health endpoint also reports the active conformal method when calibration is loaded.

## Notes

- The backend is designed for local demo use.
- The app helps show the flow from profile input to model output.
- The app is not a substitute for the paper benchmark; it is the interface layer of the project.
