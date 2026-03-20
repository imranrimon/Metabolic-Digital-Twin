# Metabolic Digital Twin Prototype: User Guide

This guide explains how to run the dashboard as a prototype demo of the research pipeline.

## What the App Currently Demonstrates

- risk prediction through the backend API,
- diet-strategy selection when the policy model is available,
- a frontend shell for displaying results.

The dashboard should be treated as a prototype interface. Some backend modules are optional and may fall back if the larger model weights are not available.

## Quick Start

### 1. Start the backend

From the project root:

```bash
python mobile_app/backend/api.py
```

Then open:

`http://localhost:8001/health`

You should see a JSON response showing which models were loaded.
If conformal calibration is active, the health response also reports the current conformal method.

### 2. Open the frontend

Open:

`mobile_app/frontend/index.html`

in a browser such as Chrome or Edge.

## What to Expect

When you submit a profile:

- the risk endpoint uses the production XGBoost model if it is available,
- the risk response may include an APS-style conformal prediction set, per-label p-values, and ranked labels when the conformal calibration artifact is present,
- the diet endpoint uses the policy model if it is loaded,
- trend or advanced components may fall back depending on which weights exist locally.

## Troubleshooting

- If the frontend does not update, confirm the backend is still running.
- If the API reports only partial model availability, the app is still usable as a prototype but some outputs may be simplified.
- If you see fallback messages in the backend logs, that means the corresponding model weights were not loaded.

## Technical Note

The frontend communicates with the FastAPI backend through JSON requests to endpoints such as:

- `/predict/risk`
- `/recommend/diet`
- `/recommend/meals`
- `/health`
