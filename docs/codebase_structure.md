# Codebase Structure

This repository now follows a safer, more maintainable split between reusable package code and runnable experiment scripts.

## Core layout

- `src/metabolic_twin/`
  - `features/`: shared feature engineering utilities
  - `risk/`: production risk pipeline and conformal utilities
  - `recommendation/`: heuristic, RL, and next-generation meal recommendation logic
  - `utils/`: shared training and checkpointing helpers
- `src/`
  - runnable scripts and backward-compatible module wrappers
- `results/`
  - paper-facing outputs, plots, and inspection artifacts
- `models/`
  - serialized production and benchmark metadata artifacts
- `docs/`
  - paper framing, walkthroughs, and structure documentation
- `mobile_app/`
  - backend and frontend demo application

## Compatibility policy

Existing commands such as `python src/run_all_experiments.py` still work.
The flat `src/*.py` module names remain as wrappers so older scripts do not break.

## Output policy

- paper figures and benchmark CSVs live under `results/`
- inspection logs and EDA graphics live under `results/inspection/`
- ad hoc local run logs under `results/reports/` are ignored from Git
