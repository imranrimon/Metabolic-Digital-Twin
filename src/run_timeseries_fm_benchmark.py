"""
Time-series foundation-model benchmark for the CGM branch.

Uses a patient-level temporal holdout on the Shanghai CGM data and compares:
- Persistence
- Glucose LSTM
- ST-Attention LSTM
- TabICLForecaster

The neural baselines train on rolling next-step windows from each patient's
history prior to the holdout horizon. The foundation-model forecaster predicts
the held-out horizon from the same recent context.
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from metabolic_twin.config import (
    CGM_FM_BENCHMARK_PATH,
    GLUCOSE_LSTM_TEMPORAL_HOLDOUT_PATH,
    SHANGHAI_TOTAL_DATA_PATH,
    ST_ATTENTION_TEMPORAL_HOLDOUT_PATH,
)
from models import LSTMForecaster, STAttentionLSTM
from training_utils import ValidationCheckpoint, load_model_state, progress, update_progress


DATA_PATH = SHANGHAI_TOTAL_DATA_PATH
RESULTS_PATH = CGM_FM_BENCHMARK_PATH
MODEL_PATHS = {
    "Glucose LSTM": GLUCOSE_LSTM_TEMPORAL_HOLDOUT_PATH,
    "ST-Attention LSTM": ST_ATTENTION_TEMPORAL_HOLDOUT_PATH,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-length", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--train-stride", type=int, default=4)
    parser.add_argument("--max-series", type=int, default=0, help="Optional cap on evaluated patient series.")
    parser.add_argument("--max-train-windows", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None, help="Torch device override for neural baselines.")
    parser.add_argument("--fm-device", type=str, default=None, help="Optional TabICL device override.")
    parser.add_argument("--fm-estimators", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_shanghai_frame():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values(["patient_id", "Date"]).reset_index(drop=True)


def build_temporal_holdout(df, context_length: int, horizon: int, max_series: int = 0):
    context_frames = []
    future_frames = []
    eval_contexts = []
    eval_targets = []
    train_series = []
    kept_series = 0

    for patient_id, patient_df in df.groupby("patient_id", sort=True):
        patient_df = patient_df.sort_values("Date").reset_index(drop=True)
        if len(patient_df) < context_length + horizon:
            continue

        history_df = patient_df.iloc[:-horizon].copy()
        context_df = history_df.iloc[-context_length:].copy()
        future_df = patient_df.iloc[-horizon:].copy()

        context_frames.append(
            context_df.rename(
                columns={
                    "patient_id": "item_id",
                    "Date": "timestamp",
                    "glucose": "target",
                }
            )[["item_id", "timestamp", "target"]]
        )
        future_frames.append(
            future_df.rename(
                columns={
                    "patient_id": "item_id",
                    "Date": "timestamp",
                    "glucose": "target",
                }
            )[["item_id", "timestamp", "target"]]
        )

        eval_contexts.append(context_df["glucose"].to_numpy(dtype=float))
        eval_targets.append(future_df["glucose"].to_numpy(dtype=float))
        train_series.append(history_df["glucose"].to_numpy(dtype=float))
        kept_series += 1

        if max_series and kept_series >= max_series:
            break

    if not context_frames:
        raise RuntimeError("No eligible Shanghai series found for the requested context and horizon.")

    return {
        "context_df": pd.concat(context_frames, ignore_index=True),
        "future_truth_df": pd.concat(future_frames, ignore_index=True),
        "eval_contexts": np.asarray(eval_contexts, dtype=float),
        "eval_targets": np.asarray(eval_targets, dtype=float),
        "train_series": train_series,
        "num_series": kept_series,
    }


def build_train_windows(train_series, context_length: int, train_stride: int, max_train_windows: int, seed: int):
    windows = []
    targets = []

    for series in train_series:
        if len(series) <= context_length:
            continue

        for start in range(0, len(series) - context_length, train_stride):
            stop = start + context_length
            if stop >= len(series):
                break
            windows.append(series[start:stop])
            targets.append(series[stop])

    if not windows:
        raise RuntimeError("No training windows were created from the Shanghai holdout histories.")

    windows = np.asarray(windows, dtype=float)
    targets = np.asarray(targets, dtype=float)

    if max_train_windows and len(windows) > max_train_windows:
        rng = np.random.default_rng(seed)
        keep_idx = np.sort(rng.choice(len(windows), size=max_train_windows, replace=False))
        windows = windows[keep_idx]
        targets = targets[keep_idx]

    return windows, targets


def inverse_transform_matrix(scaler: MinMaxScaler, values: np.ndarray) -> np.ndarray:
    return scaler.inverse_transform(values.reshape(-1, 1)).reshape(values.shape)


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray):
    mse = mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1))
    return {
        "MAE": mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
    }


def collect_result(model_name, metrics, runtime, num_series, context_length, horizon, train_windows=0, protocol=None):
    return {
        "Model": model_name,
        "MAE": metrics["MAE"],
        "MSE": metrics["MSE"],
        "RMSE": metrics["RMSE"],
        "Time(s)": runtime,
        "EvalSeries": num_series,
        "TrainWindows": train_windows,
        "ContextLength": context_length,
        "Horizon": horizon,
        "Protocol": protocol or "Patient temporal holdout",
    }


def persistence_forecast(contexts: np.ndarray, horizon: int) -> np.ndarray:
    return np.repeat(contexts[:, -1:], horizon, axis=1)


def recursive_neural_forecast(model, scaled_contexts: np.ndarray, horizon: int, device: torch.device):
    rolling_context = torch.FloatTensor(scaled_contexts).unsqueeze(-1).to(device)
    preds = []

    model.eval()
    with torch.no_grad():
        for _ in range(horizon):
            step_pred = model(rolling_context).squeeze(-1)
            preds.append(step_pred.cpu().numpy())
            rolling_context = torch.cat([rolling_context[:, 1:, :], step_pred[:, None, None]], dim=1)

    return np.stack(preds, axis=1)


def train_neural_baseline(
    model_name: str,
    model_cls,
    train_X_scaled: np.ndarray,
    train_y_scaled: np.ndarray,
    eval_contexts_raw: np.ndarray,
    eval_targets_raw: np.ndarray,
    scaler: MinMaxScaler,
    args,
):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_X_fit, val_X_fit, train_y_fit, val_y_fit = train_test_split(
        train_X_scaled,
        train_y_scaled,
        test_size=0.15,
        random_state=args.seed,
    )

    dataset = TensorDataset(
        torch.FloatTensor(train_X_fit).unsqueeze(-1),
        torch.FloatTensor(train_y_fit).unsqueeze(-1),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = model_cls().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    checkpoint = ValidationCheckpoint(MODEL_PATHS[model_name], metric_name="val_mse", mode="min")

    start = time.time()
    epoch_bar = progress(range(1, args.epochs + 1), desc=model_name)
    for epoch in epoch_bar:
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_context = torch.FloatTensor(val_X_fit).unsqueeze(-1).to(device)
            val_preds = model(val_context).squeeze(-1).cpu().numpy()
        val_mse = mean_squared_error(val_y_fit, val_preds)
        avg_train_loss = epoch_loss / len(loader)
        checkpoint.update(
            model,
            epoch,
            val_mse,
            extra_metadata={"train_loss": avg_train_loss},
        )
        update_progress(epoch_bar, train_loss=avg_train_loss, val_mse=val_mse)

    load_model_state(model, MODEL_PATHS[model_name], map_location=device)

    scaled_eval_contexts = scaler.transform(eval_contexts_raw.reshape(-1, 1)).reshape(eval_contexts_raw.shape)
    pred_scaled = recursive_neural_forecast(model, scaled_eval_contexts, args.horizon, device)
    pred_raw = inverse_transform_matrix(scaler, pred_scaled)
    runtime = time.time() - start

    metrics = evaluate_forecast(eval_targets_raw, pred_raw)
    return collect_result(
        model_name=model_name,
        metrics=metrics,
        runtime=runtime,
        num_series=len(eval_contexts_raw),
        context_length=args.context_length,
        horizon=args.horizon,
        train_windows=len(train_X_fit),
    )


def run_tabicl_forecaster(context_df: pd.DataFrame, future_truth_df: pd.DataFrame, args):
    try:
        from tabicl import TabICLForecaster
    except ImportError as exc:  # pragma: no cover - runtime dependency check
        raise ImportError(
            "TabICL forecast extras are not installed. Run: python -m pip install \"tabicl[forecast]\""
        ) from exc

    if TabICLForecaster is None:
        raise ImportError("TabICL forecast extras are not installed. Run: python -m pip install \"tabicl[forecast]\"")

    model = TabICLForecaster(
        max_context_length=args.context_length,
        tabicl_config={
            "n_estimators": args.fm_estimators,
            "device": args.fm_device,
            "allow_auto_download": True,
            "verbose": False,
            "random_state": args.seed,
        },
    )

    future_df = future_truth_df[["item_id", "timestamp"]].copy()

    start = time.time()
    predictions = model.predict_df(context_df=context_df, future_df=future_df)
    runtime = time.time() - start

    pred_df = predictions.reset_index()[["item_id", "timestamp", "target"]]
    merged = future_truth_df.merge(
        pred_df,
        on=["item_id", "timestamp"],
        how="inner",
        suffixes=("_true", "_pred"),
    ).sort_values(["item_id", "timestamp"])

    if len(merged) != len(future_truth_df):
        raise RuntimeError("TabICL predictions did not align with all holdout timestamps.")

    metrics = evaluate_forecast(
        merged["target_true"].to_numpy(dtype=float),
        merged["target_pred"].to_numpy(dtype=float),
    )
    return collect_result(
        model_name="TabICLForecaster",
        metrics=metrics,
        runtime=runtime,
        num_series=context_df["item_id"].nunique(),
        context_length=args.context_length,
        horizon=args.horizon,
    )


def main():
    args = parse_args()
    set_seed(args.seed)

    df = load_shanghai_frame()
    holdout = build_temporal_holdout(
        df,
        context_length=args.context_length,
        horizon=args.horizon,
        max_series=args.max_series,
    )
    train_X_raw, train_y_raw = build_train_windows(
        holdout["train_series"],
        context_length=args.context_length,
        train_stride=args.train_stride,
        max_train_windows=args.max_train_windows,
        seed=args.seed,
    )

    scaler = MinMaxScaler()
    scaler.fit(np.concatenate(holdout["train_series"]).reshape(-1, 1))
    train_X_scaled = scaler.transform(train_X_raw.reshape(-1, 1)).reshape(train_X_raw.shape)
    train_y_scaled = scaler.transform(train_y_raw.reshape(-1, 1)).reshape(-1)

    print(
        f"Series: {holdout['num_series']} | Train windows: {len(train_X_raw)} | "
        f"context={args.context_length} | horizon={args.horizon}"
    )

    results = []

    persistence_preds = persistence_forecast(holdout["eval_contexts"], args.horizon)
    persistence_metrics = evaluate_forecast(holdout["eval_targets"], persistence_preds)
    results.append(
        collect_result(
            model_name="Persistence",
            metrics=persistence_metrics,
            runtime=0.0,
            num_series=holdout["num_series"],
            context_length=args.context_length,
            horizon=args.horizon,
        )
    )

    results.append(
        train_neural_baseline(
            model_name="Glucose LSTM",
            model_cls=LSTMForecaster,
            train_X_scaled=train_X_scaled,
            train_y_scaled=train_y_scaled,
            eval_contexts_raw=holdout["eval_contexts"],
            eval_targets_raw=holdout["eval_targets"],
            scaler=scaler,
            args=args,
        )
    )
    results.append(
        train_neural_baseline(
            model_name="ST-Attention LSTM",
            model_cls=STAttentionLSTM,
            train_X_scaled=train_X_scaled,
            train_y_scaled=train_y_scaled,
            eval_contexts_raw=holdout["eval_contexts"],
            eval_targets_raw=holdout["eval_targets"],
            scaler=scaler,
            args=args,
        )
    )

    try:
        results.append(run_tabicl_forecaster(holdout["context_df"], holdout["future_truth_df"], args))
    except Exception as exc:
        results.append(
            {
                "Model": "TabICLForecaster",
                "MAE": np.nan,
                "MSE": np.nan,
                "RMSE": np.nan,
                "Time(s)": np.nan,
                "EvalSeries": holdout["num_series"],
                "TrainWindows": 0,
                "ContextLength": args.context_length,
                "Horizon": args.horizon,
                "Protocol": "Patient temporal holdout",
                "Error": str(exc),
            }
        )
        print(f"TabICLForecaster unavailable: {exc}")

    results_df = pd.DataFrame(results).sort_values("MSE", ascending=True, na_position="last")
    print(results_df.to_string(index=False))
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"Saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
