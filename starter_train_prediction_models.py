"""
train_prediction_models.py

Trains and compares three regressors for predicting near-future optimal
CPU and RAM allocation per VM: Linear Regression, Random Forest, and
XGBoost. Uses a CHRONOLOGICAL train/test split (never a random split --
time series data leaks future information into training under a random
split).

Target: predict cpu_percent_raw and ram_used_mb_raw N steps ahead, using the
engineered features from feature_engineering.py as inputs.

Run:
    python train_prediction_models.py
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

FEATURES_PATH = "data/processed/features.csv"
MODEL_DIR = "models/saved"
os.makedirs(MODEL_DIR, exist_ok=True)

PREDICTION_HORIZON_STEPS = 6  # e.g. 6 steps ahead at 30s cadence ~= 3 minutes ahead
TARGETS = ["cpu_percent_raw", "ram_used_mb_raw"]


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_PATH, parse_dates=["timestamp"])
    return df.sort_values(["vm_id", "timestamp"])


def build_supervised(df: pd.DataFrame) -> pd.DataFrame:
    """Shift targets forward per VM to create a forecasting supervised-learning setup."""
    frames = []
    for vm_id, vm_df in df.groupby("vm_id"):
        vm_df = vm_df.copy()
        for target in TARGETS:
            vm_df[f"target_{target}"] = vm_df[target].shift(-PREDICTION_HORIZON_STEPS)
        frames.append(vm_df)
    out = pd.concat(frames).dropna().reset_index(drop=True)
    return out


def chronological_split(df: pd.DataFrame, test_fraction: float = 0.2):
    split_idx = int(len(df) * (1 - test_fraction))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"  {name:20s}  MAE={mae:8.3f}  RMSE={rmse:8.3f}  R2={r2:6.3f}")
    return {"model": name, "mae": mae, "rmse": rmse, "r2": r2}


def train_and_compare(df: pd.DataFrame, target_col: str, feature_cols: list):
    train_df, test_df = chronological_split(df)
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    print(f"\n=== Target: {target_col} ===")
    results = []

    lin = LinearRegression().fit(X_train, y_train)
    results.append(evaluate("LinearRegression", y_test, lin.predict(X_test)))

    rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results.append(evaluate("RandomForest", y_test, rf.predict(X_test)))

    xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
    xgb.fit(X_train, y_train)
    results.append(evaluate("XGBoost", y_test, xgb.predict(X_test)))

    best = min(results, key=lambda r: r["rmse"])
    best_model = {"LinearRegression": lin, "RandomForest": rf, "XGBoost": xgb}[best["model"]]
    joblib.dump(best_model, os.path.join(MODEL_DIR, f"prediction_{target_col}.pkl"))
    print(f"  -> best model: {best['model']} (saved)")


if __name__ == "__main__":
    df = load_dataset()
    df = build_supervised(df)

    exclude_cols = {"timestamp", "vm_id"} | {f"target_{t}" for t in TARGETS}
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    for target in TARGETS:
        train_and_compare(df, f"target_{target}", feature_cols)
