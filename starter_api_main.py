"""
api/main.py

FastAPI app that serves both ML subsystems:
  - GET /predict/{vm_id}          -> recommended CPU/RAM allocation
  - GET /anomaly-status/{vm_id}   -> current anomaly score + type
  - GET /vm/{vm_id}/history       -> recent time series for charting

This is a thin serving layer over the models trained by
train_prediction_models.py and train_anomaly_models.py. Wire the TODOs
below to your actual feature-computation and model-loading code.

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 9000
"""
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

app = FastAPI(title="VM AI Resource + Anomaly API")

MODEL_DIR = "models/saved"
FEATURES_PATH = "data/processed/features.csv"


def _load_model(name: str):
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def _latest_features_for_vm(vm_id: str) -> pd.DataFrame:
    df = pd.read_csv(FEATURES_PATH, parse_dates=["timestamp"])
    vm_df = df[df["vm_id"] == vm_id].sort_values("timestamp")
    if vm_df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for vm_id={vm_id}")
    return vm_df.tail(1)


@app.get("/predict/{vm_id}")
async def predict(vm_id: str):
    cpu_model = _load_model("prediction_target_cpu_percent_raw.pkl")
    ram_model = _load_model("prediction_target_ram_used_mb_raw.pkl")
    if cpu_model is None or ram_model is None:
        raise HTTPException(status_code=503, detail="Prediction models not trained yet")

    latest = _latest_features_for_vm(vm_id)
    feature_cols = [c for c in latest.columns if c not in ("timestamp", "vm_id")]
    X = latest[feature_cols]

    predicted_cpu = float(cpu_model.predict(X)[0])
    predicted_ram_mb = float(ram_model.predict(X)[0])

    safety_margin = 1.2  # 20% headroom
    return {
        "vm_id": vm_id,
        "recommended_cpu_percent": round(predicted_cpu * safety_margin, 1),
        "recommended_ram_mb": round(predicted_ram_mb * safety_margin, 1),
    }


@app.get("/anomaly-status/{vm_id}")
async def anomaly_status(vm_id: str):
    scaler = _load_model("anomaly_scaler.pkl")
    iso_forest = _load_model("isolation_forest.pkl")
    type_clf = _load_model("anomaly_type_classifier.pkl")
    if scaler is None or iso_forest is None:
        raise HTTPException(status_code=503, detail="Anomaly models not trained yet")

    latest = _latest_features_for_vm(vm_id)
    feature_cols = [c for c in latest.columns if c not in ("timestamp", "vm_id")]
    X = scaler.transform(latest[feature_cols])

    is_anomaly = bool(iso_forest.predict(X)[0] == -1)
    anomaly_type = None
    if is_anomaly and type_clf is not None:
        anomaly_type = str(type_clf.predict(latest[feature_cols])[0])

    return {
        "vm_id": vm_id,
        "is_anomaly": is_anomaly,
        "anomaly_type": anomaly_type,
        "timestamp": str(latest["timestamp"].iloc[0]),
    }


@app.get("/vm/{vm_id}/history")
async def vm_history(vm_id: str, limit: int = 200):
    df = pd.read_csv(FEATURES_PATH, parse_dates=["timestamp"])
    vm_df = df[df["vm_id"] == vm_id].sort_values("timestamp").tail(limit)
    return vm_df.to_dict(orient="records")


@app.get("/health")
async def health():
    return {"status": "healthy"}
