"""
feature_engineering.py

Loads raw per-sample metrics (from data/raw/*.csv), cleans them, and builds
rolling-window features per VM: mean, std, max, and rate-of-change over a
short window (e.g. 30s) and a longer window (e.g. 5min).

These engineered features -- not raw instantaneous values -- are what the
prediction and anomaly models train on. Rolling windows smooth noise and
capture sustained patterns (which distinguish real anomalies from one-off
blips).

Run:
    python feature_engineering.py
"""
import glob
import os

import pandas as pd

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

SHORT_WINDOW = "30s"
LONG_WINDOW = "5min"

BASE_METRICS = [
    "cpu_percent",
    "ram_percent",
    "ram_used_mb",
    "disk_read_bytes",
    "disk_write_bytes",
    "net_bytes_sent",
    "net_bytes_recv",
    "process_count",
]


def load_raw() -> pd.DataFrame:
    files = glob.glob(os.path.join(RAW_DIR, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No raw CSV files found in {RAW_DIR}")
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.sort_values(["vm_id", "timestamp"]).drop_duplicates(["vm_id", "timestamp"])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    all_vm_frames = []

    for vm_id, vm_df in df.groupby("vm_id"):
        vm_df = vm_df.set_index("timestamp").sort_index()
        feats = pd.DataFrame(index=vm_df.index)
        feats["vm_id"] = vm_id

        for metric in BASE_METRICS:
            if metric not in vm_df.columns:
                continue
            feats[f"{metric}_raw"] = vm_df[metric]
            feats[f"{metric}_mean_{SHORT_WINDOW}"] = vm_df[metric].rolling(SHORT_WINDOW).mean()
            feats[f"{metric}_std_{SHORT_WINDOW}"] = vm_df[metric].rolling(SHORT_WINDOW).std()
            feats[f"{metric}_max_{LONG_WINDOW}"] = vm_df[metric].rolling(LONG_WINDOW).max()
            feats[f"{metric}_rate_of_change"] = vm_df[metric].diff()

        all_vm_frames.append(feats)

    out = pd.concat(all_vm_frames).reset_index().rename(columns={"index": "timestamp"})
    out = out.dropna().reset_index(drop=True)
    return out


if __name__ == "__main__":
    raw = load_raw()
    processed = engineer_features(raw)
    out_path = os.path.join(PROCESSED_DIR, "features.csv")
    processed.to_csv(out_path, index=False)
    print(f"Wrote {len(processed)} feature rows to {out_path}")
