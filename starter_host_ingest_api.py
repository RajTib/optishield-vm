"""
host_ingest_api.py -- runs on the HOST.

Minimal FastAPI endpoint that receives metric samples POSTed by each VM's
collector_agent.py and appends them to a per-day CSV file under data/raw/.

Run:
    uvicorn host_ingest_api:app --host 0.0.0.0 --port 8000
"""
import csv
import os
from datetime import datetime, timezone

from fastapi import FastAPI

app = FastAPI(title="VM Metrics Ingestion API")

DATA_DIR = "data/raw"
os.makedirs(DATA_DIR, exist_ok=True)


def _csv_path_for_today() -> str:
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return os.path.join(DATA_DIR, f"raw_metrics_{day}.csv")


@app.post("/ingest")
async def ingest(sample: dict):
    path = _csv_path_for_today()
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(sample.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(sample)
    return {"status": "ok"}


@app.get("/health")
async def health():
    return {"status": "healthy"}
