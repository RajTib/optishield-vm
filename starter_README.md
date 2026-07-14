# vm-ai-monitor — Starter Code

Starter code for **AI-Based Secure Resource Allocation and Anomaly Detection in Virtualized Environments**.

Full project guide (concepts, architecture, algorithms, week-by-week plan): see
`AI_VM_Resource_Anomaly_Project_Guide.md`.

These files are flat here for easy download; recreate the nested folder structure described
in Section 7 of the guide when you set up your actual repo:

```
vm-ai-monitor/
├── collectors/collector_agent.py            <- starter_collector_agent.py
├── collectors/host_ingest_api.py            <- starter_host_ingest_api.py
├── data_pipeline/feature_engineering.py     <- starter_feature_engineering.py
├── models/prediction/train_prediction_models.py  <- starter_train_prediction_models.py
├── models/anomaly/train_anomaly_models.py   <- starter_train_anomaly_models.py
├── simulation/attack_simulators.py          <- starter_attack_simulators.py
├── api/main.py                              <- starter_api_main.py
└── requirements.txt                         <- starter_requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt

# 1. On the host: start the metrics ingestion API
uvicorn collectors.host_ingest_api:app --host 0.0.0.0 --port 8000

# 2. Inside each VM: run the collector agent (edit VM_ID and HOST_COLLECTOR_URL first)
python collectors/collector_agent.py

# 3. After collecting enough data, engineer features
python data_pipeline/feature_engineering.py

# 4. Train resource-prediction models
python models/prediction/train_prediction_models.py

# 5. Train anomaly-detection models
python models/anomaly/train_anomaly_models.py

# 6. Serve predictions + anomaly status
uvicorn api.main:app --host 0.0.0.0 --port 9000
```

## Safety note

Everything in `attack_simulators.py` is meant to run **only inside your own isolated
VirtualBox VMs**, on a host-only or NAT network with no route to the public internet or to
systems you don't own. Do not point any of these scripts at real infrastructure.
