# AI-Based Secure Resource Allocation and Anomaly Detection in Virtualized Environments

## A Complete Beginner-to-Implementation Project Guide

---

## 0. Executive Summary

This project builds a system that watches virtual machines (VMs) the way a nurse watches a patient's vital signs. Two things happen at once:

1. A **prediction engine** learns how much CPU and RAM a VM actually needs, so resources aren't wasted or starved.
2. A **detection engine** learns what "normal" VM behavior looks like, so it can flag anything that looks like an attack — a sudden CPU spike, a cryptomining process quietly stealing cycles, one VM hogging everything, or a denial-of-service pattern.

Everything is built with free, beginner-friendly tools: VirtualBox to create the VMs, Python + psutil to read their vital signs, pandas/scikit-learn to learn from the data, and FastAPI + plotly for a simple dashboard.

---

## 1. Research Problem Statement

Modern data centers and cloud platforms run hundreds to millions of VMs on shared physical hardware. Two persistent, related problems show up at that scale:

**Problem A — Static, wasteful resource allocation.** Administrators typically assign a fixed number of CPU cores and a fixed amount of RAM to each VM based on rough guesses or worst-case estimates. This causes either over-provisioning (expensive, wasteful, idle capacity) or under-provisioning (slow VMs, unhappy users). A system that predicts the resources a VM actually needs, based on its historical usage pattern, would let infrastructure scale more efficiently.

**Problem B — Slow or missed detection of malicious resource use.** Attackers who compromise a VM often do one of a few things: install cryptomining software that quietly consumes CPU for profit, launch a denial-of-service attack that floods network/CPU resources, or simply abuse shared resources to degrade other tenants (a "noisy neighbor" attack in multi-tenant clouds). Traditional rule-based monitoring (e.g., "alert if CPU > 90%") is noisy and easy to evade — a data science job legitimately spikes CPU too. What's missing is a system that learns the *shape* of normal behavior per-VM and flags deviations, including subtle, sustained abuse that never crosses a naive threshold.

**Research question:** Can a lightweight, unsupervised/semi-supervised machine learning pipeline, trained purely on locally-collected VM telemetry (CPU, RAM, disk I/O, network I/O, process counts), simultaneously (a) forecast near-optimal resource allocation per VM and (b) detect anomalous or malicious resource-usage patterns with low false-positive rates, without requiring VM-intrusive agents, kernel modification, or labeled attack data at training time?

This framing is deliberately dual-purpose: allocation and security share the same underlying data (resource-usage time series), so one collection pipeline can feed two ML subsystems — which is itself part of the project's novelty (see Section 10).

---

## 2. Virtualization Concepts (Explained Simply)

Skip this section only if you already know what a VM, hypervisor, and host/guest are.

**Physical machine (Host):** Your actual laptop/desktop with real CPU, RAM, and disk.

**Virtual Machine (VM / Guest):** A software-emulated computer that runs *inside* your host machine. It thinks it's a real computer — it has its own virtual CPU, RAM, disk, and OS — but it's actually just a program running on the host, with the host's hypervisor deciding how much of the real hardware it gets to use.

**Hypervisor:** The referee that creates and manages VMs, deciding how physical resources are shared among them. VirtualBox is a "Type 2" hypervisor — it runs as an application on top of your normal OS (Windows/Mac/Linux). This is different from "Type 1" hypervisors (like VMware ESXi or Xen) that run directly on bare hardware, which is what real data centers use — but the concepts you'll learn transfer directly.

**Why virtualization matters for this project:** One physical machine can host many VMs, each isolated from the others. This is exactly how cloud providers (AWS, Azure, GCP) sell computing — you rent a VM, not a physical machine. Because many VMs share one physical machine's resources, allocation and security both become live, ongoing problems: give a VM too little and it stalls; give one VM too much (or let it steal cycles maliciously) and its neighbors suffer.

**Key resource metrics you'll monitor per VM:**

| Metric | What it means | Why it matters here |
|---|---|---|
| CPU usage (%) | How much of the VM's allotted CPU is being used | Core signal for both allocation and spike/mining detection |
| RAM usage (MB / %) | How much memory is occupied | Detects memory pressure, leaks, abuse |
| Disk I/O (read/write bytes/sec) | Data moving to/from virtual disk | DoS and abuse patterns often spike disk too |
| Network I/O (bytes/sec, packet count) | Data moving in/out over the network | Core signal for DoS detection and data exfiltration |
| Process count / top processes | Number and identity of running processes | Helps identify *what* is consuming resources (e.g., a suspicious `xmrig`-like process is a classic cryptominer signature) |
| Context switches / load average | How busy the CPU scheduler is | Secondary signal for abnormal load patterns |

**Your beginner setup:** Install VirtualBox on your host machine, create 2–4 lightweight VMs (e.g., Ubuntu Server, minimal install), and treat each VM as a "patient" you're monitoring. You don't need a real data center — the same code and models apply whether you have 3 VMs or 3,000.

---

## 3. System Architecture

Two ML subsystems share one collection pipeline:

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                         HOST MACHINE                              │
 │                                                                     │
 │   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐  │
 │   │  VM 1     │   │  VM 2     │   │  VM 3     │   │  VM N     │  │
 │   │ (Ubuntu)  │   │ (Ubuntu)  │   │ (Ubuntu)  │   │  ...      │  │
 │   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘  │
 │         │  VBoxManage / psutil / SSH agent               │         │
 │         └────────────────┬─────────────────────────────┘         │
 │                           ▼                                        │
 │                 ┌───────────────────┐                             │
 │                 │  Metrics Collector │  (Python, polls every N sec)│
 │                 └─────────┬─────────┘                             │
 │                           ▼                                        │
 │                 ┌───────────────────┐                             │
 │                 │  Data Pipeline     │  (clean, window, engineer   │
 │                 │  (pandas)          │   features, store to CSV/DB)│
 │                 └─────────┬─────────┘                             │
 │                           ▼                                        │
 │            ┌──────────────┴───────────────┐                       │
 │            ▼                               ▼                       │
 │  ┌─────────────────────┐        ┌─────────────────────────┐       │
 │  │ Resource Prediction  │        │  Anomaly Detection        │       │
 │  │ (Linear Regression,  │        │  (Isolation Forest,        │       │
 │  │  Random Forest,      │        │   One-Class SVM)           │       │
 │  │  XGBoost)             │        │                             │       │
 │  └──────────┬───────────┘        └───────────┬─────────────┘       │
 │             ▼                                 ▼                     │
 │   Recommended CPU/RAM                 Anomaly score + label         │
 │   allocation per VM                   (spike / mining / abuse / DoS)│
 │             │                                 │                     │
 │             └────────────────┬────────────────┘                     │
 │                               ▼                                     │
 │                    ┌────────────────────┐                          │
 │                    │  FastAPI Backend    │                          │
 │                    └─────────┬──────────┘                          │
 │                               ▼                                     │
 │                    ┌────────────────────┐                          │
 │                    │ Dashboard (plotly)  │  Live charts + alerts    │
 │                    └────────────────────┘                          │
 └─────────────────────────────────────────────────────────────────┘
```

**Layer-by-layer explanation:**

1. **VMs (data sources):** Real or simulated VirtualBox VMs, each running a mix of normal workloads and (in test scenarios) injected malicious/heavy workloads.
2. **Metrics Collector:** A Python script using `psutil` (run inside each VM or via `VBoxManage metrics` from the host) that samples CPU%, RAM%, disk I/O, network I/O, and process lists every few seconds.
3. **Data Pipeline:** Cleans raw samples, aggregates into time windows (e.g., 30-second rolling averages), engineers features (rolling mean, rolling std, rate of change), and stores to CSV or SQLite.
4. **Resource Prediction Engine:** Regression models that take historical usage windows and predict the near-future optimal CPU core count and RAM allocation.
5. **Anomaly Detection Engine:** Unsupervised models that score how "unusual" the current window is relative to that VM's learned normal behavior, then a rule/classifier layer labels the anomaly type.
6. **FastAPI Backend:** Serves both model outputs over REST endpoints (e.g., `/predict`, `/anomaly-status`) and triggers alerts.
7. **Dashboard:** Live plotly/matplotlib charts showing per-VM usage, predicted allocation, and anomaly flags in one view.

---

## 4. Dataset Generation Methodology

You need two kinds of data: **normal behavior** and **abnormal/malicious behavior**. Since you likely don't have access to a real attacked data center (good — you don't want that), you'll generate both synthetically and semi-synthetically.

**Step 1 — Baseline (normal) data collection.**
Run your VMs under realistic mixed workloads: idle time, light web browsing/serving, file compression, a small web server under light load, database queries, compilation jobs. Run each scenario for 30–60 minutes while your collector samples every 5–10 seconds. This becomes your "normal" class.

**Step 2 — Controlled anomaly injection (semi-synthetic, the recommended approach).**
Deliberately run these workloads inside test VMs (never on production systems, and always inside your own isolated VirtualBox VMs) and label the time windows:

- *CPU spike*: run a CPU-bound stress tool (e.g., `stress-ng --cpu 4 --timeout 120s`) for short bursts.
- *Cryptomining simulation*: run a benign CPU/GPU-intensive hashing loop (e.g., a simple SHA-256 loop in Python or a sandboxed open-source miner like `xmrig` pointed at a test-only pool, run in an isolated offline VM — never against a live pool with real payout) to reproduce mining's signature: sustained near-100% CPU, low network use, and one process dominating.
- *Resource abuse*: spawn many processes competing for RAM (`stress-ng --vm 4 --vm-bytes 1G`) to simulate a "noisy neighbor."
- *Denial-of-Service pattern*: use a controlled traffic generator (e.g., `hping3` or `iperf3`, again only against your own isolated VM, never external targets) to simulate a flood of network requests/packets.

Label each sample with its ground-truth class (`normal`, `cpu_spike`, `cryptomining`, `resource_abuse`, `dos`). This labeled data is used to *evaluate* your unsupervised anomaly detectors (precision/recall) even though the detectors themselves train only on normal data — the standard way to validate anomaly detection.

**Step 3 — Feature engineering.**
From raw per-second/per-5-second metrics, compute rolling-window features (e.g., 30-second and 5-minute windows): mean, standard deviation, max, rate-of-change (first derivative) for CPU%, RAM%, disk I/O, network I/O; plus process-count and "top process CPU share." These engineered features, not raw instantaneous values, are what your models actually train on — this smooths noise and captures *sustained* patterns, which is what distinguishes a real anomaly from a one-second blip.

**Step 4 — Public datasets as supplements (optional, strengthens publication credibility).**
Consider supplementing your own collected data with public cloud-workload traces such as the Google Cluster Workload Traces, Alibaba Cluster Trace, or Bitbrains VM traces — these give you large-scale realistic "normal" resource-usage patterns to pretrain or validate your prediction models against, even though they won't contain labeled attacks.

**Ethical/safety note:** All stress-testing, mining-simulation, and DoS-simulation tools must be run *only* inside your own isolated VirtualBox VMs, on a host you own, with network traffic contained to a host-only or NAT network that cannot reach the public internet or other people's systems. Never point mining software at a real pool with your own wallet, and never direct traffic tools at any system you don't own.

---

## 5. Collecting VM Metrics with Python

`psutil` is a cross-platform Python library that reads CPU, memory, disk, network, and process stats from the OS. Two collection strategies, depending on your setup:

**Strategy A — Agent inside each VM (simplest, most accurate).**
Install Python + psutil inside each VM's guest OS, run a small script that samples metrics and either writes them to a shared folder, sends them over the network (e.g., a lightweight HTTP POST to your host's FastAPI collector endpoint), or writes to a file synced via VirtualBox Shared Folders.

```python
# collector_agent.py -- runs INSIDE each VM
import psutil
import time
import json
import requests  # to send data to the host collector API

VM_ID = "vm1"
HOST_COLLECTOR_URL = "http://10.0.2.2:8000/ingest"  # 10.0.2.2 = default VirtualBox NAT gateway to host

def sample_metrics():
    cpu_percent = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    disk_io = psutil.disk_io_counters()
    net_io = psutil.net_io_counters()
    procs = sorted(
        psutil.process_iter(["name", "cpu_percent"]),
        key=lambda p: p.info["cpu_percent"] or 0,
        reverse=True,
    )[:5]

    return {
        "vm_id": VM_ID,
        "timestamp": time.time(),
        "cpu_percent": cpu_percent,
        "ram_percent": mem.percent,
        "ram_used_mb": mem.used / (1024 * 1024),
        "disk_read_bytes": disk_io.read_bytes,
        "disk_write_bytes": disk_io.write_bytes,
        "net_bytes_sent": net_io.bytes_sent,
        "net_bytes_recv": net_io.bytes_recv,
        "top_processes": [p.info["name"] for p in procs],
    }

if __name__ == "__main__":
    while True:
        data = sample_metrics()
        try:
            requests.post(HOST_COLLECTOR_URL, json=data, timeout=2)
        except requests.exceptions.RequestException:
            pass  # host temporarily unreachable; skip this sample
        time.sleep(5)
```

**Strategy B — Host-side via VBoxManage (no agent needed inside VM, less granular).**
VirtualBox exposes per-VM CPU/RAM/network/disk metrics from the host using the `VBoxManage metrics` command, without installing anything inside the guest.

```python
# host_vboxmanage_collector.py -- runs on the HOST
import subprocess
import re
import time

def enable_metrics(vm_name):
    subprocess.run(["VBoxManage", "metrics", "setup", "--period", "1", "--samples", "1", vm_name])

def collect_metrics(vm_name):
    result = subprocess.run(
        ["VBoxManage", "metrics", "query", vm_name],
        capture_output=True, text=True
    )
    return result.stdout  # parse CPU/Load, RAM/Usage, Net/Rate, Disk/Usage lines

if __name__ == "__main__":
    vm_name = "Ubuntu-VM-1"
    enable_metrics(vm_name)
    while True:
        raw = collect_metrics(vm_name)
        print(raw)  # replace with a parser + CSV/DB writer
        time.sleep(5)
```

For a beginner project, **Strategy A is recommended** — psutil's in-guest output is richer (per-process detail, exact top-CPU-consuming process names, which is essential for spotting a cryptominer process by name/behavior) and the code is simpler to reason about.

**On the host, a small FastAPI endpoint receives and stores incoming samples:**

```python
# host_ingest_api.py
from fastapi import FastAPI
import csv, os, time

app = FastAPI()
CSV_PATH = "data/raw_metrics.csv"

@app.post("/ingest")
async def ingest(sample: dict):
    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sample.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(sample)
    return {"status": "ok"}
```

---

## 6. Which ML Algorithms Fit Which Job (and Why)

Two different problems need two different families of algorithms: **regression** for resource prediction (you're predicting a number — how many cores, how much RAM), and **unsupervised outlier detection** for anomalies (you don't have reliable labeled attack data at training time, so you learn "normal" and flag deviations).

### 6.1 Resource Allocation Prediction (Regression)

| Algorithm | Why it's suitable | Limitations |
|---|---|---|
| **Linear Regression** | Simplest possible baseline. Fast, interpretable (you can literally read off "for every 1% CPU increase, RAM need increases by X MB"). Good for establishing whether a workload's resource need is roughly linear over time. Always start here — if a complex model doesn't beat it by much, that tells you something. | Assumes linear relationships; VM workloads are often bursty and non-linear (e.g., a compile job's RAM need doesn't scale smoothly), so accuracy is usually mediocre alone. |
| **Random Forest Regressor** | Handles non-linear relationships and interactions between features (e.g., "high disk I/O + high process count" together predicting RAM need) without you hand-engineering interaction terms. Robust to noisy/outlier-heavy VM data. Gives feature importance, useful for your report ("CPU rate-of-change was the strongest predictor"). | Larger model, slightly slower to train/predict than linear regression; can overfit on small datasets if not tuned (limit tree depth, use enough trees). |
| **XGBoost (Gradient Boosted Trees)** | Typically the strongest tabular-data regressor available; sequentially corrects errors of previous trees, usually beats Random Forest on accuracy with proper tuning. Handles missing values natively (useful since network samples occasionally drop). Widely used in real cloud-resource-prediction research, which helps your novelty/publication framing. | More hyperparameters to tune (learning rate, depth, number of estimators); easier to overfit if not validated carefully; less interpretable than linear regression. |

**Recommended approach:** Train all three, compare with MAE/RMSE/R² on a held-out time period (not a random split — time series data needs chronological splits to avoid leaking future information into training). Report all three in your results section; this comparison *is* a legitimate research contribution on its own.

### 6.2 Anomaly / Malicious Behavior Detection (Unsupervised Outlier Detection)

| Algorithm | Why it's suitable | Limitations |
|---|---|---|
| **Isolation Forest** | Purpose-built for anomaly detection: it isolates points by random recursive splitting, and anomalies (being rare and different) get isolated in fewer splits — no need for labeled attack data at training time. Scales well, fast on tabular data, handles multi-dimensional feature spaces (CPU + RAM + disk + net together) naturally. This is your **primary/recommended anomaly detector**. | Can struggle with anomalies that only differ subtly in one dimension when many other dimensions are normal (\"local\" anomalies); contamination parameter (expected anomaly rate) needs rough tuning. |
| **One-Class SVM** | Learns a boundary around normal data in feature space; anything outside the boundary is flagged. Good for comparison/ensemble — it captures a different notion of \"outlier\" (density/boundary-based rather than isolation-based), so combining both often improves overall precision/recall. | Slower on larger datasets (doesn't scale as well as Isolation Forest); sensitive to feature scaling (must standardize features first) and to the choice of `nu`/kernel hyperparameters. |

**Recommended approach:** Train Isolation Forest as your main detector on normal-only data. Train One-Class SVM in parallel as a comparison/ensemble check. Evaluate both against your labeled semi-synthetic attack data (Section 4, Step 2) using precision, recall, F1, and ROC-AUC — since you *do* have ground-truth labels for evaluation, even though the models never see labels during training. Optionally combine the two (e.g., flag as anomaly if either model flags it, or average their anomaly scores) — this ensemble comparison is another solid novelty/publication angle.

**After detection — classifying the anomaly *type*:** Isolation Forest and One-Class SVM tell you *something is wrong*, not *what*. Add a lightweight second-stage classifier (e.g., a small Random Forest classifier trained on your labeled semi-synthetic data) that takes the flagged window's features (top-process name, CPU/network ratio, sustained-duration, etc.) and predicts which category it best matches: `cpu_spike`, `cryptomining`, `resource_abuse`, or `dos`. This two-stage design (unsupervised detection → supervised classification of the flagged cases only) is realistic, efficient, and a nice architectural talking point for a paper.

---

## 7. Folder Structure

```
vm-ai-monitor/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                     # raw collected samples (per VM, per session)
│   ├── labeled/                 # semi-synthetic labeled anomaly windows
│   └── processed/               # feature-engineered, windowed datasets
│
├── collectors/
│   ├── collector_agent.py       # runs INSIDE each VM (psutil + POST to host)
│   ├── host_vboxmanage_collector.py   # optional host-side alternative
│   └── host_ingest_api.py       # FastAPI endpoint that receives + stores samples
│
├── data_pipeline/
│   ├── clean.py                 # de-dupe, handle missing/NaN, fix timestamps
│   ├── feature_engineering.py   # rolling mean/std/rate-of-change windows
│   └── dataset_builder.py       # merges normal + labeled anomaly data into train/test sets
│
├── models/
│   ├── prediction/
│   │   ├── train_linear_regression.py
│   │   ├── train_random_forest.py
│   │   ├── train_xgboost.py
│   │   └── evaluate_prediction_models.py
│   ├── anomaly/
│   │   ├── train_isolation_forest.py
│   │   ├── train_one_class_svm.py
│   │   ├── anomaly_type_classifier.py     # 2nd-stage: labels the anomaly type
│   │   └── evaluate_anomaly_models.py
│   └── saved/                   # serialized .pkl / .joblib model files
│
├── simulation/
│   ├── workload_normal.sh       # scripts to generate normal VM load
│   ├── attack_cpu_spike.sh
│   ├── attack_cryptomining.py   # sandboxed hashing-loop simulator (test-only)
│   ├── attack_resource_abuse.sh
│   └── attack_dos.sh
│
├── api/
│   ├── main.py                  # FastAPI app: /predict, /anomaly-status, /vm/{id}/history
│   └── schemas.py
│
├── dashboard/
│   ├── app.py                   # plotly/dash or streamlit live dashboard
│   └── charts.py
│
├── notebooks/
│   ├── 01_eda.ipynb              # exploratory data analysis
│   ├── 02_prediction_experiments.ipynb
│   └── 03_anomaly_experiments.ipynb
│
└── docs/
    ├── architecture_diagram.png
    ├── flowcharts/
    └── report.docx               # your final written report
```

---

## 8. Step-by-Step Implementation Roadmap

**Phase 1 — Environment setup.**
Install VirtualBox, create 2–4 Ubuntu Server VMs (1–2 GB RAM, 1–2 vCPUs each is enough for a mini project). Install Python 3.10+, psutil, pandas, scikit-learn, xgboost, fastapi, uvicorn, plotly inside your dev environment and (for Strategy A) inside each VM.

**Phase 2 — Build the collector.**
Write and test `collector_agent.py` inside one VM first. Confirm samples arrive at your host's `host_ingest_api.py` and land correctly in `data/raw/`. Then replicate to all VMs, each tagged with its own `vm_id`.

**Phase 3 — Collect normal baseline data.**
Run mixed normal workloads across all VMs for several hours (spread across a few sessions/days for variety) while the collector runs continuously. Aim for at least 5,000–10,000 samples per VM as a beginner-scale target.

**Phase 4 — Generate labeled anomaly data.**
Run each attack-simulation script in `simulation/` for controlled bursts (2–10 minutes each), repeated several times, always noting exact start/end timestamps so you can label those windows precisely in `data/labeled/`.

**Phase 5 — Data pipeline.**
Build `clean.py` and `feature_engineering.py`. Produce a clean, windowed, feature-rich dataset in `data/processed/`. Do exploratory data analysis in `01_eda.ipynb` — plot CPU/RAM/network over time, sanity-check that your injected anomalies are visually distinguishable.

**Phase 6 — Train resource prediction models.**
Chronological train/test split. Train Linear Regression, Random Forest, XGBoost. Compare MAE/RMSE/R². Pick a "recommended allocation" formula (e.g., predicted peak + safety margin) and document your reasoning.

**Phase 7 — Train anomaly detection models.**
Train Isolation Forest and One-Class SVM on normal-only windows. Evaluate against your labeled anomaly data (precision/recall/F1/ROC-AUC). Tune the contamination/nu parameters. Build the second-stage anomaly-type classifier.

**Phase 8 — Build the API.**
Wire up FastAPI endpoints: `/predict` (returns recommended CPU/RAM for a VM), `/anomaly-status` (returns current anomaly score + type per VM), `/vm/{id}/history` (returns recent time series for charting).

**Phase 9 — Build the dashboard.**
A simple Streamlit or Plotly Dash page showing live per-VM charts, current recommended allocation, and a red/yellow/green anomaly indicator with the detected type when triggered.

**Phase 10 — Evaluation, write-up, polish.**
Run end-to-end tests (inject a fresh, previously-unseen attack run and confirm detection), compute final metrics tables, write the report/paper, prepare diagrams, and record a short demo.

---

## 9. Week-by-Week Plan (10 Weeks, Beginner Pace)

| Week | Focus | Deliverable |
|---|---|---|
| 1 | Learn virtualization basics; install VirtualBox; create and configure 2–4 VMs; install Python/psutil in each | Working VMs reachable from host |
| 2 | Build `collector_agent.py` + `host_ingest_api.py`; verify end-to-end metric flow | Live metrics streaming from VMs into a CSV/DB |
| 3 | Collect normal baseline data across varied workloads (several sessions) | 5,000–10,000+ labeled-as-normal samples per VM |
| 4 | Build and run attack-simulation scripts (CPU spike, mining sim, resource abuse, DoS sim); collect labeled anomaly data | Labeled anomaly dataset with precise timestamps |
| 5 | Data cleaning + feature engineering pipeline; exploratory data analysis notebook | Clean, windowed, feature-rich processed dataset |
| 6 | Train and compare Linear Regression, Random Forest, XGBoost for resource prediction | Prediction model comparison table + chosen model |
| 7 | Train and compare Isolation Forest, One-Class SVM for anomaly detection; build 2nd-stage anomaly-type classifier | Anomaly detection metrics (precision/recall/F1/ROC-AUC) |
| 8 | Build FastAPI backend exposing prediction + anomaly endpoints | Working REST API with test requests |
| 9 | Build live dashboard (Streamlit/Plotly Dash); wire it to the API; end-to-end fresh-attack test | Working live dashboard demo |
| 10 | Final evaluation, diagrams, documentation, report/paper write-up, demo recording | Final report + repo + demo video |

*(If you have less time, Weeks 3–4 and 6–7 compress most easily — reduce data-collection duration and try only Random Forest + Isolation Forest as your primary pair.)*

---

## 10. Novelty Points for Publication

A student mini-project becomes a publishable/conference-worthy contribution when it goes beyond "I ran scikit-learn on some data." Genuine angles here:

1. **Unified dual-purpose pipeline.** Most existing work treats VM resource prediction and VM anomaly/intrusion detection as separate research areas with separate pipelines. Building one shared telemetry-collection and feature-engineering pipeline that feeds *both* a regression subsystem and an anomaly-detection subsystem is an architectural contribution — it reduces monitoring overhead and shows the same signals serve two purposes.

2. **Two-stage anomaly detection + classification.** Rather than stopping at a binary "anomaly / not anomaly" score (which is what most Isolation Forest / One-Class SVM papers do), adding a second-stage classifier that names the anomaly *type* (CPU spike vs. cryptomining vs. resource abuse vs. DoS) makes the output directly actionable for an administrator, closer to a real security operations use case.

3. **Ensemble comparison of isolation-based vs. boundary-based unsupervised detectors.** Systematically comparing Isolation Forest (isolation-based) against One-Class SVM (density/boundary-based) on the *same* VM telemetry, and testing whether combining their scores improves precision/recall over either alone, is a concrete, reproducible experimental contribution.

4. **Process-level attribution for cryptomining detection.** Most CPU-spike anomaly detectors only look at aggregate CPU%. Incorporating top-process-name and per-process CPU share as features lets the system distinguish "a legitimate batch job spiked CPU" from "an unrecognized process is monopolizing CPU with near-zero I/O" — the latter being cryptomining's real signature. This process-attribution angle is under-explored in lightweight, agent-based (non-kernel-level) detection literature.

5. **Resource-allocation recommendations validated against real security constraints.** Most VM resource-prediction papers optimize purely for cost/performance. Framing the prediction model's safety margin explicitly around anomaly-detection sensitivity (i.e., "how much headroom do we allocate so a real spike doesn't get misread as an attack, and vice versa") ties the two subsystems together conceptually, not just architecturally.

6. **Reproducible, VirtualBox-based, non-invasive collection.** A lot of published anomaly-detection work relies on kernel-level hooks, hypervisor-level introspection, or proprietary cloud telemetry not available to independent researchers. Showing a fully reproducible pipeline built entirely on freely available tools (VirtualBox + psutil) that still achieves competitive detection accuracy is valuable for reproducibility-focused venues and for practitioners without access to enterprise tooling.

*(Grounding note: recent literature — e.g., Isolation Forest/One-Class SVM approaches to VM workload anomaly detection, and cryptojacking-specific ML detection studies published in 2024–2025 — confirms these are active, still-open research directions rather than solved problems, which supports a solid related-work section.)*

---

## 11. Future Scope

- **Deep learning extensions:** LSTM/GRU or Transformer-based time-series models for resource prediction once you have larger longitudinal datasets; autoencoders or graph neural networks for anomaly detection across VM-to-VM interaction patterns.
- **Real cloud deployment:** Port the collector to real cloud VMs (AWS EC2, Azure VM, GCP Compute Engine) using their native monitoring APIs (CloudWatch, Azure Monitor, Stackdriver) instead of VirtualBox-specific tooling.
- **Container/Kubernetes extension:** Extend the same architecture to Docker containers and Kubernetes pods, where resource contention and cryptojacking are equally (if not more) common in production.
- **Online/incremental learning:** Replace static batch-trained models with online learning so the system adapts as a VM's "normal" behavior legitimately shifts over time (e.g., seasonal traffic patterns), instead of needing periodic retraining.
- **Federated/multi-tenant learning:** In a multi-tenant cloud, train shared anomaly-detection models across tenants without centralizing raw telemetry (federated learning), preserving privacy while still benefiting from cross-tenant attack patterns.
- **Automated response/self-healing:** Beyond detection, trigger automated mitigation — throttle or quarantine a VM showing cryptomining signatures, auto-scale resources when the prediction model forecasts an upcoming shortfall.
- **Explainability layer:** Add SHAP/LIME explanations to both the prediction and anomaly models so administrators see *why* a VM was flagged or *why* a particular allocation was recommended — important for trust in production security tooling.
- **Adversarial robustness testing:** Evaluate whether a sophisticated attacker could disguise cryptomining or DoS traffic to evade the trained detectors (e.g., throttling CPU usage to stay under the learned "normal" envelope), and study defenses against such evasion.

---

## 12. Flowcharts and System Diagrams

**Figure 1 — System Architecture** is in Section 3 above. Two more diagrams below.

**Figure 2 — End-to-End ML Pipeline Flow**

```
 ┌───────────────┐   ┌────────────────┐   ┌───────────────────┐
 │ Raw metrics    │──▶│ Cleaning +      │──▶│ Feature engineering │
 │ (psutil, 5s)   │   │ timestamp fix   │   │ (rolling windows)   │
 └───────────────┘   └────────────────┘   └──────────┬─────────┘
                                                         │
                              ┌──────────────────────────┴───────────────────────┐
                              ▼                                                    ▼
                 ┌─────────────────────────┐                         ┌─────────────────────────┐
                 │ Chronological train/test │                         │ Normal-only training set │
                 │ split (for regression)   │                         │ (for anomaly detectors)  │
                 └────────────┬─────────────┘                         └────────────┬─────────────┘
                              ▼                                                    ▼
        ┌─────────────────────────────────────┐              ┌─────────────────────────────────────┐
        │ Train: Linear Regression,             │              │ Train: Isolation Forest,              │
        │ Random Forest, XGBoost                │              │ One-Class SVM                          │
        └────────────────────┬───────────────────┘              └────────────────────┬───────────────────┘
                              ▼                                                    ▼
              Evaluate: MAE / RMSE / R²                        Evaluate vs. labeled anomalies:
                              │                                  Precision / Recall / F1 / ROC-AUC
                              ▼                                                    ▼
                    Save best model (.pkl)                          Save detector + 2nd-stage
                              │                                  anomaly-type classifier (.pkl)
                              └──────────────────┬─────────────────────┘
                                                  ▼
                                    Serve both via FastAPI (/predict, /anomaly-status)
                                                  ▼
                                        Live dashboard (plotly / Streamlit)
```

**Figure 3 — Anomaly Detection Decision Flow**

```
        ┌───────────────────────┐
        │ New metrics window     │
        │ (CPU, RAM, disk, net,  │
        │  top process, 30s agg) │
        └───────────┬────────────┘
                     ▼
        ┌───────────────────────┐
        │ Isolation Forest +     │
        │ One-Class SVM score    │
        └───────────┬────────────┘
                     ▼
              ┌─────────────┐
              │ Anomaly       │   No  ┌─────────────────────┐
              │ score beyond   ├──────▶│ Mark window "normal" │
              │ threshold?     │       │ no alert              │
              └──────┬────────┘       └─────────────────────┘
                     │ Yes
                     ▼
        ┌───────────────────────────┐
        │ 2nd-stage classifier:       │
        │ label anomaly type using    │
        │ top-process, CPU/net ratio, │
        │ duration, RAM pressure      │
        └───────────┬─────────────────┘
                     ▼
     ┌───────────────┼────────────────┬──────────────────┐
     ▼               ▼                ▼                  ▼
 CPU spike     Cryptomining      Resource abuse         DoS pattern
 (short burst, (sustained ~100%  (RAM/process flood,    (network flood,
  known proc)   CPU, unknown      one VM starving        packet/connection
                proc, low net)    neighbors)              spike)
     │               │                │                  │
     └───────────────┴────────┬───────┴──────────────────┘
                               ▼
                  Raise alert on dashboard + log
                  (VM id, type, confidence, timestamp)
```

**Figure 4 — 10-Week Timeline (Gantt-style)**

```
Week:            1  2  3  4  5  6  7  8  9  10
VM setup         ██
Collector build     ██
Normal data covg       ██
Attack simulation         ██
Data pipeline                ██
Prediction models                ██
Anomaly models                       ██
API build                                ██
Dashboard                                    ██
Eval + report                                    ██
```

---

## Appendix A — requirements.txt (starter)

```
psutil==6.0.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.0
xgboost==2.1.0
fastapi==0.111.0
uvicorn==0.30.1
plotly==5.22.0
streamlit==1.36.0
requests==2.32.3
joblib==1.4.2
matplotlib==3.9.0
```

## Appendix B — Evaluation Metrics Reference

- **Resource prediction (regression):** MAE, RMSE, R², MAPE — report on a chronological (not random) held-out test window.
- **Anomaly detection (classification-style evaluation against labeled data):** Precision, Recall, F1-score, ROC-AUC, confusion matrix per anomaly type.
- **System-level:** Detection latency (seconds from attack start to alert), false-positive rate per hour of normal operation (critical for real-world usability — a noisy detector gets ignored).
