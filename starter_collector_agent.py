"""
collector_agent.py -- runs INSIDE each VM.

Samples CPU / RAM / disk / network / top-process metrics every few seconds
using psutil and POSTs each sample to the host's ingestion API.

Setup inside the guest VM:
    pip install psutil requests
    python collector_agent.py
"""
import time
import psutil
import requests

# --- Configure per VM ---
VM_ID = "vm1"  # change per VM: vm1, vm2, vm3, ...
HOST_COLLECTOR_URL = "http://10.0.2.2:8000/ingest"  # 10.0.2.2 = VirtualBox NAT gateway to host
SAMPLE_INTERVAL_SECONDS = 5


def sample_metrics() -> dict:
    cpu_percent = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    disk_io = psutil.disk_io_counters()
    net_io = psutil.net_io_counters()

    top_processes = sorted(
        psutil.process_iter(["name", "cpu_percent"]),
        key=lambda p: (p.info.get("cpu_percent") or 0),
        reverse=True,
    )[:5]

    return {
        "vm_id": VM_ID,
        "timestamp": time.time(),
        "cpu_percent": cpu_percent,
        "ram_percent": mem.percent,
        "ram_used_mb": mem.used / (1024 * 1024),
        "disk_read_bytes": disk_io.read_bytes if disk_io else 0,
        "disk_write_bytes": disk_io.write_bytes if disk_io else 0,
        "net_bytes_sent": net_io.bytes_sent if net_io else 0,
        "net_bytes_recv": net_io.bytes_recv if net_io else 0,
        "process_count": len(psutil.pids()),
        "top_process_1_name": top_processes[0].info["name"] if top_processes else None,
        "top_process_1_cpu": top_processes[0].info.get("cpu_percent") if top_processes else 0,
    }


def run():
    print(f"[{VM_ID}] collector agent starting, sampling every {SAMPLE_INTERVAL_SECONDS}s")
    while True:
        sample = sample_metrics()
        try:
            resp = requests.post(HOST_COLLECTOR_URL, json=sample, timeout=2)
            if resp.status_code != 200:
                print(f"[{VM_ID}] ingest returned {resp.status_code}")
        except requests.exceptions.RequestException as exc:
            print(f"[{VM_ID}] could not reach host collector: {exc}")
        time.sleep(SAMPLE_INTERVAL_SECONDS)


if __name__ == "__main__":
    run()
