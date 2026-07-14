"""
attack_simulators.py

Controlled workload generators used ONLY to create labeled training/evaluation
data for the anomaly detectors. Run these ONLY inside your own isolated
VirtualBox VMs, on a host-only or NAT network with no route to the public
internet or to any system you don't own.

Each function prints a start/end timestamp -- record these and use them to
label the corresponding rows in your collected metrics as the given anomaly
type when building data/labeled/labeled_features.csv.

Requires `stress-ng` installed in the guest VM for the shell-based helpers
(sudo apt install stress-ng), or run the pure-Python CPU spike / mining
simulator below if you'd rather not install extra tools.
"""
import hashlib
import multiprocessing
import subprocess
import time


def log_window(label: str, seconds: int):
    start = time.time()
    print(f"[{label}] START at {start}")
    yield
    end = time.time()
    print(f"[{label}] END at {end} (duration {end - start:.1f}s)")


def cpu_spike(duration_seconds: int = 60):
    """Short CPU-bound burst -- simulates a sudden legitimate-looking spike."""
    print(f"[cpu_spike] START at {time.time()}")
    subprocess.run(["stress-ng", "--cpu", "4", "--timeout", f"{duration_seconds}s"])
    print(f"[cpu_spike] END at {time.time()}")


def _hash_worker(stop_time: float):
    data = b"vm-ai-monitor-cryptomining-simulation"
    while time.time() < stop_time:
        hashlib.sha256(data).hexdigest()


def cryptomining_simulation(duration_seconds: int = 300, workers: int = 4):
    """
    Pure-Python, sandboxed hashing-loop simulator that reproduces
    cryptomining's real signature: sustained near-100% CPU, low network use,
    one process (this one) dominating CPU share. No real mining pool is
    contacted -- this never earns or spends anything.
    """
    print(f"[cryptomining_simulation] START at {time.time()}")
    stop_time = time.time() + duration_seconds
    procs = [multiprocessing.Process(target=_hash_worker, args=(stop_time,)) for _ in range(workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print(f"[cryptomining_simulation] END at {time.time()}")


def resource_abuse(duration_seconds: int = 120):
    """RAM/process flood -- simulates a 'noisy neighbor' starving other VMs."""
    print(f"[resource_abuse] START at {time.time()}")
    subprocess.run(["stress-ng", "--vm", "4", "--vm-bytes", "1G", "--timeout", f"{duration_seconds}s"])
    print(f"[resource_abuse] END at {time.time()}")


def dos_pattern_simulation(duration_seconds: int = 60, target_host: str = "127.0.0.1", target_port: int = 5001):
    """
    Network flood against a target YOU control (e.g. an iperf3 server running
    on another one of your own isolated VMs). Requires iperf3 installed.
    NEVER point this at a host you do not own.
    """
    print(f"[dos_pattern_simulation] START at {time.time()}")
    subprocess.run([
        "iperf3", "-c", target_host, "-p", str(target_port), "-t", str(duration_seconds), "-b", "500M",
    ])
    print(f"[dos_pattern_simulation] END at {time.time()}")


if __name__ == "__main__":
    print("Run individual functions manually and record their printed start/end timestamps,")
    print("e.g.: python -c \"from attack_simulators import cpu_spike; cpu_spike(60)\"")
