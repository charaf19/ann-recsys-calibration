"""Energy measurement primitives (library).

Reviewer concern addressed: "no direct energy/power measurement". This
module measures CPU package energy via Intel RAPL when the platform exposes
it (Linux powercap sysfs), optionally GPU energy via NVIDIA NVML
(supplementary only — the benchmark itself is CPU-only), and NEVER fabricates
energy numbers: on platforms without a direct counter (e.g. Windows), it
reports timing + CPU utilization with direct_energy_available=false and NA
energy fields.

See docs/energy_measurement_protocol.md for the measurement protocol.
"""
import os
import time
from pathlib import Path

import psutil

RAPL_ROOT = Path("/sys/class/powercap")
NA = "NA"
FALLBACK_NOTE = ("Direct package-energy measurement unavailable on this "
                 "platform; reporting timing and utilization only.")


# ----------------------------
# Intel RAPL (Linux powercap)
# ----------------------------

def rapl_domains():
    """List readable RAPL package domains: [(name, energy_uj_path, max_uj)]."""
    domains = []
    if not RAPL_ROOT.is_dir():
        return domains
    for d in sorted(RAPL_ROOT.glob("intel-rapl:*")):
        efile = d / "energy_uj"
        if not efile.is_file() or not os.access(efile, os.R_OK):
            continue
        try:
            name = (d / "name").read_text().strip()
            max_uj = int((d / "max_energy_range_uj").read_text().strip())
            int(efile.read_text().strip())  # readability probe
            domains.append((name, efile, max_uj))
        except (OSError, ValueError, PermissionError):
            continue
    return domains


def read_rapl_uj(domains):
    return {name: int(efile.read_text().strip()) for name, efile, _ in domains}


def rapl_delta_joules(before, after, domains):
    """Sum energy deltas across domains, handling counter wraparound."""
    total_uj = 0
    max_by_name = {name: max_uj for name, _, max_uj in domains}
    for name, b in before.items():
        a = after.get(name, b)
        d = a - b
        if d < 0:  # wrapped
            d += max_by_name.get(name, 0)
        total_uj += max(0, d)
    return total_uj / 1e6


def rapl_available():
    return len(rapl_domains()) > 0


# ----------------------------
# NVIDIA NVML (supplementary only)
# ----------------------------

def nvml_handles():
    """Return (pynvml, [handles]) or (None, []) when NVML is unavailable."""
    try:
        import pynvml
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        return pynvml, [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]
    except Exception:
        return None, []


def nvml_total_energy_mj(pynvml, handles):
    """Total GPU energy counters in millijoules (None if unsupported)."""
    if pynvml is None or not handles:
        return None
    try:
        return sum(pynvml.nvmlDeviceGetTotalEnergyConsumption(h) for h in handles)
    except Exception:
        return None


# ----------------------------
# Measurement wrapper
# ----------------------------

def measure(workload_fn, sample_interval_sec=0.2):
    """Run workload_fn() while measuring energy (if available), wall time,
    and CPU utilization. Returns a dict with honest NA fields when direct
    energy is not measurable.
    """
    domains = rapl_domains()
    pynvml, handles = nvml_handles()

    proc = psutil.Process(os.getpid())
    psutil.cpu_percent(interval=None)  # prime system-wide counter

    rapl_before = read_rapl_uj(domains) if domains else None
    gpu_before = nvml_total_energy_mj(pynvml, handles)
    t0 = time.perf_counter()

    result = workload_fn()

    wall = time.perf_counter() - t0
    rapl_after = read_rapl_uj(domains) if domains else None
    gpu_after = nvml_total_energy_mj(pynvml, handles)
    cpu_util = psutil.cpu_percent(interval=None)
    rss = proc.memory_info().rss / (1024 * 1024)

    if domains:
        cpu_j = rapl_delta_joules(rapl_before, rapl_after, domains)
        backend = "intel_rapl"
        direct = True
        notes = f"RAPL domains: {sorted(rapl_before)}"
    else:
        cpu_j = NA
        backend = "none"
        direct = False
        notes = FALLBACK_NOTE

    if gpu_before is not None and gpu_after is not None:
        gpu_j = (gpu_after - gpu_before) / 1e3
        notes += " | NVML GPU energy is supplementary; benchmark is CPU-only."
    else:
        gpu_j = NA

    return {
        "measurement_backend": backend,
        "direct_energy_available": direct,
        "cpu_energy_joules": cpu_j,
        "gpu_energy_joules": gpu_j,
        "wall_time_sec": round(wall, 4),
        "cpu_utilization_mean": float(cpu_util),
        "rss_mb": round(rss, 1),
        "notes": notes,
        "workload_result": result,
    }


def per_query_energy(measurement, n_queries):
    """energy_per_query_joules, honest NA propagation."""
    e = measurement["cpu_energy_joules"]
    if e == NA or n_queries <= 0:
        return NA
    return float(e) / float(n_queries)
