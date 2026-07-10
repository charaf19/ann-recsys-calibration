"""CPU energy measurement primitives using Intel RAPL when available."""
import os
import time
from pathlib import Path

import psutil

RAPL_ROOT = Path("/sys/class/powercap")
NA = "NA"
FALLBACK_NOTE = ("Direct package-energy measurement unavailable on this "
                 "platform; reporting timing and utilization only.")


def rapl_domains():
    domains = []
    if not RAPL_ROOT.is_dir():
        return domains
    for directory in sorted(RAPL_ROOT.glob("intel-rapl:*")):
        energy_file = directory / "energy_uj"
        if not energy_file.is_file() or not os.access(energy_file, os.R_OK):
            continue
        try:
            name = (directory / "name").read_text().strip()
            max_uj = int((directory / "max_energy_range_uj").read_text().strip())
            int(energy_file.read_text().strip())
            domains.append((name, energy_file, max_uj))
        except (OSError, ValueError, PermissionError):
            continue
    return domains


def read_rapl_uj(domains):
    return {name: int(path.read_text().strip()) for name, path, _ in domains}


def rapl_delta_joules(before, after, domains):
    total_uj = 0
    maxima = {name: maximum for name, _, maximum in domains}
    for name, start in before.items():
        delta = after.get(name, start) - start
        if delta < 0:
            delta += maxima.get(name, 0)
        total_uj += max(0, delta)
    return total_uj / 1e6


def rapl_available():
    return bool(rapl_domains())


def measure(workload_fn, sample_interval_sec=0.2):
    """Measure a CPU workload; unavailable direct counters remain honest NA."""
    del sample_interval_sec  # reserved for a future CPU sampler
    domains = rapl_domains()
    proc = psutil.Process(os.getpid())
    psutil.cpu_percent(interval=None)
    before = read_rapl_uj(domains) if domains else None
    started = time.perf_counter()
    result = workload_fn()
    wall = time.perf_counter() - started
    after = read_rapl_uj(domains) if domains else None

    if domains:
        cpu_j = rapl_delta_joules(before, after, domains)
        backend, direct = "intel_rapl", True
        notes = f"RAPL domains: {sorted(before)}"
    else:
        cpu_j, backend, direct, notes = NA, "none", False, FALLBACK_NOTE
    return {
        "measurement_backend": backend,
        "direct_energy_available": direct,
        "cpu_energy_joules": cpu_j,
        "wall_time_sec": round(wall, 4),
        "cpu_utilization_mean": float(psutil.cpu_percent(interval=None)),
        "rss_mb": round(proc.memory_info().rss / (1024 * 1024), 1),
        "notes": notes,
        "workload_result": result,
    }


def per_query_energy(measurement, n_queries):
    energy = measurement["cpu_energy_joules"]
    if energy == NA or n_queries <= 0:
        return NA
    return float(energy) / float(n_queries)
