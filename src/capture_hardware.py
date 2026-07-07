"""Capture the hardware/software environment for CPU-only reproducibility.

Records platform, CPU, RAM, Python and library versions, BLAS/OpenMP thread
settings, and whether any GPU was used for the main experiments (should be
false for this benchmark). Also writes a `pip freeze` snapshot.

Outputs (results/hardware/): hardware.json, hardware.md, env_freeze.txt
"""
import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import psutil

SCRIPT = "capture_hardware"

PACKAGES = ["numpy", "scipy", "pandas", "scikit-learn", "faiss-cpu",
            "psutil", "matplotlib", "PyYAML"]
THREAD_ENV_VARS = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]


def _pkg_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def _faiss_threads():
    try:
        import faiss
        return int(faiss.omp_get_max_threads())
    except Exception:
        return None


def _str2bool(v):
    return str(v).strip().lower() in ("1", "true", "yes", "y")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="results/hardware")
    ap.add_argument("--main_experiments_gpu_used", default="false",
                    help="record whether any GPU was used for the main experiments")
    ap.add_argument("--label", default="default", help="hardware profile label")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    print(f"[{SCRIPT}] starting...")
    print(f"[{SCRIPT}] input path: (live system introspection)")
    print(f"[{SCRIPT}] output path: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    freq = None
    try:
        f = psutil.cpu_freq()
        freq = {"current_mhz": f.current, "min_mhz": f.min, "max_mhz": f.max} if f else None
    except Exception:
        pass

    info = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "main_experiments_gpu_used": _str2bool(args.main_experiments_gpu_used),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "cpu": {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "freq": freq,
        },
        "memory": {
            "total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "packages": {p: _pkg_version(p) for p in PACKAGES},
        "threads": {
            "env": {v: os.environ.get(v) for v in THREAD_ENV_VARS},
            "faiss_omp_max_threads": _faiss_threads(),
        },
    }

    json_path = out_dir / "hardware.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"[{SCRIPT}] output path: {json_path}")

    md_path = out_dir / "hardware.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Hardware / environment capture\n\n")
        f.write(f"- Captured (UTC): {info['captured_at_utc']}\n")
        f.write(f"- GPU used for main experiments: "
                f"**{info['main_experiments_gpu_used']}**\n")
        f.write(f"- OS: {info['platform']['system']} {info['platform']['release']} "
                f"({info['platform']['machine']})\n")
        f.write(f"- CPU: {info['platform']['processor']} — "
                f"{info['cpu']['physical_cores']} physical / "
                f"{info['cpu']['logical_cores']} logical cores\n")
        f.write(f"- RAM: {info['memory']['total_gb']} GB\n")
        f.write(f"- Python: {sys.version.split()[0]}\n\n")
        f.write("| package | version |\n| --- | --- |\n")
        for p, v in info["packages"].items():
            f.write(f"| {p} | {v or 'not installed'} |\n")
    print(f"[{SCRIPT}] output path: {md_path}")

    freeze_path = out_dir / "env_freeze.txt"
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        freeze_path.write_text(out, encoding="utf-8")
        print(f"[{SCRIPT}] output path: {freeze_path}")
    except Exception as e:
        print(f"[{SCRIPT}] WARN: pip freeze failed ({e}); skipping {freeze_path}")

    print(f"[{SCRIPT}] completed.")


if __name__ == "__main__":
    main()
