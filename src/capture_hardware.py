"""Capture the CPU experiment environment and passive GPU-presence metadata."""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import psutil

SCRIPT = "capture_hardware"
PACKAGES = ["numpy", "scipy", "pandas", "scikit-learn", "faiss-cpu",
            "psutil", "matplotlib", "PyYAML", "torch"]
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


def _gpu_presence():
    """Report presence only; do not import CUDA, PyNVML, torch, or FAISS GPU APIs."""
    detected = shutil.which("nvidia-smi") is not None
    return {
        "present": detected,
        "detection": "nvidia-smi executable present" if detected else "not detected",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="results/_meta/hardware")
    ap.add_argument("--label", default="default", help="hardware profile label")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{SCRIPT}] starting...")
    print(f"[{SCRIPT}] output path: {out_dir}")

    try:
        freq_obj = psutil.cpu_freq()
        freq = ({"current_mhz": freq_obj.current, "min_mhz": freq_obj.min,
                 "max_mhz": freq_obj.max} if freq_obj else None)
    except Exception:
        freq = None

    gpu = _gpu_presence()
    info = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "gpu_present": gpu["present"],
        "gpu_used_in_main_experiments": False,
        "gpu": gpu,
        "platform": {k: getattr(platform, k)() for k in
                     ("system", "release", "version", "machine", "processor")},
        "cpu": {"physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True), "freq": freq},
        "memory": {"total_gb": round(psutil.virtual_memory().total / 1024 ** 3, 2)},
        "python": {"version": sys.version, "executable": sys.executable},
        "packages": {p: _pkg_version(p) for p in PACKAGES},
        "threads": {"env": {v: os.environ.get(v) for v in THREAD_ENV_VARS},
                    "faiss_omp_max_threads": _faiss_threads()},
    }

    (out_dir / "hardware.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    lines = ["# Hardware / environment capture", "",
             f"- Captured (UTC): {info['captured_at_utc']}",
             f"- GPU present on machine: {info['gpu_present']} ({gpu['detection']})",
             "- GPU used in main experiments: **False**",
             f"- OS: {info['platform']['system']} {info['platform']['release']}",
             f"- CPU: {info['platform']['processor']}",
             f"- RAM: {info['memory']['total_gb']} GB", "",
             "| package | version |", "| --- | --- |"]
    lines.extend(f"| {p} | {v or 'not installed'} |" for p, v in info["packages"].items())
    (out_dir / "hardware.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        (out_dir / "env_freeze.txt").write_text(freeze, encoding="utf-8")
    except Exception as exc:
        print(f"[{SCRIPT}] WARN: pip freeze unavailable: {exc}")
    print(f"[{SCRIPT}] completed.")


if __name__ == "__main__":
    main()
