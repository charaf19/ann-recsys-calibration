"""Capture GPU hardware/software state for the OPTIONAL GPU experiments.

Complements src/capture_hardware.py (which records the canonical CPU
environment and whether any GPU was used in the main experiments). This
script probes GPU presence, driver/CUDA versions, and FAISS/torch GPU
availability — honestly: every probe degrades to false/None when the stack
is absent, and nothing is ever fabricated.

Outputs:
    results/gpu_experiments/hardware/gpu_hardware.json
    results/gpu_experiments/hardware/gpu_hardware.md
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

SCRIPT = "capture_gpu_hardware"
DEFAULT_OUT = "results/gpu_experiments/hardware"


def _probe_nvml():
    out = {"gpu_present": False, "gpu_count": 0, "gpu_names": [],
           "driver_version": None, "cuda_runtime_version": None}
    try:
        import pynvml
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        out["gpu_present"] = n > 0
        out["gpu_count"] = int(n)
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            out["gpu_names"].append(name.decode() if isinstance(name, bytes)
                                    else str(name))
        drv = pynvml.nvmlSystemGetDriverVersion()
        out["driver_version"] = drv.decode() if isinstance(drv, bytes) else str(drv)
        try:
            v = pynvml.nvmlSystemGetCudaDriverVersion()
            out["cuda_runtime_version"] = f"{v // 1000}.{(v % 1000) // 10}"
        except Exception:
            pass
    except Exception:
        pass  # pynvml optional / no NVIDIA driver
    return out


def _probe_faiss_gpu():
    try:
        import faiss
        return bool(getattr(faiss, "get_num_gpus", lambda: 0)() > 0)
    except Exception:
        return False


def _probe_torch_cuda():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _main_experiments_gpu_used():
    """Read the declared flag from the canonical CPU hardware capture."""
    p = Path("results/hardware/hardware.json")
    if p.is_file():
        try:
            with open(p, "r", encoding="utf-8") as f:
                return bool(json.load(f).get("main_experiments_gpu_used", False))
        except Exception:
            pass
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=DEFAULT_OUT)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    print(f"[{SCRIPT}] starting...")
    print(f"[{SCRIPT}] input path: (live system introspection)")
    print(f"[{SCRIPT}] output path: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    nvml = _probe_nvml()
    info = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_present": nvml["gpu_present"],
        "gpu_count": nvml["gpu_count"],
        "gpu_names": nvml["gpu_names"],
        "cuda_available": nvml["gpu_present"],
        "faiss_gpu_available": _probe_faiss_gpu(),
        "torch_cuda_available": _probe_torch_cuda(),
        "driver_version": nvml["driver_version"],
        "cuda_runtime_version": nvml["cuda_runtime_version"],
        "main_cpu_experiments_gpu_used": _main_experiments_gpu_used(),
        "gpu_experiments_are_exploratory": True,
        "notes": ("All fields are probed live; false/null means the probe "
                  "found nothing (missing driver, missing pynvml, or a "
                  "faiss-cpu-only build) — no GPU details are ever fabricated. "
                  "GPU experiments are exploratory extensions; the canonical "
                  "reproducible benchmark is CPU-only."),
    }

    json_path = out_dir / "gpu_hardware.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"[{SCRIPT}] output path: {json_path}")

    md_path = out_dir / "gpu_hardware.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# GPU hardware capture (optional GPU experiments)\n\n")
        f.write(f"- Captured (UTC): {info['captured_at_utc']}\n")
        f.write(f"- GPU present: {info['gpu_present']} "
                f"(count={info['gpu_count']})\n")
        for name in info["gpu_names"]:
            f.write(f"  - {name}\n")
        f.write(f"- Driver version: {info['driver_version'] or 'unavailable'}\n")
        f.write(f"- CUDA (driver-supported) version: "
                f"{info['cuda_runtime_version'] or 'unavailable'}\n")
        f.write(f"- FAISS GPU available: {info['faiss_gpu_available']}\n")
        f.write(f"- torch CUDA available: {info['torch_cuda_available']}\n")
        f.write(f"- GPU used in main CPU experiments: "
                f"**{info['main_cpu_experiments_gpu_used']}**\n")
        f.write(f"- GPU experiments are exploratory: "
                f"**{info['gpu_experiments_are_exploratory']}**\n\n")
        f.write(f"> {info['notes']}\n")
    print(f"[{SCRIPT}] output path: {md_path}")
    print(f"[{SCRIPT}] completed.")


if __name__ == "__main__":
    main()
