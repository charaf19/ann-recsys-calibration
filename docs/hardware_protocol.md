# CPU hardware protocol

Run `python src/capture_hardware.py` before a fresh canonical run. It writes platform, CPU, memory, Python, package, thread, and environment metadata to `results/_meta/hardware/`.

The benchmark uses `faiss-cpu`. GPU presence is disclosed passively from the host environment; `gpu_used_in_main_experiments` is fixed to false. No canonical workflow requires `faiss-gpu`, CUDA, PyNVML, or GPU outputs.

IndexWise-Recsys is evaluated as a CPU-only framework. GPU acceleration and GPU-specific latency behavior are outside the current scope. GPU latency, transfer behavior, and GPU memory behavior are not evaluated.
