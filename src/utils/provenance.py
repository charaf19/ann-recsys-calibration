"""Run provenance: manifests, hardware disclosure, environment capture,
and per-artifact source sidecars.

Canonical outputs (all under results/_meta/):
    run_manifest.json   who/what/when of an experiment run
    hardware.json/.md   CPU environment + passive accelerator disclosure
    environment.txt     pip freeze snapshot

CPU-only policy: accelerator presence is detected passively (nvidia-smi on
PATH) purely as environment metadata. No GPU package is imported and no GPU
is ever used by the canonical experiments:
    accelerator_present                  bool (metadata only)
    main_experiments_accelerator_used    always False
"""
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from utils.config import config_hash as compute_config_hash
from utils.paths import RESULTS
from utils.result_io import write_json_atomic, _atomic_write_text

PACKAGES = ["numpy", "scipy", "pandas", "scikit-learn", "faiss-cpu",
            "psutil", "matplotlib", "PyYAML", "torch"]
THREAD_ENV_VARS = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    """Current commit SHA, or the literal 'unknown' (never fabricated)."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10)
        sha = out.stdout.strip()
        return sha if out.returncode == 0 and sha else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def git_dirty():
    """True/False, or ``"unknown"`` when Git metadata is unavailable."""
    try:
        out = subprocess.run(["git", "status", "--porcelain"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return "unknown"
        return bool(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def package_versions(packages=PACKAGES) -> dict:
    from importlib.metadata import version
    versions = {}
    for p in packages:
        try:
            versions[p] = version(p)
        except Exception:
            versions[p] = None
    return versions


def accelerator_present() -> bool:
    """Passive presence check only — no GPU Python dependency is imported."""
    return shutil.which("nvidia-smi") is not None


def hardware_info() -> dict:
    import psutil
    try:
        freq_obj = psutil.cpu_freq()
        freq = ({"current_mhz": freq_obj.current, "min_mhz": freq_obj.min,
                 "max_mhz": freq_obj.max} if freq_obj else None)
    except Exception:
        freq = None
    try:
        import faiss
        faiss_threads = int(faiss.omp_get_max_threads())
    except Exception:
        faiss_threads = None
    return {
        "captured_at_utc": utc_now(),
        "accelerator_present": accelerator_present(),
        "main_experiments_accelerator_used": False,
        "accelerator_note": ("The presence of an accelerator is environment "
                             "metadata only. No GPU was used by the "
                             "canonical experiments."),
        "platform": {k: getattr(platform, k)() for k in
                     ("system", "release", "version", "machine", "processor")},
        "cpu": {"physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True), "freq": freq},
        "memory": {"total_gb": round(psutil.virtual_memory().total / 1024 ** 3, 2)},
        "python": {"version": sys.version, "executable": sys.executable},
        "packages": package_versions(),
        "threads": {"env": {v: os.environ.get(v) for v in THREAD_ENV_VARS},
                    "faiss_omp_max_threads": faiss_threads},
    }


def write_hardware_report(meta_dir=None, info=None) -> list:
    """Write results/_meta/hardware.json and hardware.md; returns paths."""
    meta_dir = Path(meta_dir or RESULTS["meta"])
    info = info or hardware_info()
    json_path = meta_dir / "hardware.json"
    write_json_atomic(info, json_path, mode="replace")
    lines = [
        "# Hardware / environment capture",
        "",
        f"- Captured (UTC): {info['captured_at_utc']}",
        f"- Accelerator present on machine: {info['accelerator_present']}",
        "- Accelerator used in main experiments: **False**",
        "",
        "> The presence of an accelerator is environment metadata only.",
        "> No GPU was used by the canonical experiments.",
        "",
        f"- OS: {info['platform']['system']} {info['platform']['release']}",
        f"- CPU: {info['platform']['processor']}",
        f"- Physical cores: {info['cpu']['physical_cores']}",
        f"- RAM: {info['memory']['total_gb']} GB",
        "",
        "| package | version |",
        "| --- | --- |",
    ]
    lines.extend(f"| {p} | {v or 'not installed'} |"
                 for p, v in info["packages"].items())
    md_path = meta_dir / "hardware.md"
    _atomic_write_text("\n".join(lines) + "\n", md_path)
    return [json_path, md_path]


def write_environment(meta_dir=None):
    """Write results/_meta/environment.txt (pip freeze snapshot)."""
    meta_dir = Path(meta_dir or RESULTS["meta"])
    path = meta_dir / "environment.txt"
    try:
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"],
                                         text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        freeze = f"# pip freeze unavailable: {e}\n"
    _atomic_write_text(freeze, path)
    return path


def make_run_id(cfg_hash: str) -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "__" + cfg_hash


class RunManifest:
    """Lifecycle manifest for an experiment run (results/_meta/run_manifest.json).

    Usage:
        manifest = RunManifest.start(script, resolved_cfg, cfg_hash,
                                     datasets=..., methods=..., modalities=...)
        manifest.add_output(path); manifest.record_failure(combo, error)
        manifest.finish("completed" | "failed")
    """

    def __init__(self, doc: dict, path: Path):
        self.doc = doc
        self.path = path

    @classmethod
    def start(cls, script: str, resolved_config: dict, cfg_hash: str,
              datasets=None, methods=None, modalities=None, meta_dir=None):
        computed_hash = compute_config_hash(resolved_config)
        if cfg_hash != computed_hash:
            raise ValueError(
                f"configuration hash mismatch: supplied {cfg_hash}, "
                f"computed {computed_hash}")
        meta_dir = Path(meta_dir or RESULTS["meta"])
        path = meta_dir / "run_manifest.json"
        hardware = hardware_info()
        doc = {
            "run_id": make_run_id(cfg_hash),
            "project": (resolved_config.get("project") or {}).get(
                "name", "IndexWise-Recsys"),
            "script": script,
            "started_at_utc": utc_now(),
            "finished_at_utc": None,
            "status": "running",
            "resolved_configuration": resolved_config,
            "configuration_hash": cfg_hash,
            "git_commit": git_commit(),
            "git_dirty_state": git_dirty(),
            "python_version": sys.version,
            "dependency_versions": package_versions(),
            "hardware": hardware,
            "datasets": list(datasets or []),
            "methods": list(methods or []),
            "modalities": list(modalities or []),
            "outputs": [],
            "failed_combinations": [],
        }
        write_hardware_report(meta_dir, info=hardware)
        write_environment(meta_dir)
        m = cls(doc, path)
        m.flush()
        return m

    def add_output(self, path):
        p = str(path)
        if p not in self.doc["outputs"]:
            self.doc["outputs"].append(p)
        self.flush()

    def record_failure(self, combination: dict, error: str):
        self.doc["failed_combinations"].append(
            {**combination, "error": str(error)})
        self.flush()

    def finish(self, status: str = "completed"):
        self.doc["status"] = status
        self.doc["finished_at_utc"] = utc_now()
        self.flush()

    def flush(self):
        write_json_atomic(self.doc, self.path, mode="replace")

    @property
    def run_id(self):
        return self.doc["run_id"]


def provenance_columns(run_id: str, cfg_hash: str) -> dict:
    """Extra provenance columns for consolidated CSV rows (metric columns
    stay untouched)."""
    return {
        "run_id": run_id,
        "config_hash": cfg_hash,
        "code_commit": git_commit(),
        "created_at_utc": utc_now(),
    }


def file_sha256(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def infer_config_hash(manifest_path=None) -> str:
    """Return the hash recorded by the canonical run manifest.

    Paper post-processing commands do not resolve an experiment config of
    their own.  Their provenance therefore follows the evidence-producing
    run recorded in ``results/_meta/run_manifest.json``.  ``"unknown"`` is
    returned when that metadata is unavailable or unreadable; metadata is
    never invented.
    """
    path = Path(manifest_path or Path(RESULTS["meta"]) / "run_manifest.json")
    if not path.is_file():
        return "unknown"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return "unknown"
    value = doc.get("configuration_hash", doc.get("config_hash"))
    if value not in (None, ""):
        return str(value)
    resolved = doc.get("resolved_configuration", doc.get("resolved_config"))
    if isinstance(resolved, dict):
        return compute_config_hash(resolved)
    return "unknown"


def sources_sidecar_path(artifact_path) -> Path:
    """Return ``<full-filename>.sources.json`` for an artifact path."""
    artifact_path = Path(artifact_path)
    return artifact_path.with_name(artifact_path.name + ".sources.json")


def write_sources_sidecar(artifact_path, source_files, script: str,
                          cfg_hash: str | None = None,
                          mode: str = "replace"):
    """Atomically write provenance for one generated artifact format.

    The sidecar name retains the artifact's complete filename, so a table's
    CSV, Markdown and LaTeX representations (and a figure's PNG and PDF)
    each receive independent provenance.
    """
    artifact_path = Path(artifact_path)
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"cannot write provenance for missing artifact: {artifact_path}")
    sources = []
    for s in source_files:
        s = Path(s)
        entry = {"path": str(s).replace("\\", "/")}
        if s.is_file():
            entry["sha256"] = file_sha256(s)
        else:
            entry["sha256"] = None
            entry["note"] = "source missing at generation time"
        sources.append(entry)
    source_paths = [entry["path"] for entry in sources]
    source_hashes = {entry["path"]: entry["sha256"] for entry in sources}
    doc = {
        "artifact": artifact_path.name,
        "artifact_path": str(artifact_path).replace("\\", "/"),
        "script": script,
        "generated_at_utc": utc_now(),
        "git_commit": git_commit(),
        "config_hash": (str(cfg_hash) if cfg_hash is not None
                        else infer_config_hash()),
        "source_paths": source_paths,
        "source_file_hashes": source_hashes,
        "sources": sources,
    }
    sidecar = sources_sidecar_path(artifact_path)
    write_json_atomic(doc, sidecar, mode=mode)
    return sidecar
