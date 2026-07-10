"""CPU-only scope: no GPU execution paths, imports, or dependencies."""
import re
from pathlib import Path

GPU_EXECUTION_PATTERNS = [
    r"--use_gpu",
    r"faiss-gpu",
    r"faiss_gpu",
    r"index_cpu_to_gpu",
    r"StandardGpuResources",
    r"import\s+pynvml",
    r"from\s+pynvml",
    r"gpu_energy_joules",
]


def _active_py_files(repo_root):
    return [p for p in (repo_root / "src").rglob("*.py")
            if "__pycache__" not in p.parts]


def test_no_gpu_execution_references_in_src(repo_root):
    offenders = []
    for f in _active_py_files(repo_root):
        text = f.read_text(encoding="utf-8")
        for pat in GPU_EXECUTION_PATTERNS:
            if re.search(pat, text):
                offenders.append(f"{f.name}: {pat}")
    assert not offenders, f"GPU execution references found: {offenders}"


def test_no_pynvml_or_gpu_packages_in_requirements(repo_root):
    for req in ("requirements-cpu.txt", "requirements-optional.txt"):
        text = (repo_root / req).read_text(encoding="utf-8").lower()
        for line in text.splitlines():
            line = line.split("#")[0].strip()   # installable spec only
            assert "pynvml" not in line, f"pynvml listed in {req}"
            assert "faiss-gpu" not in line, f"faiss-gpu listed in {req}"


def test_passive_detection_has_no_gpu_dependency(repo_root):
    text = (repo_root / "src" / "utils" / "provenance.py").read_text(
        encoding="utf-8")
    assert 'shutil.which("nvidia-smi")' in text, \
        "passive detection must use a dependency-free presence check"
    assert "pynvml" not in text.lower().replace("no gpu", "")


def test_hardware_fields_are_canonical():
    from utils.provenance import hardware_info
    info = hardware_info()
    assert "accelerator_present" in info
    assert info["main_experiments_accelerator_used"] is False
    assert isinstance(info["accelerator_present"], bool)
    assert "environment metadata only" in info["accelerator_note"]


def test_no_gpu_configs_or_gpu_scripts(repo_root):
    assert not (repo_root / "src" / "run_gpu_experiments.py").exists()
    assert not (repo_root / "src" / "capture_gpu_hardware.py").exists()
    assert not (repo_root / "configs" / "gpu_experiments.yml").exists()
    assert not (repo_root / "configs" / "gpu_optional.yml").exists()
    assert not (repo_root / "legacy" / "experimental_gpu").exists()
    assert not (repo_root / "results" / "gpu_experiments").exists()
    assert not (repo_root / "docs" / "gpu_experiment_protocol.md").exists()


def test_manifest_requires_cpu_scope(repo_root):
    import yaml
    with open(repo_root / "configs" / "paper_evidence_manifest.yml",
              encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    scope = manifest["scope"]
    assert scope["cpu_only"] is True
    assert scope["require_fields"]["main_experiments_accelerator_used"] is False
