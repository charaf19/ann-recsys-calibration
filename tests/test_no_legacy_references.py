"""Fail if a legacy pipeline filename or path is reintroduced anywhere in
active documentation, configuration, source, or tests."""
from pathlib import Path

LEGACY_NAMES = [
    "run_grid.py",
    "eval_end2end.py",
    "deployment_guidance.py",
    "synthetic_scaling.py",
    "run_all.sh",
    "run_all.ps1",
    "tasks.sh",
    "sweep.py",
    "smoke_tests.py",
    "generate_synth.py",
    "summary_all.csv",
    "results/figures/",
    "results/report.md",
    "results/scaling/",
    # removed in this cleanup; must not come back either
    "download_datasets.py",
    "clean_data_artifacts.py",
    "prepare_all_datasets",
]

# the cleanup report documents deletions BY NAME on purpose; it is a
# historical record, not active documentation
ALLOWED = {"repository_cleanup_report.md"}

SCAN_GLOBS = [
    ("README.md",), ("reproduction.md",),
    ("docs", "**/*.md"), ("configs", "**/*.yml"),
    ("src", "**/*.py"), ("tests", "**/*.py"),
]


def _files_to_scan(repo_root):
    files = []
    for spec in SCAN_GLOBS:
        base = repo_root / spec[0]
        if len(spec) == 1:
            if base.is_file():
                files.append(base)
        else:
            files.extend(p for p in base.glob(spec[1])
                         if "__pycache__" not in p.parts)
    return [f for f in files if f.name not in ALLOWED]


def test_no_legacy_filenames_anywhere(repo_root):
    offenders = []
    for f in _files_to_scan(repo_root):
        text = f.read_text(encoding="utf-8", errors="replace")
        for name in LEGACY_NAMES:
            if name in text and f.name != Path(__file__).name:
                offenders.append(f"{f.relative_to(repo_root)}: {name}")
    assert not offenders, ("legacy pipeline references reintroduced:\n  "
                           + "\n  ".join(offenders))


def test_legacy_files_do_not_exist(repo_root):
    for name in ["run_grid.py", "eval_end2end.py", "figures.py", "report.py",
                 "deployment_guidance.py", "synthetic_scaling.py", "sweep.py",
                 "smoke_tests.py", "generate_synth.py",
                 "download_datasets.py", "clean_data_artifacts.py"]:
        assert not (repo_root / "src" / name).exists(), \
            f"legacy script resurrected: src/{name}"
    for name in ["run_all.sh", "run_all.ps1", "tasks.sh",
                 "prepare_all_datasets.sh", "prepare_all_datasets.ps1"]:
        assert not (repo_root / name).exists(), \
            f"legacy entry point resurrected: {name}"


def test_no_canonical_script_imports_legacy_modules(repo_root):
    legacy_modules = {"run_grid", "eval_end2end", "deployment_guidance",
                      "synthetic_scaling", "sweep", "smoke_tests",
                      "generate_synth", "download_datasets",
                      "clean_data_artifacts"}
    offenders = []
    for f in (repo_root / "src").rglob("*.py"):
        if "__pycache__" in f.parts:
            continue
        text = f.read_text(encoding="utf-8")
        for mod in legacy_modules:
            if f"import {mod}" in text or f"from {mod}" in text:
                offenders.append(f"{f.name} imports {mod}")
    assert not offenders, offenders
