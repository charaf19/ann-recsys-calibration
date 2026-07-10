"""CLI contracts: --help works and advertises the canonical flags.

Runs each entry point with --help in a subprocess (argparse exits before any
scientific work), so this is safe and cheap.
"""
import subprocess
import sys

import pytest

CLIS = {
    "run_revision_experiments.py": ["--config", "--write_mode",
                                    "--allow_missing_datasets"],
    "run_calibration_sensitivity.py": ["--config", "--write_mode"],
    "bootstrap_significance.py": ["--config", "--n_boot", "--write_mode"],
    "run_embedding_backbone_sensitivity.py": ["--config", "--write_mode"],
    "run_scale_stress.py": ["--config", "--write_mode", "--dimensions"],
    "run_exposure_analysis.py": ["--write_mode"],
    "run_pq_diagnostics.py": ["--config", "--write_mode"],
    "effect_size_tables.py": ["--write_mode"],
    "ann_decision_framework.py": ["--config", "--write_mode"],
    "claim_support_audit.py": ["--write_mode", "--validation_report"],
    "tables_paper.py": ["--write_mode"],
    "figures_paper.py": ["--write_mode"],
    "validate_paper_evidence.py": ["--manifest"],
}


def run_help(repo_root, script):
    return subprocess.run(
        [sys.executable, str(repo_root / "src" / script), "--help"],
        capture_output=True, text=True, cwd=str(repo_root), timeout=120)


@pytest.mark.parametrize("script,flags", sorted(CLIS.items()))
def test_help_exits_zero_and_advertises_flags(repo_root, script, flags):
    proc = run_help(repo_root, script)
    assert proc.returncode == 0, f"{script} --help failed: {proc.stderr[-500:]}"
    for flag in flags:
        assert flag in proc.stdout, f"{script} help lacks {flag}"


def test_bootstrap_has_no_iterations_flag(repo_root):
    proc = run_help(repo_root, "bootstrap_significance.py")
    assert "--iterations" not in proc.stdout, \
        "the nonexistent --iterations flag must never (re)appear"


def test_bootstrap_rejects_non_positive_counts(repo_root):
    proc = subprocess.run(
        [sys.executable, str(repo_root / "src" / "bootstrap_significance.py"),
         "--n_boot", "0"],
        capture_output=True, text=True, cwd=str(repo_root), timeout=120)
    assert proc.returncode != 0
    assert "positive" in (proc.stdout + proc.stderr)
