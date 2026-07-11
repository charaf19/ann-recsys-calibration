"""Canonical result-path contract (utils/paths.py RESULTS mapping)."""
import pytest

from utils.paths import RESULTS, results_path, require_input

EXPECTED = {
    "meta": "results/_meta",
    "main": "results/main",
    "aggregates": "results/main/aggregates",
    "perquery": "results/main/perquery",
    "calibration": "results/main/calibration",
    "status": "results/main/status",
    "calibration_sensitivity": "results/analyses/calibration_sensitivity",
    "bootstrap": "results/analyses/bootstrap",
    "effect_sizes": "results/analyses/effect_sizes",
    "embedding_sensitivity": "results/analyses/embedding_sensitivity",
    "exposure_analysis": "results/analyses/exposure",
    "pq_diagnostics": "results/analyses/pq_diagnostics",
    "scale_stress": "results/analyses/scale_stress",
    "decision_framework": "results/analyses/decision_framework",
    "optional_backends": "results/analyses/optional_backends",
    "energy": "results/analyses/energy",
    "paper_tables": "results/paper/tables",
    "paper_figures": "results/paper/figures",
    "paper_supplementary": "results/paper/supplementary",
}

LEGACY_KEYS = ["hardware", "validation", "deployment_guidance",
               "figures_paper", "scaling", "gpu_experiments", "archive"]


def test_results_mapping_matches_contract():
    assert RESULTS == EXPECTED


def test_legacy_keys_removed():
    for key in LEGACY_KEYS:
        assert key not in RESULTS, f"legacy RESULTS key reintroduced: {key}"


def test_no_first_existing_fallback():
    import utils.paths as paths
    assert not hasattr(paths, "first_existing"), \
        "legacy-result fallback first_existing() must not exist"


def test_results_path_helper():
    p = results_path("bootstrap", "bootstrap_cis.csv")
    assert str(p).replace("\\", "/") == \
        "results/analyses/bootstrap/bootstrap_cis.csv"
    with pytest.raises(KeyError):
        results_path("gpu_experiments")


def test_require_input_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="canonical input missing"):
        require_input(tmp_path / "absent.csv", "run something first")


def test_directory_scaffold_exists(repo_root):
    for rel in EXPECTED.values():
        assert (repo_root / rel).is_dir(), f"missing scaffold dir: {rel}"


def test_no_archive_directory(repo_root):
    assert not (repo_root / "results" / "archive").exists(), \
        "results/archive must not exist; canonical scripts never read it"


def test_run_config_canonical_path(repo_root):
    """The orchestrator, the manifest, and the validator must agree on ONE
    run-configuration location: results/main/run_config.json."""
    import yaml
    import run_revision_experiments as orchestrator

    resolved = orchestrator.run_config_path()
    assert str(resolved).replace("\\", "/") == "results/main/run_config.json"

    with open(repo_root / "configs" / "paper_evidence_manifest.yml",
              encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    assert manifest["main"]["run_config_file"] == \
        str(resolved).replace("\\", "/")
