"""Strict validator behavior against tiny synthetic contract fixtures.

The passing fixture is generated FROM configs/paper_evidence_manifest.yml —
datasets, methods, modalities, metrics, grids, counts, and required columns
are read from the manifest, never duplicated by hand — so the test tracks
the real contract. All fixtures live in pytest tmp dirs only; the
repository's real results/ tree stays empty and nothing here fabricates
real evidence.
"""
import json
import shutil
from itertools import product
from pathlib import Path

import pandas as pd
import pytest
import yaml

import validate_paper_evidence as V
from utils.config import config_hash

MANIFEST_RELPATH = Path("configs") / "paper_evidence_manifest.yml"


@pytest.fixture(scope="module")
def manifest(repo_root):
    with open(repo_root / MANIFEST_RELPATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# manifest-driven fixture builders (one per critical artifact)
# ---------------------------------------------------------------------------

def _write_csv(root: Path, rel: str, rows):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_json(root: Path, rel: str, doc):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path


def _build_main(root, m):
    spec = m["main"]
    rows = [{
        "dataset": d, "weighting": spec["weighting"], "dim": spec["dim"],
        "modality": mo, "method": me, "seed": spec["seed"], "queries": 100,
        **{c: 0.5 for c in spec["metric_columns_no_nulls"]},
    } for d, mo, me in product(spec["datasets"], spec["modalities"],
                               spec["methods"])]
    assert len(rows) == spec["expected_rows"]
    _write_csv(root, spec["file"], rows)
    _write_json(root, spec["run_config_file"], {
        "datasets": list(spec["datasets"]),
        "modalities": list(spec["modalities"]),
        "methods": list(spec["methods"]),
        **spec["run_config_must_match"],
    })


def _build_calibration(root, m):
    spec = m["calibration_sensitivity"]
    rows = [{
        "dataset": d, "weighting": spec["weighting"], "dim": spec["dim"],
        "method": me, "target_recall": t, "target_reached": True,
        "param_name": "ef" if me == "hnsw" else "nprobe",
        "calibrated_param_value": 32, "achieved_recall_vs_exact": t + 0.005,
        "latency_p50_ms": 0.4, "latency_p95_ms": 1.2,
        "n_calibration_queries": spec["calibration_queries"],
        "seed": spec["seed"],
    } for d, me, t in product(spec["datasets"], spec["methods"],
                              spec["targets"])]
    assert len(rows) == spec["expected_rows"]
    _write_csv(root, spec["file"], rows)


def _build_embedding(root, m):
    spec = m["embedding_sensitivity"]
    rows = [{
        "dataset": d, "backbone": b, "modality": mo, "method": me,
        "dim": spec["dim"], "seed": spec["seed"],
        "ndcg_at_10": 0.05, "recall_at_10": 0.1,
        "ann_recall_vs_exact_at_k_mean": 0.96,
        "backend_available": True, "status": spec["required_success_status"],
        "error_message": "", "ann_ranking_stability": 0.9,
    } for d, b, mo, me in product(spec["datasets"], spec["backbones"],
                                  spec["modalities"], spec["methods"])]
    assert len(rows) == spec["expected_rows"]
    _write_csv(root, spec["file"], rows)


def _build_scale(root, m):
    spec = m["scale_stress"]
    rows = [{
        "n_items": n, "dim": dd, "method": me, "seed": spec["seed"],
        "build_wall_time_sec": 1.5, "index_size_mb": 12.0,
        "rss_mb_after": 300.0, "calibration_target": 0.95,
        "target_reached": True, "achieved_recall_vs_exact": 0.955,
        "latency_p50_ms": 0.6, "latency_p95_ms": 2.4,
        "quality_measured": False,
    } for n, dd, me in product(spec["catalog_sizes"], spec["dimensions"],
                               spec["methods"])]
    assert len(rows) == spec["expected_rows"]
    _write_csv(root, spec["file"], rows)


def _build_bootstrap(root, m):
    spec = m["bootstrap"]
    common = {"weighting": spec["weighting"], "dim": spec["dim"],
              "seed": spec["seed"], "n_boot": spec["n_boot"]}
    cis = [{
        "dataset": d, "modality": mo, "method": me, "metric": met,
        "mean": 0.1, "ci_low": 0.09, "ci_high": 0.11, "n": 100, **common,
    } for d, mo, me, met in product(spec["datasets"], spec["modalities"],
                                    spec["methods"], spec["metrics"])]
    assert len(cis) == spec["expected_cis_rows"]
    _write_csv(root, spec["cis_file"], cis)
    tests = [{
        "dataset": d, "modality": mo, "method": me, "baseline": "flat",
        "metric": met, "mean_diff": -0.001, "ci_low": -0.002,
        "ci_high": 0.001, "p_value": 0.4, "n": 100, **common,
    } for d, mo, me, met in product(spec["datasets"], spec["modalities"],
                                    spec["comparison_methods"],
                                    spec["comparison_metrics"])]
    assert len(tests) == spec["expected_tests_rows"]
    _write_csv(root, spec["tests_file"], tests)


def _build_effect_sizes(root, m):
    spec = m["effect_sizes"]
    rows = [{
        "dataset": d, "weighting": spec["weighting"], "dim": spec["dim"],
        "modality": mo, "method": me, "baseline": spec["baseline"],
        "metric": met, "mean_diff": -0.001,
        "cohens_d": -0.01, "cohens_d_magnitude": "negligible",
        "cliffs_delta": -0.02, "cliffs_delta_magnitude": "negligible",
        "n": 100, "seed": spec["seed"],
    } for d, mo, me, met in product(spec["datasets"], spec["modalities"],
                                    spec["methods"], spec["metrics"])]
    assert len(rows) == spec["expected_rows"]
    _write_csv(root, spec["file"], rows)


def _build_exposure(root, m):
    spec = m["exposure"]
    scalar_scope = {
        "user_popularity_calibration_error": "popularity_calibration_proxy",
    }
    default_scope = "long_tail_exposure_proxy_only"
    rows = []
    for d, mo, me in product(spec["datasets"], spec["modalities"],
                             spec["methods"]):
        base = {"dataset": d, "weighting": spec["weighting"],
                "dim": spec["dim"], "modality": mo, "method": me,
                "seed": spec["seed"]}
        for metric in spec["required_metrics"]:
            scope = scalar_scope.get(metric, default_scope)
            if metric == "exposure_share_decile":
                for decile in range(10):
                    rows.append({**base, "metric": metric, "k": 10,
                                 "decile": decile, "group": None,
                                 "value": 0.1, "fairness_scope": scope})
            elif metric == "exposure_share_group":
                for group in ("head", "mid", "tail"):
                    rows.append({**base, "metric": metric, "k": 10,
                                 "decile": None, "group": group,
                                 "value": 0.33, "fairness_scope": scope})
            else:
                for k in (10, 100):
                    rows.append({**base, "metric": metric, "k": k,
                                 "decile": None, "group": None,
                                 "value": 0.2, "fairness_scope": scope})
    assert len(rows) >= spec["expected_rows_min"]
    _write_csv(root, spec["file"], rows)


def _build_pq(root, m):
    spec = m["pq_diagnostics"]
    rows = [{
        "dataset": d, "weighting": spec["weighting"], "dim": spec["dim"],
        "method": me, "metric": met, "decile": None, "value": 0.3,
        "seed": spec["seed"],
    } for d, me, met in product(spec["datasets"], spec["methods"],
                                spec["required_metrics"])]
    assert len(rows) >= spec["expected_rows_min"]
    _write_csv(root, spec["file"], rows)


def _build_decision(root, m):
    spec = m["decision_framework"]
    weights = spec["required_values"]
    rows = []
    for d, mo in product(spec["datasets"], spec["modalities"]):
        for rank, me in enumerate(spec["methods"], start=1):
            rows.append({
                "dataset": d, "weighting": spec["weighting"],
                "dim": spec["dim"], "modality": mo, "method": me,
                "seed": spec["seed"], "quality_retention": 1.0,
                "latency_score": 0.8, "memory_score": 0.7,
                "exposure_score": 0.6, **weights,
                "deployment_score": 0.8, "deployment_rank": rank,
                "recommended_use_case": "online_serving_recommended",
                "recommendation_reason": "synthetic fixture row",
            })
    assert len(rows) == spec["expected_rows"]
    _write_csv(root, spec["file"], rows)


def _build_dataset_stats(root, m):
    spec = m["dataset_stats"]
    rows = [{
        "dataset": d, "users": 1000, "items": 500, "interactions": 20000,
        "density": 0.04, "inter_per_user_median": 12.0,
        "inter_per_user_p95": 60.0, "train_interactions": 19000,
        "test_users": 1000, "popularity_gini": 0.6,
        **spec["required_values"],
    } for d in spec["datasets"]]
    assert len(rows) == spec["expected_rows"]
    _write_csv(root, spec["file"], rows)


def _build_hardware(root, m):
    spec = m["hardware"]
    doc = {
        "captured_at_utc": "2026-07-11T00:00:00+00:00",
        "accelerator_present": False,
        "main_experiments_accelerator_used": False,
        "accelerator_note": ("The presence of an accelerator is environment "
                             "metadata only. No GPU was used by the "
                             "canonical experiments."),
        "platform": {"system": "TestOS"},
        "cpu": {"physical_cores": 4},
        "memory": {"total_gb": 16},
        "python": {"version": "3.11-test"},
        "packages": {"numpy": "test"},
        "threads": {"faiss_omp_max_threads": 1},
    }
    missing = [f for f in spec["required_fields"] if f not in doc]
    assert not missing, f"fixture out of date with manifest: {missing}"
    _write_json(root, spec["file"], doc)


def _build_run_manifest(root, m):
    spec = m["run_manifest"]
    resolved = {
        "project": {"name": "IndexWise-Recsys"},
        "reproducibility": {"seed": 42},
        "embedding": {"weighting": "bm25", "dim": 128},
        "hardware": {"cpu_only": True},
        "datasets": list(spec["datasets"]),
        "retrieval": {"methods": list(spec["methods"]),
                      "modalities": list(spec["modalities"])},
    }
    cfg_hash = config_hash(resolved)
    doc = {
        "run_id": f"20260711T000000Z__{cfg_hash}",
        "project": spec["required_values"]["project"],
        "started_at_utc": "2026-07-11T00:00:00+00:00",
        "finished_at_utc": "2026-07-11T01:00:00+00:00",
        "status": spec["required_values"]["status"],
        "resolved_configuration": resolved,
        "configuration_hash": cfg_hash,
        "git_commit": "testfixture123",
        "git_dirty_state": False,
        "python_version": "3.11-test",
        "dependency_versions": {"numpy": "test"},
        "hardware": {"main_experiments_accelerator_used": False},
        "datasets": list(spec["datasets"]),
        "methods": list(spec["methods"]),
        "modalities": list(spec["modalities"]),
        "outputs": [str(spec["required_output"])],
        "failed_combinations": [],
    }
    missing = [f for f in spec["required_fields"] if f not in doc]
    assert not missing, f"fixture out of date with manifest: {missing}"
    _write_json(root, spec["file"], doc)


def build_passing_fixture(root: Path, m: dict):
    """Write every critical artifact the manifest requires. The optional
    claim-support audit is deliberately absent (its absence must not flip
    the critical verdict)."""
    _build_main(root, m)
    _build_calibration(root, m)
    _build_embedding(root, m)
    _build_scale(root, m)
    _build_bootstrap(root, m)
    _build_effect_sizes(root, m)
    _build_exposure(root, m)
    _build_pq(root, m)
    _build_decision(root, m)
    _build_dataset_stats(root, m)
    _build_hardware(root, m)
    _build_run_manifest(root, m)


@pytest.fixture()
def contract_env(tmp_path, monkeypatch, repo_root, manifest):
    """Isolated cwd with the real manifest and a passing synthetic fixture."""
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    shutil.copy(repo_root / MANIFEST_RELPATH,
                cfg_dir / MANIFEST_RELPATH.name)
    build_passing_fixture(tmp_path, manifest)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def run_validator():
    return V.main([])


def load_report(root):
    return json.loads((root / "results/_meta/validation_report.json")
                      .read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# behavior
# ---------------------------------------------------------------------------

def test_complete_tiny_fixture_passes(contract_env):
    assert run_validator() == 0
    report = load_report(contract_env)
    assert report["valid"] is True
    assert report["n_critical_failures"] == 0
    for section in ("main", "bootstrap", "run_manifest", "hardware", "scope"):
        assert report["sections"][section]["status"] == "pass", section
    # the optional claim audit may be absent without flipping the verdict
    audit = report["sections"]["claim_support_audit"]
    assert audit["severity"] == "optional"


def test_fails_with_only_three_datasets(contract_env, manifest):
    p = contract_env / manifest["main"]["file"]
    df = pd.read_csv(p)
    df = df[df["dataset"] != "amazon-books"]
    df.to_csv(p, index=False)
    assert run_validator() == 1
    assert load_report(contract_env)["sections"]["main"]["status"] == "fail"


def test_fails_with_1000_bootstrap_iterations(contract_env, manifest):
    for key in ("cis_file", "tests_file"):
        p = contract_env / manifest["bootstrap"][key]
        df = pd.read_csv(p)
        df["n_boot"] = 1000
        df.to_csv(p, index=False)
    assert run_validator() == 1
    assert load_report(contract_env)["sections"]["bootstrap"]["status"] == "fail"


def test_fails_when_bpr_backbone_absent(contract_env, manifest):
    p = contract_env / manifest["embedding_sensitivity"]["file"]
    df = pd.read_csv(p)
    df = df[df["backbone"] != "bpr_matrix_factorization"]
    df.to_csv(p, index=False)
    assert run_validator() == 1
    assert (load_report(contract_env)["sections"]["embedding_sensitivity"]
            ["status"] == "fail")


def test_fails_when_scale_row_claims_quality(contract_env, manifest):
    p = contract_env / manifest["scale_stress"]["file"]
    df = pd.read_csv(p)
    df.loc[0, "quality_measured"] = True
    df.to_csv(p, index=False)
    assert run_validator() == 1
    assert (load_report(contract_env)["sections"]["scale_stress"]["status"]
            == "fail")


def test_fails_when_gpu_field_in_hardware(contract_env, manifest):
    p = contract_env / manifest["hardware"]["file"]
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["main_experiments_accelerator_used"] = True
    doc["gpu_device"] = "TestGPU"
    p.write_text(json.dumps(doc), encoding="utf-8")
    assert run_validator() == 1
    assert (load_report(contract_env)["sections"]["hardware"]["status"]
            == "fail")


def test_fails_when_run_config_at_legacy_meta_path(contract_env, manifest):
    """run_config.json under results/_meta must NOT satisfy the contract."""
    canonical = contract_env / manifest["main"]["run_config_file"]
    legacy = contract_env / "results/_meta/run_config.json"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    canonical.replace(legacy)
    assert run_validator() == 1
    assert load_report(contract_env)["sections"]["main"]["status"] == "fail"


def test_fails_on_empty_results(tmp_path, monkeypatch, repo_root):
    """An empty results tree is the expected pre-rerun state: FAIL, loudly."""
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    shutil.copy(repo_root / MANIFEST_RELPATH,
                cfg_dir / MANIFEST_RELPATH.name)
    monkeypatch.chdir(tmp_path)
    assert run_validator() == 1
    report = load_report(tmp_path)
    assert report["valid"] is False
    assert report["n_critical_failures"] > 0
