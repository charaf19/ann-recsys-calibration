"""Strict validator behavior against tiny synthetic contract fixtures.

The fixtures live in pytest tmp dirs only — the repository's results/ tree
stays empty, and nothing here fabricates real evidence.
"""
import json
import shutil
from itertools import product
from pathlib import Path

import pandas as pd
import pytest

import validate_paper_evidence as V

DATASETS = ["ml-1m", "ml-20m", "goodbooks", "amazon-books"]
MODALITIES = ["u2i", "i2i"]
METHODS = ["flat", "hnsw", "ivfflat", "ivfpq", "flatpq"]
TUNABLE = ["hnsw", "ivfflat", "ivfpq"]
TARGETS = [0.90, 0.95, 0.98]
BACKBONES = ["svd_bm25", "svd_tfidf", "svd_none", "bpr_matrix_factorization"]
SIZES = [10000, 50000, 100000, 500000, 1000000]
DIMS = [64, 128, 256]


def build_passing_fixture(root: Path):
    """Write a synthetic evidence set satisfying the full contract."""
    def w(rel, df):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)

    meta = root / "results/_meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "hardware.json").write_text(json.dumps({
        "accelerator_present": False,
        "main_experiments_accelerator_used": False,
    }), encoding="utf-8")

    main = root / "results/main"
    main.mkdir(parents=True, exist_ok=True)
    (main / "run_config.json").write_text(json.dumps(
        {"weighting": "bm25", "dim": 128, "seed": 42}), encoding="utf-8")
    w("results/main/summary_main.csv", pd.DataFrame([
        {"dataset": d, "weighting": "bm25", "dim": 128, "modality": mo,
         "method": me, "seed": 42, "queries": 100,
         "recall_at_k_mean": 0.1, "ndcg_at_k_mean": 0.05,
         "ann_recall_vs_exact_at_k_mean": 0.96,
         "long_tail_exposure": 0.2, "long_tail_uplift": 0.01,
         "latency_p50_ms": 0.5, "latency_p95_ms": 1.5}
        for d, mo, me in product(DATASETS, MODALITIES, METHODS)]))

    w("results/analyses/calibration_sensitivity/calibration_sensitivity.csv",
      pd.DataFrame([
          {"dataset": d, "weighting": "bm25", "dim": 128, "method": me,
           "target_recall": t, "target_reached": True,
           "calibrated_param_value": 32, "achieved_recall_vs_exact": t + 0.01,
           "latency_p95_ms": 1.0, "seed": 42}
          for d, me, t in product(DATASETS, TUNABLE, TARGETS)]))

    w("results/analyses/embedding_sensitivity/"
      "embedding_backbone_sensitivity_all.csv",
      pd.DataFrame([
          {"dataset": "ml-1m", "backbone": b, "modality": "u2i", "method": me,
           "dim": 128, "seed": 42, "backend_available": True, "status": "ok",
           "error_message": "", "ann_ranking_stability": 0.9}
          for b, me in product(BACKBONES, METHODS)]))

    w("results/analyses/scale_stress/scale_stress_all.csv", pd.DataFrame([
        {"n_items": n, "dim": dd, "method": me, "seed": 42,
         "build_wall_time_sec": 1.0, "index_size_mb": 10.0,
         "latency_p95_ms": 2.0, "quality_measured": False}
        for n, dd, me in product(SIZES, DIMS, METHODS)]))

    w("results/analyses/bootstrap/bootstrap_cis.csv", pd.DataFrame([
        {"dataset": d, "modality": mo, "method": me, "metric": met,
         "mean": 0.1, "ci_low": 0.09, "ci_high": 0.11, "n_boot": 2000}
        for d, mo, me, met in product(DATASETS, MODALITIES, METHODS,
                                      ["recall", "ndcg"])]))
    w("results/analyses/bootstrap/paired_tests.csv", pd.DataFrame([
        {"dataset": d, "modality": mo, "method": me, "baseline": "flat",
         "metric": met, "mean_diff": -0.001, "p_value": 0.4, "n_boot": 2000}
        for d, mo, me, met in product(DATASETS, MODALITIES,
                                      [m for m in METHODS if m != "flat"],
                                      ["recall", "ndcg"])]))

    w("results/analyses/effect_sizes/effect_sizes.csv", pd.DataFrame([
        {"dataset": d, "modality": mo, "method": me, "baseline": "flat",
         "metric": "ndcg", "cohens_d": -0.01, "cliffs_delta": -0.02}
        for d, mo, me in product(DATASETS, MODALITIES,
                                 [m for m in METHODS if m != "flat"])]))

    w("results/analyses/exposure/exposure_analysis_all.csv", pd.DataFrame([
        {"dataset": d, "weighting": "bm25", "modality": mo, "method": me,
         "metric": "long_tail_exposure", "value": 0.2,
         "fairness_scope": "long_tail_exposure_proxy_only"}
        for d, mo, me in product(DATASETS, MODALITIES, METHODS)]))

    w("results/analyses/pq_diagnostics/pq_diagnostics_all.csv", pd.DataFrame([
        {"dataset": d, "weighting": "bm25", "dim": 128, "method": me,
         "metric": met, "value": 0.3, "seed": 42}
        for d, me, met in product(DATASETS, ["flatpq", "ivfpq"],
                                  ["reconstruction_error_rel_mean",
                                   "neighbor_overlap_at_10",
                                   "neighbor_overlap_at_100"])]))


@pytest.fixture()
def contract_env(tmp_path, monkeypatch, repo_root):
    """Isolated cwd with the real manifest and a passing synthetic fixture."""
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    shutil.copy(repo_root / "configs" / "paper_evidence_manifest.yml",
                cfg_dir / "paper_evidence_manifest.yml")
    build_passing_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def run_validator():
    return V.main([])


def test_complete_tiny_fixture_passes(contract_env):
    assert run_validator() == 0
    report = json.loads((contract_env / "results/_meta/validation_report.json")
                        .read_text(encoding="utf-8"))
    assert report["valid"] is True
    assert report["sections"]["main"]["status"] == "pass"


def test_fails_with_only_three_datasets(contract_env):
    p = contract_env / "results/main/summary_main.csv"
    df = pd.read_csv(p)
    df = df[df["dataset"] != "amazon-books"]
    df.to_csv(p, index=False)
    assert run_validator() == 1
    report = json.loads((contract_env / "results/_meta/validation_report.json")
                        .read_text(encoding="utf-8"))
    assert report["sections"]["main"]["status"] == "fail"


def test_fails_with_1000_bootstrap_iterations(contract_env):
    for name in ("bootstrap_cis.csv", "paired_tests.csv"):
        p = contract_env / "results/analyses/bootstrap" / name
        df = pd.read_csv(p)
        df["n_boot"] = 1000
        df.to_csv(p, index=False)
    assert run_validator() == 1
    report = json.loads((contract_env / "results/_meta/validation_report.json")
                        .read_text(encoding="utf-8"))
    assert report["sections"]["bootstrap"]["status"] == "fail"


def test_fails_when_bpr_backbone_absent(contract_env):
    p = (contract_env / "results/analyses/embedding_sensitivity"
         / "embedding_backbone_sensitivity_all.csv")
    df = pd.read_csv(p)
    df = df[df["backbone"] != "bpr_matrix_factorization"]
    df.to_csv(p, index=False)
    assert run_validator() == 1
    report = json.loads((contract_env / "results/_meta/validation_report.json")
                        .read_text(encoding="utf-8"))
    assert report["sections"]["embedding_sensitivity"]["status"] == "fail"


def test_fails_when_scale_row_claims_quality(contract_env):
    p = contract_env / "results/analyses/scale_stress/scale_stress_all.csv"
    df = pd.read_csv(p)
    df.loc[0, "quality_measured"] = True
    df.to_csv(p, index=False)
    assert run_validator() == 1
    report = json.loads((contract_env / "results/_meta/validation_report.json")
                        .read_text(encoding="utf-8"))
    assert report["sections"]["scale_stress"]["status"] == "fail"


def test_fails_on_empty_results(tmp_path, monkeypatch, repo_root):
    """An empty results tree is the expected pre-rerun state: FAIL, loudly."""
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    shutil.copy(repo_root / "configs" / "paper_evidence_manifest.yml",
                cfg_dir / "paper_evidence_manifest.yml")
    monkeypatch.chdir(tmp_path)
    assert run_validator() == 1


def test_fails_when_gpu_field_in_hardware(contract_env):
    p = contract_env / "results/_meta/hardware.json"
    p.write_text(json.dumps({"accelerator_present": True,
                             "main_experiments_accelerator_used": True}),
                 encoding="utf-8")
    assert run_validator() == 1
    report = json.loads((contract_env / "results/_meta/validation_report.json")
                        .read_text(encoding="utf-8"))
    assert report["sections"]["cpu_scope"]["status"] == "fail"
