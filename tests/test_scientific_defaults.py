"""The resolved configuration must equal the paper protocol exactly."""
import yaml

from utils.config import load_config, cfg_get


def resolved(repo_root, name):
    return load_config(repo_root / "configs" / name)


def test_embedding_contract(repo_root):
    cfg = resolved(repo_root, "main_cpu.yml")
    assert cfg_get(cfg, "embedding.method") == "truncated_svd"
    assert cfg_get(cfg, "embedding.weighting") == "bm25"
    assert cfg_get(cfg, "embedding.dim", type=int) == 128
    assert cfg_get(cfg, "embedding.normalize") == "l2"
    assert cfg_get(cfg, "embedding.bm25_k1", type=float) == 1.2
    assert cfg_get(cfg, "embedding.bm25_b", type=float) == 0.75
    assert cfg_get(cfg, "reproducibility.seed", type=int) == 42


def test_retrieval_contract(repo_root):
    cfg = resolved(repo_root, "main_cpu.yml")
    assert cfg_get(cfg, "retrieval.methods") == \
        ["flat", "hnsw", "ivfflat", "ivfpq", "flatpq"]
    assert cfg_get(cfg, "retrieval.modalities") == ["u2i", "i2i"]
    assert cfg_get(cfg, "retrieval.topk", type=int) == 100
    assert cfg_get(cfg, "retrieval.metric_topk", type=int) == 10
    assert cfg_get(cfg, "retrieval.budget_mb", type=int) == 100


def test_index_construction_contract(repo_root):
    cfg = resolved(repo_root, "main_cpu.yml")
    assert cfg_get(cfg, "index.hnsw.M", type=int) == 24
    assert cfg_get(cfg, "index.hnsw.ef_construction", type=int) == 200
    assert cfg_get(cfg, "index.ivf.nlist") == "auto"
    assert cfg_get(cfg, "index.pq.m", type=int) == 32
    assert cfg_get(cfg, "index.pq.bits", type=int) == 8
    assert cfg_get(cfg, "index.ivfpq.use_opq", type=bool) is True


def test_calibration_and_statistics_contract(repo_root):
    cfg = resolved(repo_root, "main_cpu.yml")
    assert [float(t) for t in cfg_get(cfg, "calibration.targets")] == \
        [0.90, 0.95, 0.98]
    assert cfg_get(cfg, "calibration.primary_target", type=float) == 0.95
    assert cfg_get(cfg, "calibration.queries", type=int) == 1000
    assert cfg_get(cfg, "statistics.bootstrap_iterations", type=int) == 2000


def test_evaluation_protocol_values(repo_root):
    cfg = resolved(repo_root, "main_cpu.yml")
    assert str(cfg_get(cfg, "evaluation.queries_large")) == "10000"
    assert str(cfg_get(cfg, "evaluation.queries_ml1m")) == "full"
    assert cfg_get(cfg, "evaluation.latency_queries", type=int) == 2000
    assert cfg_get(cfg, "hardware.cpu_only", type=bool) is True
    assert cfg_get(cfg, "reproducibility.omp_threads", type=int) == 1


def test_scale_stress_grid_is_20_cost_only_cells(repo_root):
    cfg = resolved(repo_root, "analyses.yml")
    ss = cfg_get(cfg, "scale_stress")
    assert ss["catalog_sizes"] == [10000, 50000, 100000, 500000, 1000000]
    assert ss["dimensions"] == [128]
    assert ss["methods"] == ["flat", "hnsw", "ivfflat", "ivfpq"]
    assert float(ss["calibration_target"]) == 0.95
    assert ss["quality_measured"] is False
    n_cells = (len(ss["catalog_sizes"]) * len(ss["dimensions"])
               * len(ss["methods"]))
    assert n_cells == 20


def test_embedding_sensitivity_protocol(repo_root):
    cfg = resolved(repo_root, "analyses.yml")
    es = cfg_get(cfg, "embedding_sensitivity")
    assert es["datasets"] == ["ml-1m"]
    assert es["modalities"] == ["u2i"]
    assert es["backbones"] == ["svd_bm25", "svd_tfidf", "svd_none",
                               "bpr_matrix_factorization"]
    assert es["optional_backbones"] == ["two_tower_mlp"]
    assert len(es["methods"]) == 5
    assert int(es["queries"]) == 10000


def test_decision_framework_weights(repo_root):
    cfg = resolved(repo_root, "analyses.yml")
    w = cfg_get(cfg, "decision_framework.weights")
    assert w == {"quality_retention": 0.45, "latency": 0.30,
                 "memory": 0.15, "long_tail_exposure": 0.10}
    assert cfg_get(cfg, "decision_framework.allow_flatpq_online",
                   type=bool) is False


def test_bootstrap_code_fallback_is_paper_value():
    import bootstrap_significance as B
    assert B.FALLBACK_N_BOOT == 2000


def test_test_tiny_config_is_marked_non_paper(repo_root):
    fixture = repo_root / "tests" / "fixtures" / "test_tiny.yml"
    text = fixture.read_text(encoding="utf-8")
    assert "NOT for paper reproduction" in text
    cfg = load_config(fixture)
    # tiny config may shrink counts, but never silently changes the protocol
    assert cfg_get(cfg, "embedding.weighting") == "bm25"
    assert cfg_get(cfg, "embedding.dim", type=int) == 128


def test_only_canonical_configs_exist(repo_root):
    found = sorted(p.name for p in (repo_root / "configs").glob("*.yml"))
    assert found == ["analyses.yml", "defaults.yml", "main_cpu.yml",
                     "paper_evidence_manifest.yml"]


def test_all_yaml_files_parse(repo_root):
    paths = list((repo_root / "configs").glob("*.yml"))
    paths += list((repo_root / "tests" / "fixtures").glob("*.yml"))
    for p in paths:
        with open(p, encoding="utf-8") as f:
            assert yaml.safe_load(f) is not None
