"""Configuration loader contract: inheritance, precedence, paper defaults."""
import pytest

from utils.config import (load_config, cfg_get, config_hash, deep_merge,
                          apply_cli_overrides, ConfigError)


def test_inheritance_resolves_defaults(repo_root):
    cfg = load_config(repo_root / "configs" / "main_cpu.yml")
    # values only present in defaults.yml must be visible through main_cpu.yml
    assert cfg_get(cfg, "embedding.weighting") == "bm25"
    assert cfg_get(cfg, "index.hnsw.M", type=int) == 24
    assert cfg_get(cfg, "project.name") == "IndexWise-Recsys"


def test_cli_override_beats_yaml(repo_root):
    cfg = load_config(repo_root / "configs" / "main_cpu.yml",
                      cli_overrides={"embedding.dim": 64,
                                     "reproducibility.seed": 7})
    assert cfg_get(cfg, "embedding.dim", type=int) == 64
    assert cfg_get(cfg, "reproducibility.seed", type=int) == 7


def test_none_cli_override_is_ignored(repo_root):
    cfg = load_config(repo_root / "configs" / "main_cpu.yml",
                      cli_overrides={"embedding.dim": None})
    assert cfg_get(cfg, "embedding.dim", type=int) == 128


def test_defaults_are_paper_values(repo_root):
    cfg = load_config(repo_root / "configs" / "defaults.yml")
    assert cfg_get(cfg, "embedding.bm25_k1", type=float) == 1.2
    assert cfg_get(cfg, "embedding.bm25_b", type=float) == 0.75
    assert cfg_get(cfg, "embedding.normalize") == "l2"
    assert cfg_get(cfg, "embedding.dim", type=int) == 128
    assert cfg_get(cfg, "reproducibility.seed", type=int) == 42
    assert cfg_get(cfg, "calibration.primary_target", type=float) == 0.95
    assert cfg_get(cfg, "calibration.queries", type=int) == 1000
    assert [float(t) for t in cfg_get(cfg, "calibration.targets")] == \
        [0.90, 0.95, 0.98]


def test_amazon_books_in_main_config(repo_root):
    cfg = load_config(repo_root / "configs" / "main_cpu.yml")
    datasets = cfg_get(cfg, "datasets")
    assert datasets == ["ml-1m", "ml-20m", "goodbooks", "amazon-books"], \
        "Amazon Books must be part of the canonical paper configuration"


def test_bootstrap_default_resolves_to_2000(repo_root):
    cfg = load_config(repo_root / "configs" / "main_cpu.yml")
    assert cfg_get(cfg, "statistics.bootstrap_iterations", type=int) == 2000


def test_deep_merge_semantics():
    base = {"a": {"x": 1, "y": 2}, "l": [1, 2]}
    out = deep_merge(base, {"a": {"y": 3}, "l": [9]})
    assert out == {"a": {"x": 1, "y": 3}, "l": [9]}
    assert base["a"]["y"] == 2  # no mutation


def test_missing_config_is_clear_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yml")


def test_required_key_error_names_the_path():
    with pytest.raises(ConfigError, match="statistics.bootstrap_iterations"):
        cfg_get({}, "statistics.bootstrap_iterations", required=True)


def test_config_hash_is_deterministic(repo_root):
    a = load_config(repo_root / "configs" / "main_cpu.yml")
    b = load_config(repo_root / "configs" / "main_cpu.yml")
    assert config_hash(a) == config_hash(b)
    c = apply_cli_overrides(a, {"embedding.dim": 64})
    assert config_hash(a) != config_hash(c)
