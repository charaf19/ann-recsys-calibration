from types import SimpleNamespace

import numpy as np

import bootstrap_significance as B
from utils.metrics import bootstrap_ci, paired_bootstrap_test
from utils import provenance


def test_grouped_bootstrap_matches_reference_and_is_deterministic():
    rng = np.random.default_rng(9)
    a = rng.normal(size=17)
    b = rng.normal(size=17)
    grouped = B.grouped_bootstrap_means(
        {"a": a, "diff": a - b}, n_boot=200, seed=42, batch_size=13)
    again = B.grouped_bootstrap_means(
        {"a": a, "diff": a - b}, n_boot=200, seed=42, batch_size=31)
    np.testing.assert_array_equal(grouped["a"], again["a"])
    ci = B._ci(a, grouped["a"])
    ref_ci = bootstrap_ci(a, n_boot=200, seed=42)
    for key in ("mean", "ci_low", "ci_high"):
        assert ci[key] == ref_ci[key]
    test = B._paired(a, b, grouped["diff"])
    ref_test = paired_bootstrap_test(a, b, n_boot=200, seed=42)
    for key in ("mean_diff", "ci_low", "ci_high", "p_value"):
        assert test[key] == ref_test[key]


def test_exact_cpu_name_uses_cim_then_falls_back(monkeypatch):
    monkeypatch.setattr(provenance.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        provenance.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="Intel(R) Core(TM) Ultra 7 155H\n"))
    assert provenance.exact_cpu_name() == "Intel(R) Core(TM) Ultra 7 155H"

    monkeypatch.setattr(provenance.subprocess, "run",
                        lambda *args, **kwargs: (_ for _ in ()).throw(OSError()))
    monkeypatch.setattr(provenance.platform, "processor", lambda: "fallback cpu")
    assert provenance.exact_cpu_name() == "fallback cpu"
