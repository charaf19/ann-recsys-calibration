"""Canonical on-disk layout shared by every pipeline script.

RESULTS is the single mapping from logical output names to directories.
Canonical scripts must build result paths through it — never from string
literals — and must fail clearly when a canonical input is missing instead
of falling back to legacy filenames.
"""
from pathlib import Path

# dataset name -> normalized interactions CSV (produced by prepare_dataset.py)
DATASET_CSV = {
    "ml-1m": "data/ml1m.csv",
    "ml-20m": "data/ml20m.csv",
    "goodbooks": "data/goodbooks.csv",
    "amazon-books": "data/amazon_books.csv",
}


def dataset_csv(dataset: str) -> str:
    return DATASET_CSV.get(dataset, f"data/{dataset}.csv")


def dataset_stem(dataset: str) -> str:
    """Filesystem-safe stem, e.g. ml-1m -> ml1m."""
    return Path(dataset_csv(dataset)).stem


def emb_dir(dataset: str, weighting: str, dim: int) -> str:
    return f"data/emb_{dataset_stem(dataset)}_{weighting}_d{dim}"


def index_dir(dataset: str, weighting: str, dim: int, method: str) -> str:
    return f"data/index_{dataset_stem(dataset)}_{weighting}_d{dim}_{method}"


RESULTS = {
    "meta": "results/_meta",

    "main": "results/main",
    "aggregates": "results/main/aggregates",
    "perquery": "results/main/perquery",
    "calibration": "results/main/calibration",
    "status": "results/main/status",

    "calibration_sensitivity":
        "results/analyses/calibration_sensitivity",
    "bootstrap":
        "results/analyses/bootstrap",
    "effect_sizes":
        "results/analyses/effect_sizes",
    "embedding_sensitivity":
        "results/analyses/embedding_sensitivity",
    "exposure_analysis":
        "results/analyses/exposure",
    "pq_diagnostics":
        "results/analyses/pq_diagnostics",
    "scale_stress":
        "results/analyses/scale_stress",
    "decision_framework":
        "results/analyses/decision_framework",
    "optional_backends":
        "results/analyses/optional_backends",
    "energy":
        "results/analyses/energy",

    "paper_tables":
        "results/paper/tables",
    "paper_figures":
        "results/paper/figures",
    "paper_supplementary":
        "results/paper/supplementary",
}


def results_path(key: str, *parts) -> Path:
    """Canonical result path: results_path("bootstrap", "bootstrap_cis.csv")."""
    if key not in RESULTS:
        raise KeyError(f"unknown RESULTS key '{key}' "
                       f"(known: {sorted(RESULTS)})")
    return Path(RESULTS[key]).joinpath(*parts)


def require_input(path, hint: str) -> Path:
    """Fail clearly when a canonical input is missing (no legacy fallback)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"canonical input missing: {p}\n  produce it first with: {hint}")
    return p
