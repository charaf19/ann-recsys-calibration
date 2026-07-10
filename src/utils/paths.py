"""Canonical on-disk layout shared by the revision pipeline scripts.

Keeping the conventions in one place lets run_revision_experiments.py,
run_calibration_sensitivity.py and the reporting scripts agree on where
embeddings, indexes, and results live.
"""
from pathlib import Path

# dataset name -> normalized interactions CSV (produced by prepare_dataset.py)
DATASET_CSV = {
    "ml-1m": "data/ml1m.csv",
    "ml-20m": "data/ml20m.csv",
    "goodbooks": "data/goodbooks.csv",
    "amazon-books": "data/amazon_books.csv",
    "synth": "data/synth.csv",
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
    "hardware": "results/_meta/hardware",
    "status": "results/_meta/status",
    "validation": "results/_meta/validation",
    "main": "results/main",
    "perquery": "results/main/perquery",
    "calibration": "results/main/calibration",
    "analyses": "results/analyses",
    "calibration_sensitivity": "results/analyses/calibration_sensitivity",
    "bootstrap": "results/analyses/bootstrap",
    "effect_sizes": "results/analyses/effect_sizes",
    "deployment_guidance": "results/analyses/ann_decision_framework",
    "embedding_sensitivity": "results/analyses/embedding_sensitivity",
    "pq_diagnostics": "results/analyses/pq_diagnostics",
    "exposure_analysis": "results/analyses/exposure_analysis",
    "scale_stress": "results/analyses/scale_stress",
    "optional_backends": "results/analyses/optional_backends",
    "energy": "results/analyses/energy",
    "paper": "results/paper",
    "paper_tables": "results/paper/tables",
    "figures_paper": "results/paper/figures",
    "archive": "results/archive",
}


def first_existing(*candidates):
    """Return the first existing path among candidates (or the first candidate
    if none exist). Lets consumers accept alternate result filenames, e.g.
    summary_main.csv vs main_results_all.csv."""
    for c in candidates:
        if Path(c).is_file():
            return str(c)
    return str(candidates[0])
