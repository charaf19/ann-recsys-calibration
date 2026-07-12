"""Neural embedding trainer applies the canonical k-core population filter.

These tests exercise the population-filtering path only. Test A runs the BPR
backbone with a tiny CSV and a single epoch (a few dozen SGD steps) — no real
dataset, no index build, no experiment grid.
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINER = REPO_ROOT / "src" / "train_neural_embeddings.py"


def _write_interactions(path):
    """Two qualifying users (>=5) and one below-threshold user.

    raw: 3 users, 8 items, 13 interactions.
    filtered (min=5): 2 users, 6 items, 11 interactions (u3 and items i7,i8 gone).
    """
    rows = []
    for it in range(1, 6):          # u1: 5 interactions on i1..i5
        rows.append({"user_id": "u1", "item_id": f"i{it}", "timestamp": it})
    for it in range(1, 7):          # u2: 6 interactions on i1..i6
        rows.append({"user_id": "u2", "item_id": f"i{it}", "timestamp": it})
    for it in (7, 8):               # u3: 2 interactions (below threshold)
        rows.append({"user_id": "u3", "item_id": f"i{it}", "timestamp": it})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_trainer_filters_population_and_records_metadata(tmp_path):
    """Test A: low-interaction users removed; metadata records raw + filtered."""
    csv = tmp_path / "interactions.csv"
    _write_interactions(csv)
    out_dir = tmp_path / "emb"

    proc = subprocess.run(
        [sys.executable, str(TRAINER),
         "--interactions", str(csv),
         "--backbone", "bpr_matrix_factorization",
         "--dim", "8", "--epochs", "1", "--seed", "42",
         "--min_user_interactions", "5",
         "--out_dir", str(out_dir)],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=180)
    assert proc.returncode == 0, f"trainer failed: {proc.stderr[-800:]}"

    # Explicit population reporting on stdout.
    assert "filtered users=2" in proc.stdout
    assert "filtered items=6" in proc.stdout
    assert "filtered interactions=11" in proc.stdout
    assert "min_user_interactions=5" in proc.stdout

    meta = json.loads((out_dir / "embedding_meta.json").read_text(encoding="utf-8"))
    assert meta["min_user_interactions"] == 5
    # Raw counts preserved.
    assert meta["raw_n_users"] == 3
    assert meta["raw_n_items"] == 8
    assert meta["raw_n_interactions"] == 13
    # Filtered population is authoritative.
    assert meta["n_users"] == 2
    assert meta["n_items"] == 6
    assert meta["n_interactions"] == 11

    # The saved item vectors span only the filtered item population.
    import numpy as np
    item_ids = np.load(out_dir / "item_ids.npy", allow_pickle=True)
    assert len(item_ids) == 6


def test_trainer_rejects_degenerate_population(tmp_path):
    """A threshold that removes everyone must fail loudly, never train raw."""
    csv = tmp_path / "interactions.csv"
    _write_interactions(csv)
    out_dir = tmp_path / "emb"

    proc = subprocess.run(
        [sys.executable, str(TRAINER),
         "--interactions", str(csv),
         "--backbone", "bpr_matrix_factorization",
         "--dim", "8", "--epochs", "1", "--seed", "42",
         "--min_user_interactions", "100",
         "--out_dir", str(out_dir)],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=180)
    assert proc.returncode != 0
    assert not (out_dir / "embedding_meta.json").exists()
