"""Dataset statistics table for the paper.

Computes users/items/interactions, density, interactions-per-user quantiles,
temporal LOO train/test sizes, and popularity skew (Gini of item popularity)
for each normalized interactions CSV.

Usage:
    python src/dataset_stats.py --datasets ml-1m:data/ml1m.csv ml-20m:data/ml20m.csv \
        goodbooks:data/goodbooks.csv --min_user_interactions 5 \
        --out_dir results/paper/tables
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils.metrics import gini_exposure
from utils.paths import RESULTS
from utils.preprocessing import (filter_min_user_interactions,
                                 DEFAULT_MIN_USER_INTERACTIONS)
from utils.provenance import write_sources_sidecar
from utils.splits import temporal_leave_one_out
from utils.reporting import write_table

SCRIPT = "dataset_stats"


def stats_for(name, csv_path, min_user_interactions):
    df = pd.read_csv(csv_path, usecols=["user_id", "item_id", "timestamp"])
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)

    # Same canonical k-core filter used before embeddings/splitting/eval, so
    # this table describes exactly the evaluated population (Phase 5 audit).
    df = filter_min_user_interactions(df, min_user_interactions)
    per_user = df.groupby("user_id").size()

    n_users = df["user_id"].nunique()
    n_items = df["item_id"].nunique()
    n_inter = len(df)
    density = n_inter / float(n_users * n_items) if n_users and n_items else 0.0

    train_df, test_df = temporal_leave_one_out(df)
    pop = df.groupby("item_id").size().to_numpy()

    return {
        "dataset": name,
        "users": int(n_users),
        "items": int(n_items),
        "interactions": int(n_inter),
        "density": float(density),
        "inter_per_user_median": float(per_user.median()) if len(per_user) else 0.0,
        "inter_per_user_p95": float(per_user.quantile(0.95)) if len(per_user) else 0.0,
        "train_interactions": int(len(train_df)),
        "test_users": int(len(test_df)),
        "popularity_gini": gini_exposure(pop),
        "min_user_interactions_filter": int(min_user_interactions),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True,
                    help="name:path pairs, e.g. ml-1m:data/ml1m.csv")
    ap.add_argument("--min_user_interactions", type=int,
                    default=DEFAULT_MIN_USER_INTERACTIONS,
                    help="k-core filter; must match data.min_user_interactions "
                         "so the stats table describes the evaluated population")
    ap.add_argument("--out_dir", default=RESULTS["paper_tables"])
    args = ap.parse_args()

    print(f"[{SCRIPT}] starting...")
    out_dir = Path(args.out_dir)
    print(f"[{SCRIPT}] output path: {out_dir}")

    rows = []
    sources = []
    for spec in args.datasets:
        name, _, path = spec.partition(":")
        if not path:
            name, path = Path(spec).stem, spec
        print(f"[{SCRIPT}] input path: {path}")
        if not Path(path).is_file():
            print(f"[{SCRIPT}] WARN: {path} not found; skipping {name}.")
            continue
        rows.append(stats_for(name, path, args.min_user_interactions))
        sources.append(path)

    if not rows:
        print(f"[{SCRIPT}] WARN: no datasets processed.")
        print(f"[{SCRIPT}] completed.")
        return

    df = pd.DataFrame(rows)
    written = write_table(df, out_dir / "dataset_stats", float_fmt="{:.6f}")
    write_sources_sidecar(out_dir / "dataset_stats.csv", sources, SCRIPT)
    for p in written:
        print(f"[{SCRIPT}] output path: {p}")
    print(df.to_string(index=False))
    print(f"[{SCRIPT}] completed.")


if __name__ == "__main__":
    main()
