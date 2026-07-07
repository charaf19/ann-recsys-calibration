"""Synthetic scaling study (CPU-only).

Generates synthetic interaction datasets at increasing catalog sizes (via
generate_synth.py), trains embeddings, builds Flat + ANN indexes, calibrates
the ANN methods at a fixed agreement-recall target, and records how the
calibrated parameter and latency scale with catalog size.

Nothing is executed at import time; running the script performs the study.
Use --dry_run to print the exact commands without executing anything.

Output: results/scaling/scaling.csv (+ per-run JSONs)
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from calibrate import calibrate_index
from utils.ann_io import load_ann_index
from utils.common import set_global_seed
from utils.paths import RESULTS

SCRIPT = "synthetic_scaling"


def run(cmd, dry_run=False):
    full = [sys.executable] + cmd[1:] if cmd[0] == "python" else cmd
    print(">>", " ".join(str(c) for c in full))
    if not dry_run:
        subprocess.run(full, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items_list", type=int, nargs="*",
                    default=[5000, 20000, 50000, 100000])
    ap.add_argument("--users", type=int, default=20000)
    ap.add_argument("--interactions_per_item", type=int, default=20)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--weighting", default="bm25")
    ap.add_argument("--methods", nargs="*", default=["hnsw", "ivfpq"])
    ap.add_argument("--target", type=float, default=0.95)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--calibration_queries", type=int, default=1000)
    ap.add_argument("--budget_mb", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default=RESULTS["scaling"])
    ap.add_argument("--dry_run", action="store_true",
                    help="print the command plan without executing")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    print(f"[{SCRIPT}] starting...")
    print(f"[{SCRIPT}] input path: (synthetic data, generated on the fly)")
    print(f"[{SCRIPT}] output path: {out_dir}")

    set_global_seed(args.seed)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for n_items in args.items_list:
        tag = f"synth{n_items}"
        csv = f"data/{tag}.csv"
        emb = f"data/emb_{tag}_{args.weighting}_d{args.dim}"
        n_inter = n_items * args.interactions_per_item

        run(["python", "src/generate_synth.py", "--users", str(args.users),
             "--items", str(n_items), "--interactions", str(n_inter),
             "--out", csv], args.dry_run)
        run(["python", "src/train_embeddings.py", "--interactions", csv,
             "--dim", str(args.dim), "--weighting", args.weighting,
             "--normalize", "l2", "--seed", str(args.seed),
             "--out_dir", emb], args.dry_run)

        for method in args.methods:
            idx_dir = f"data/index_{tag}_{args.weighting}_d{args.dim}_{method}"
            build = ["python", "src/build_index.py", "--method", method,
                     "--item_vecs", f"{emb}/item_vecs.npy",
                     "--item_ids", f"{emb}/item_ids.npy",
                     "--out_dir", idx_dir, "--budget_mb", str(args.budget_mb)]
            if method == "hnsw":
                build += ["--M", "24", "--efc", "200"]
            elif method in ("ivfflat", "ivfpq"):
                build += ["--nlist", "auto"]
                if method == "ivfpq":
                    build += ["--m", "32", "--bits", "8"]
            run(build, args.dry_run)

            if args.dry_run:
                continue

            item_vecs = np.load(f"{emb}/item_vecs.npy").astype("float32")
            ann = load_ann_index(idx_dir, item_vecs.shape[1], item_vecs.shape[0])
            res = calibrate_index(ann, item_vecs, args.target, args.topk,
                                  args.calibration_queries, args.seed)
            with open(out_dir / f"{tag}__{method}.json", "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
            rows.append({
                "n_items": n_items,
                "n_users": args.users,
                "n_interactions": n_inter,
                "method": method,
                "target_recall": args.target,
                "target_reached": res["target_reached"],
                "param_name": res["param_name"],
                "calibrated_param_value": res["calibrated_param_value"],
                "achieved_recall_vs_exact": res["achieved_recall_vs_exact"],
                "latency_p50_ms": res["latency_ms_at_calibrated"]["p50"],
                "latency_p95_ms": res["latency_ms_at_calibrated"]["p95"],
                "seed": args.seed,
            })

    if rows:
        df = pd.DataFrame(rows)
        csv_path = out_dir / "scaling.csv"
        df.to_csv(csv_path, index=False)
        print(f"[{SCRIPT}] output path: {csv_path}")
    print(f"[{SCRIPT}] completed.")


if __name__ == "__main__":
    main()
