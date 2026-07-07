import json
import subprocess
import sys
from pathlib import Path


def run(cmd):
    py = sys.executable
    full = [py] + cmd[1:] if isinstance(cmd, list) and len(cmd) > 0 and cmd[0] == "python" else cmd
    print(">>", " ".join(str(c) for c in full))
    subprocess.run(full, check=True)


def main():
    run(["python", "src/generate_synth.py", "--users", "200", "--items", "500", "--interactions", "5000", "--out", "data/synth_smoke.csv"])
    run(["python", "src/train_embeddings.py", "--interactions", "data/synth_smoke.csv", "--dim", "64", "--out_dir", "data/emb_smoke"])
    run(["python", "src/build_index.py", "--method", "hnsw", "--item_vecs", "data/emb_smoke/item_vecs.npy", "--item_ids", "data/emb_smoke/item_ids.npy", "--out_dir", "data/index_smoke_hnsw", "--budget_mb", "50", "--M", "12", "--efc", "100"])
    run(["python", "src/run_device.py", "--index", "data/index_smoke_hnsw", "--item_vecs", "data/emb_smoke/item_vecs.npy", "--queries", "500", "--topk", "50", "--ef", "32"])
    run(["python", "src/eval_end2end.py", "--interactions", "data/synth_smoke.csv", "--item_vecs", "data/emb_smoke/item_vecs.npy", "--index", "data/index_smoke_hnsw", "--ann_method", "hnsw", "--queries", "500", "--topk", "50", "--metric_topk", "10", "--ef", "32"])

    efile = Path("data/index_smoke_hnsw/end2end_hnsw.json")
    if not efile.exists():
        print("Smoke failed: missing end2end JSON", file=sys.stderr)
        sys.exit(1)

    stats = json.load(open(efile))
    required = [
        "ndcg_at_k_mean",
        "hr_at_k_mean",
        "precision_at_k_mean",
        "map_at_k_mean",
        "mrr_at_k_mean",
        "coverage_at_k",
        "gini_exposure",
        "long_tail_uplift",
    ]
    miss = [k for k in required if k not in stats]
    if miss:
        print("Smoke failed: missing metrics:", miss, file=sys.stderr)
        sys.exit(2)

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
