\
import argparse, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

def fig_latency_vs_ndcg(df, out):
    plt.figure()
    for m in sorted(df["method"].unique()):
        sub = df[df["method"]==m]
        plt.scatter(sub["latency_p95_ms"], sub["ndcg_at_k_mean"], label=m)
    plt.xlabel("Latency p95 (ms)"); plt.ylabel("NDCG@K (mean)"); plt.legend(); plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")

def fig_recall_vs_latency(df, out):
    plt.figure()
    for m in sorted(df["method"].unique()):
        sub = df[df["method"]==m]
        plt.scatter(sub["recall_at_topk_mean"], sub["latency_p95_ms"], label=m)
    plt.xlabel("Recall@K (mean)"); plt.ylabel("Latency p95 (ms)"); plt.legend(); plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")

def fig_coverage_vs_ndcg(df, out):
    plt.figure()
    for m in sorted(df["method"].unique()):
        sub = df[df["method"]==m]
        plt.scatter(sub["coverage_at_k"], sub["ndcg_at_k_mean"], label=m)
    plt.xlabel("Coverage@K"); plt.ylabel("NDCG@K (mean)"); plt.legend(); plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out_dir", default="results/figures")
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if not Path(args.summary).exists():
        print(f"[figures] missing {args.summary}. Run run_grid.py first.")
        return
    df = pd.read_csv(args.summary)
    fig_latency_vs_ndcg(df, f"{args.out_dir}/latency_vs_ndcg.png")
    fig_recall_vs_latency(df, f"{args.out_dir}/recall_vs_latency.png")
    fig_coverage_vs_ndcg(df, f"{args.out_dir}/coverage_vs_ndcg.png")
    print(f"[figures] saved figures to {args.out_dir}")

if __name__ == "__main__":
    main()
