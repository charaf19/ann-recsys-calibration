"""Capture the CPU experiment environment (results/_meta/hardware.*).

CPU-only scope: accelerator presence is recorded passively (nvidia-smi on
PATH) as environment metadata only — no GPU package is imported and
main_experiments_accelerator_used is always false. Also snapshots the
Python environment to results/_meta/environment.txt.
"""
import argparse
import sys
from pathlib import Path

from utils.paths import RESULTS
from utils.provenance import write_hardware_report, write_environment

SCRIPT = "capture_hardware"


def main():
    ap = argparse.ArgumentParser(
        description="Capture CPU hardware/environment metadata for the run "
                    "(passive accelerator disclosure only; CPU-only scope).")
    ap.add_argument("--out_dir", default=RESULTS["meta"],
                    help="metadata directory (default: results/_meta)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    print(f"[{SCRIPT}] starting...")
    print(f"[{SCRIPT}] output path: {out_dir}")

    try:
        for p in write_hardware_report(out_dir):
            print(f"[{SCRIPT}] output path: {p}")
        env_path = write_environment(out_dir)
        print(f"[{SCRIPT}] output path: {env_path}")
    except OSError as e:
        print(f"[{SCRIPT}] ERROR: could not write hardware report: {e}")
        sys.exit(1)
    print(f"[{SCRIPT}] completed.")


if __name__ == "__main__":
    main()
