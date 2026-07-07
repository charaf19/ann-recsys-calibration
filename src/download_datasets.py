import argparse
from pathlib import Path
from typing import Callable

from datasets.amazon_books import prepare_amazon_books
from datasets.goodbooks import prepare_goodbooks
from datasets.movielens import prepare_movielens


DEFAULT_DATASETS = ("ml-1m", "ml-20m", "goodbooks")
ALL_DATASETS = DEFAULT_DATASETS + ("amazon-books",)


def _out_path(data_dir: Path, dataset: str) -> Path:
    names = {
        "ml-1m": "ml1m.csv",
        "ml-20m": "ml20m.csv",
        "goodbooks": "goodbooks.csv",
        "amazon-books": "amazon_books.csv",
    }
    return data_dir / names[dataset]


def _prepare_one(dataset: str, out_path: Path) -> str:
    preparers: dict[str, Callable[[str], str]] = {
        "ml-1m": lambda out: prepare_movielens("1m", out),
        "ml-20m": lambda out: prepare_movielens("20m", out),
        "goodbooks": prepare_goodbooks,
        "amazon-books": prepare_amazon_books,
    }
    return preparers[dataset](str(out_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and normalize recommendation datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=ALL_DATASETS,
        default=list(DEFAULT_DATASETS),
        help=(
            "Datasets to prepare. Defaults to the three benchmark datasets: "
            "ml-1m, ml-20m, and goodbooks."
        ),
    )
    parser.add_argument(
        "--all-supported",
        action="store_true",
        help="Prepare every supported dataset, including amazon-books.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory for normalized CSV outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    datasets = list(ALL_DATASETS) if args.all_supported else args.datasets

    print(f"[download_datasets] preparing: {', '.join(datasets)}")
    rows = []
    for dataset in datasets:
        out_path = _out_path(data_dir, dataset)
        written = Path(_prepare_one(dataset, out_path))
        size_mb = written.stat().st_size / (1024 * 1024)
        rows.append((dataset, written, size_mb))
        print(f"[download_datasets] ready {dataset}: {written} ({size_mb:.1f} MB)")

    print("[download_datasets] complete")
    for dataset, path, size_mb in rows:
        print(f"  - {dataset}: {path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
