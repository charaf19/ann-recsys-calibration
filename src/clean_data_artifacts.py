import argparse
import shutil
from pathlib import Path
from typing import Iterable


GROUP_PATTERNS = {
    "indices": ("index_*",),
    "stress": ("stress_*",),
    "embeddings": ("emb_*",),
    "synthetic": ("synth*.csv",),
    "prepared": ("ml1m.csv", "ml20m.csv", "goodbooks.csv", "amazon_books.csv"),
    "raw": ("raw", "raw_*.zip", "goodbooks-10k"),
}


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _format_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:,.1f} {unit}" if unit != "B" else f"{value:,.0f} {unit}"
        value /= 1024
    return f"{size:,} B"


def _iter_targets(data_dir: Path, groups: Iterable[str]) -> list[tuple[str, Path]]:
    seen: set[Path] = set()
    targets: list[tuple[str, Path]] = []
    for group in groups:
        for pattern in GROUP_PATTERNS[group]:
            for path in data_dir.glob(pattern):
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                targets.append((group, path))
    return targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run or delete generated files under data/."
    )
    parser.add_argument("--data-dir", default="data", help="Data directory to inspect.")
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=sorted(GROUP_PATTERNS),
        default=["indices", "stress"],
        help="Artifact groups to clean. Defaults to generated indices and stress vectors.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually remove matched artifacts. Without this, only prints a dry run.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Required with --delete to acknowledge removal.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if args.delete and not args.yes:
        raise SystemExit("--delete requires --yes")

    targets = _iter_targets(data_dir, args.groups)
    if not targets:
        print("[clean_data_artifacts] no matching artifacts")
        return

    total = 0
    action = "delete" if args.delete else "dry-run"
    for group, path in sorted(targets, key=lambda item: (item[0], item[1].name)):
        resolved = path.resolve()
        if resolved == data_dir or not _is_relative_to(resolved, data_dir):
            raise RuntimeError(f"Refusing to remove unsafe path: {resolved}")
        size = _size_bytes(resolved)
        total += size
        print(f"[{action}] {group:10s} {_format_size(size):>10s} {path}")
        if args.delete:
            if resolved.is_dir():
                shutil.rmtree(resolved)
            else:
                resolved.unlink()

    print(f"[clean_data_artifacts] matched {len(targets)} paths, total={_format_size(total)}")


if __name__ == "__main__":
    main()
