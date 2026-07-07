from pathlib import Path
from typing import Optional

import requests


def _download(
    url: str,
    dest: str,
    *,
    force: bool = False,
    timeout: tuple[int, int] = (10, 120),
    chunk_size: int = 1024 * 1024,
) -> str:
    """Download a URL to disk with streaming and a reusable local cache."""
    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.is_file() and path.stat().st_size > 0 and not force:
        print(f"[download] using cached {path} ({path.stat().st_size:,} bytes)")
        return str(path)

    tmp_path = path.with_suffix(path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    print(f"[download] {url} -> {path}")
    headers = {"User-Agent": "faiss-recsys-benchmark/1.0"}
    downloaded = 0
    expected: Optional[int] = None

    with requests.get(url, stream=True, timeout=timeout, headers=headers) as response:
        response.raise_for_status()
        content_length = response.headers.get("content-length")
        content_encoding = response.headers.get("content-encoding")
        if content_length and content_length.isdigit() and not content_encoding:
            expected = int(content_length)

        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)

    if expected is not None and downloaded != expected:
        tmp_path.unlink(missing_ok=True)
        raise IOError(
            f"Downloaded {downloaded:,} bytes for {url}, expected {expected:,}."
        )
    if downloaded == 0:
        tmp_path.unlink(missing_ok=True)
        raise IOError(f"Downloaded empty file from {url}.")

    tmp_path.replace(path)
    print(f"[download] wrote {path} ({downloaded:,} bytes)")
    return str(path)
