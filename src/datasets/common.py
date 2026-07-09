from pathlib import Path
import shutil
import sys
import time
from typing import Optional

import requests


class _ProgressBar:
    """Small stderr progress indicator without adding another dependency."""

    def __init__(
        self,
        label: str,
        total: Optional[int] = None,
        *,
        unit: str = "items",
        width: int = 16,
        min_interval: float = 0.25,
    ) -> None:
        self.label = self._shorten(label, 26)
        self.total = total if total and total > 0 else None
        self.unit = unit
        self.width = width
        self.min_interval = min_interval
        self.current = 0
        self.start = time.monotonic()
        self._last_update = 0.0
        self._last_line_len = 0
        self._interactive = sys.stderr.isatty()
        self._closed = False

    def advance(self, amount: int, *, suffix: str = "") -> None:
        self.update(self.current + amount, suffix=suffix)

    def update(self, current: int, *, suffix: str = "", force: bool = False) -> None:
        self.current = max(0, current)
        now = time.monotonic()
        if not force and now - self._last_update < self.min_interval:
            return
        self._last_update = now
        self._emit(self._render(suffix=suffix))

    def finish(self, current: Optional[int] = None, *, suffix: str = "") -> None:
        if current is not None:
            self.current = max(0, current)
        self._emit(self._render(suffix=suffix), final=True)

    def close(self) -> None:
        if self._interactive and not self._closed:
            sys.stderr.write("\n")
            sys.stderr.flush()
        self._closed = True

    def _emit(self, line: str, *, final: bool = False) -> None:
        if self._interactive:
            line = self._fit_to_terminal(line)
            padded = line.ljust(self._last_line_len)
            sys.stderr.write("\r" + padded)
            if final:
                sys.stderr.write("\n")
                self._closed = True
            sys.stderr.flush()
            self._last_line_len = len(line)
        elif final:
            print(line, file=sys.stderr)

    def _render(self, *, suffix: str = "") -> str:
        elapsed = max(time.monotonic() - self.start, 1e-9)
        rate = self.current / elapsed
        current = self._format_value(self.current)
        rate_text = f"{self._format_value(rate)}/s"
        suffix_text = f" {suffix}" if suffix else ""

        if self.total:
            ratio = min(self.current / self.total, 1.0)
            filled = int(self.width * ratio)
            bar = "#" * filled + "-" * (self.width - filled)
            total = self._format_value(self.total)
            remaining = max(self.total - self.current, 0)
            eta = self._format_duration(remaining / rate) if rate > 0 else "--:--"
            return (
                f"[{self.label}] |{bar}| {ratio * 100:6.2f}% "
                f"{current}/{total} {rate_text} eta {eta}{suffix_text}"
            )

        unit_text = "" if self.unit == "B" else f" {self.unit}"
        return f"[{self.label}] {current}{unit_text} {rate_text}{suffix_text}"

    def _format_value(self, value: float) -> str:
        if self.unit == "B":
            units = ("B", "KB", "MB", "GB", "TB")
            size = float(value)
            for unit_name in units:
                if abs(size) < 1024 or unit_name == units[-1]:
                    if unit_name == "B":
                        return f"{size:,.0f} {unit_name}"
                    return f"{size:,.1f} {unit_name}"
                size /= 1024
        return f"{value:,.0f}"

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0, int(seconds))
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    @staticmethod
    def _shorten(value: str, max_len: int) -> str:
        if len(value) <= max_len:
            return value
        return value[: max_len - 3] + "..."

    @staticmethod
    def _fit_to_terminal(value: str) -> str:
        columns = shutil.get_terminal_size((100, 20)).columns
        limit = max(20, columns - 1)
        if len(value) <= limit:
            return value
        return value[: limit - 1] + ">"


def _content_range_total(value: Optional[str]) -> Optional[int]:
    if not value or "/" not in value:
        return None
    total = value.rsplit("/", 1)[-1].strip()
    return int(total) if total.isdigit() else None


def _download(
    url: str,
    dest: str,
    *,
    force: bool = False,
    timeout: tuple[int, int] = (10, 120),
    chunk_size: int = 1024 * 1024,
    max_retries: int = 5,
    retry_delay: float = 2.0,
) -> str:
    """Download a URL to disk with streaming, cache reuse, and resume support."""
    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.is_file() and path.stat().st_size > 0 and not force:
        print(f"[download] using cached {path} ({path.stat().st_size:,} bytes)")
        return str(path)

    tmp_path = path.with_suffix(path.suffix + ".part")
    if force and tmp_path.exists():
        tmp_path.unlink()

    print(f"[download] {url} -> {path}")
    base_headers = {
        "User-Agent": "faiss-recsys-benchmark/1.0",
        "Accept-Encoding": "identity",
    }
    last_err: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        resume_from = tmp_path.stat().st_size if tmp_path.exists() and not force else 0
        headers = dict(base_headers)
        if resume_from:
            headers["Range"] = f"bytes={resume_from}-"
            print(
                f"[download] resume {tmp_path} from {resume_from:,} bytes "
                f"(attempt {attempt}/{max_retries})"
            )
        elif attempt > 1:
            print(f"[download] retry {attempt}/{max_retries}")

        try:
            with requests.get(url, stream=True, timeout=timeout, headers=headers) as response:
                if response.status_code == 416 and resume_from:
                    total = _content_range_total(response.headers.get("content-range"))
                    if total is not None and resume_from == total:
                        tmp_path.replace(path)
                        print(f"[download] wrote {path} ({resume_from:,} bytes)")
                        return str(path)

                response.raise_for_status()

                content_length = response.headers.get("content-length")
                content_encoding = response.headers.get("content-encoding")
                content_range = response.headers.get("content-range")
                resumed = resume_from > 0 and response.status_code == 206
                downloaded = resume_from if resumed else 0
                expected = _content_range_total(content_range)

                if expected is None and content_length and content_length.isdigit() and not content_encoding:
                    expected = downloaded + int(content_length)

                if resume_from and not resumed:
                    print("[download] server ignored resume request; restarting file")

                mode = "ab" if resumed else "wb"
                progress = _ProgressBar(f"download {path.name}", total=expected, unit="B")
                if downloaded:
                    progress.update(downloaded, force=True)

                with open(tmp_path, mode) as f:
                    try:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if not chunk:
                                continue
                            f.write(chunk)
                            downloaded += len(chunk)
                            progress.advance(len(chunk))
                    except Exception:
                        progress.close()
                        raise

                if expected is not None and downloaded != expected:
                    progress.close()
                    raise IOError(
                        f"Downloaded {downloaded:,} bytes for {url}, expected "
                        f"{expected:,}. Partial file retained at {tmp_path}."
                    )
                if downloaded == 0:
                    progress.close()
                    tmp_path.unlink(missing_ok=True)
                    raise IOError(f"Downloaded empty file from {url}.")

                progress.finish(downloaded)
                tmp_path.replace(path)
                print(f"[download] wrote {path} ({downloaded:,} bytes)")
                return str(path)
        except requests.HTTPError as e:
            last_err = e
            status = e.response.status_code if e.response is not None else None
            if status is not None and 400 <= status < 500 and status != 416:
                raise
        except (OSError, requests.RequestException) as e:
            last_err = e

        if attempt < max_retries:
            part_size = tmp_path.stat().st_size if tmp_path.exists() else 0
            part_note = f"; partial={part_size:,} bytes" if part_size else ""
            print(
                f"[download] attempt {attempt}/{max_retries} failed: "
                f"{last_err}{part_note}; retrying in {retry_delay:g}s"
            )
            time.sleep(retry_delay)

    part_size = tmp_path.stat().st_size if tmp_path.exists() else 0
    part_note = f" Partial file retained at {tmp_path} ({part_size:,} bytes)." if part_size else ""
    raise IOError(
        f"Failed to download {url} after {max_retries} attempts.{part_note} "
        f"Last error: {last_err}"
    )
