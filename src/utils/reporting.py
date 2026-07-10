"""Reporting helpers: standardized logging banners and table writers.

Every pipeline script uses `banner()` so logs follow the required format:

    [script_name] starting...
    [script_name] input path: ...
    [script_name] output path: ...
    [script_name] completed.
"""
import json
from pathlib import Path

from utils.provenance import sources_sidecar_path, write_sources_sidecar
from utils.result_io import (preflight_output, resolve_write_mode,
                             write_json_atomic, write_text_atomic)


def log(script: str, msg: str):
    print(f"[{script}] {msg}")


def banner_start(script: str, inputs=None, outputs=None):
    log(script, "starting...")
    for p in _as_list(inputs):
        log(script, f"input path: {p}")
    for p in _as_list(outputs):
        log(script, f"output path: {p}")


def banner_done(script: str):
    log(script, "completed.")


def _as_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path, mode="replace"):
    """Write JSON atomically (kept for compatibility with report scripts)."""
    return write_json_atomic(obj, path, mode=mode)


def df_to_markdown(df, float_fmt="{:.4f}"):
    """Render a DataFrame as a GitHub-flavored markdown table without
    requiring the optional 'tabulate' dependency."""
    def fmt(v):
        if isinstance(v, float):
            return float_fmt.format(v)
        return "" if v is None else str(v)

    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in row.tolist()) + " |")
    return "\n".join(lines) + "\n"


def table_artifact_paths(out_base, latex=True):
    """Return the concrete output paths for a multi-format table."""
    out_base = Path(out_base)
    suffixes = [".csv", ".md"] + ([".tex"] if latex else [])
    return [out_base.with_suffix(suffix) for suffix in suffixes]


def write_table(df, out_base, float_fmt="{:.4f}", latex=True, index=False,
                mode="replace", source_files=None, script=None,
                cfg_hash=None):
    """Atomically write CSV + Markdown (+ LaTeX) representations.

    out_base: path without extension, e.g. results/paper_tables/table_main

    Table formats cannot be naturally merged, so supported modes are
    ``fail_if_exists`` and ``replace``.  When ``source_files`` is supplied,
    each format receives its own ``<full-filename>.sources.json`` sidecar.
    """
    mode = resolve_write_mode(mode)
    if mode == "merge":
        raise ValueError("merge mode is not supported for rendered tables")
    paths = table_artifact_paths(out_base, latex=latex)
    if source_files is not None and not script:
        raise ValueError("script is required when writing table sidecars")

    planned = list(paths)
    if source_files is not None:
        planned.extend(sources_sidecar_path(path) for path in paths)
    for path in planned:
        preflight_output(path, mode)

    # Render every representation before publishing any of them.  A render
    # failure therefore leaves all pre-existing artifacts untouched.
    rendered = {
        ".csv": df.to_csv(index=index),
        ".md": df_to_markdown(df, float_fmt=float_fmt),
    }
    if latex:
        rendered[".tex"] = df.to_latex(
            index=index, float_format=lambda value: float_fmt.format(value))

    for path in paths:
        write_text_atomic(rendered[path.suffix], path, mode=mode)
    if source_files is not None:
        for path in paths:
            write_sources_sidecar(path, source_files, script,
                                  cfg_hash=cfg_hash, mode=mode)
    return [str(path) for path in paths]
