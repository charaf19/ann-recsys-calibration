"""Reporting helpers: standardized logging banners and table writers.

Every pipeline script uses `banner()` so logs follow the required format:

    [script_name] starting...
    [script_name] input path: ...
    [script_name] output path: ...
    [script_name] completed.
"""
import json
from pathlib import Path


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


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)


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


def write_table(df, out_base, float_fmt="{:.4f}", latex=True, index=False):
    """Write a DataFrame as CSV + Markdown (+ LaTeX) next to each other.

    out_base: path without extension, e.g. results/paper_tables/table_main
    """
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_base.with_suffix(".csv"), index=index)
    with open(out_base.with_suffix(".md"), "w", encoding="utf-8") as f:
        f.write(df_to_markdown(df, float_fmt=float_fmt))
    if latex:
        try:
            tex = df.to_latex(index=index, float_format=lambda v: float_fmt.format(v))
        except Exception:
            tex = df.to_string(index=index)
        with open(out_base.with_suffix(".tex"), "w", encoding="utf-8") as f:
            f.write(tex)
    return [str(out_base.with_suffix(ext)) for ext in (".csv", ".md") + ((".tex",) if latex else ())]
