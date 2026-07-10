"""Shared configuration loader for every canonical pipeline script.

One loader, one precedence order, everywhere:

    explicit CLI argument
        > experiment-specific YAML (e.g. configs/main_cpu.yml)
        > inherited configs/defaults.yml (via `inherits:`)
        > emergency code fallback (the caller's documented default)

Features:
    - YAML loading with clear errors (missing file, unparsable YAML)
    - single-level or chained `inherits: <relative path>` resolution
    - recursive deep merge (dicts merge key-wise; scalars/lists replace)
    - typed access via cfg_get(cfg, "a.b.c", type=int, required=True)
    - deterministic serialization + short hash for provenance

Scripts must not implement their own load_config(); they call
load_config(path, cli_overrides=...) and read resolved values.
"""
import copy
import hashlib
import json
from pathlib import Path

import yaml


REQUIRED_CORE_SECTIONS = (
    "project",
    "reproducibility",
    "embedding",
    "retrieval",
    "index",
    "calibration",
    "statistics",
)

REQUIRED_CORE_KEYS = (
    "project.name",
    "reproducibility.seed",
    "reproducibility.omp_threads",
    "embedding.method",
    "embedding.dim",
    "embedding.weighting",
    "embedding.normalize",
    "embedding.bm25_k1",
    "embedding.bm25_b",
    "retrieval.methods",
    "retrieval.modalities",
    "retrieval.topk",
    "retrieval.metric_topk",
    "retrieval.budget_mb",
    "index.hnsw.M",
    "index.hnsw.ef_construction",
    "index.ivf.nlist",
    "index.pq.m",
    "index.pq.bits",
    "index.ivfpq.use_opq",
    "calibration.targets",
    "calibration.primary_target",
    "calibration.queries",
    "statistics.bootstrap_iterations",
)


class ConfigError(ValueError):
    """Configuration file missing, unparsable, or failing validation."""


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        raise ConfigError(
            f"config file not found: {path} "
            f"(expected a YAML file; see configs/ for the canonical set)")
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"config file {path} is not valid YAML: {e}") from e
    if doc is None:
        return {}
    if not isinstance(doc, dict):
        raise ConfigError(f"config file {path} must contain a YAML mapping, "
                          f"got {type(doc).__name__}")
    return doc


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base` (override wins).

    Dicts merge key-wise; every other type (scalars, lists) is replaced
    wholesale so experiment configs can shrink or reorder lists.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path, cli_overrides: dict | None = None,
                validate: bool = True) -> dict:
    """Load a YAML config, resolving `inherits:` chains and CLI overrides.

    `inherits` is resolved relative to the config file's own directory.
    `cli_overrides` maps dotted keys to values; None values are ignored so
    argparse defaults of None never mask YAML values.
    """
    path = Path(path).expanduser().resolve()
    seen = []
    chain = []
    current = path
    while True:
        if current in seen:
            raise ConfigError(f"circular `inherits` chain at {current}: {seen}")
        seen.append(current)
        doc = _read_yaml(current)
        chain.append(doc)
        parent = doc.pop("inherits", None)
        if parent is None:
            break
        if not isinstance(parent, str) or not parent.strip():
            raise ConfigError(
                f"config key `inherits` in {current} must be a non-empty "
                f"relative path string, got {parent!r}")
        current = (current.parent / parent).resolve()

    resolved: dict = {}
    for doc in reversed(chain):  # base first, most specific last
        resolved = deep_merge(resolved, doc)

    if cli_overrides:
        resolved = apply_cli_overrides(resolved, cli_overrides)
    if validate:
        validate_config(resolved, source=path)
    return resolved


def apply_cli_overrides(cfg: dict, overrides: dict) -> dict:
    """Apply {dotted.key: value} overrides; None values are skipped."""
    out = copy.deepcopy(cfg)
    for dotted, value in overrides.items():
        if value is None:
            continue
        if not str(dotted).strip() or any(not p for p in str(dotted).split(".")):
            raise ConfigError(f"invalid empty CLI override path: {dotted!r}")
        node = out
        parts = str(dotted).split(".")
        for p in parts[:-1]:
            nxt = node.get(p)
            if not isinstance(nxt, dict):
                nxt = {}
                node[p] = nxt
            node = nxt
        node[parts[-1]] = value
    return out


def validate_config(cfg: dict, source=None) -> dict:
    """Validate the shared scientific configuration contract.

    Experiment-specific sections (``datasets``, ``evaluation``,
    ``embedding_sensitivity`` and ``scale_stress``) are validated by their
    consumers because they are intentionally absent from ``defaults.yml``.
    The core sections and keys, however, must resolve for every canonical
    experiment configuration.
    """
    label = f" in {source}" if source else ""
    if not isinstance(cfg, dict):
        raise ConfigError(f"resolved configuration{label} must be a mapping")
    missing_sections = [s for s in REQUIRED_CORE_SECTIONS
                        if not isinstance(cfg.get(s), dict)]
    if missing_sections:
        raise ConfigError(
            f"resolved configuration{label} is missing required mapping "
            f"section(s): {', '.join(missing_sections)}")
    missing_keys = []
    for dotted in REQUIRED_CORE_KEYS:
        node = cfg
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                missing_keys.append(dotted)
                break
            node = node[part]
    if missing_keys:
        raise ConfigError(
            f"resolved configuration{label} is missing required key(s): "
            f"{', '.join(missing_keys)}")
    return cfg


_MISSING = object()


def cfg_get(cfg: dict, dotted: str, default=_MISSING, type=None, required=False):
    """Typed access: cfg_get(cfg, "statistics.bootstrap_iterations", type=int).

    Raises ConfigError with the full dotted path when a required key is
    missing or a value cannot be coerced to the requested type.
    """
    node = cfg
    for p in dotted.split("."):
        if not isinstance(node, dict) or p not in node:
            if required or default is _MISSING:
                raise ConfigError(f"missing required config key: {dotted}")
            return default
        node = node[p]
    if type is not None and node is not None:
        try:
            if type is bool and isinstance(node, str):
                token = node.strip().lower()
                if token in ("1", "true", "yes", "y", "on"):
                    node = True
                elif token in ("0", "false", "no", "n", "off"):
                    node = False
                else:
                    raise ValueError(
                        "expected a boolean token (true/false, yes/no, 1/0)")
            else:
                node = type(node)
        except (TypeError, ValueError) as e:
            raise ConfigError(
                f"config key {dotted} has value {node!r}, "
                f"not coercible to {type.__name__}") from e
    return node


def resolved_config_json(cfg: dict) -> str:
    """Deterministic serialization of a resolved config (sorted keys)."""
    return json.dumps(cfg, indent=2, sort_keys=True, default=str)


def config_hash(cfg: dict) -> str:
    """Short stable hash of the resolved configuration (12 hex chars)."""
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
