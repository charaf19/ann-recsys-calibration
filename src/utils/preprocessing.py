"""Canonical interaction-population preprocessing shared by every stage.

The benchmark evaluates a *k-core* population: only users with at least
``data.min_user_interactions`` interactions are kept. This single filter is
applied — from one place — BEFORE embedding training, temporal
leave-one-out splitting, evaluation-case construction, popularity
computation, and dataset-statistics generation. Centralizing it guarantees
the reported dataset-statistics table describes exactly the population the
experiments train and evaluate on (see Phase 5 protocol audit).

The value is always resolved from configuration (``data.min_user_interactions``
in configs/defaults.yml), never a scattered constant, so changing it changes
the config hash and correctly invalidates artifacts built under a different
population.
"""
import pandas as pd

# Canonical default; the resolved config is authoritative. Kept here only so
# standalone tools (e.g. dataset_stats) share one number instead of hardcoding.
DEFAULT_MIN_USER_INTERACTIONS = 5


def filter_min_user_interactions(df: pd.DataFrame,
                                 min_user_interactions: int) -> pd.DataFrame:
    """Keep only interactions of users with >= ``min_user_interactions`` rows.

    Deterministic: membership depends solely on per-user interaction counts,
    not on row order. ``min_user_interactions <= 1`` is a no-op (returns an
    index-reset copy). Requires a ``user_id`` column.
    """
    m = int(min_user_interactions)
    if "user_id" not in df.columns:
        raise ValueError("filter_min_user_interactions requires a 'user_id' column")
    if m <= 1:
        return df.reset_index(drop=True)
    counts = df.groupby("user_id")["user_id"].transform("size")
    return df[counts >= m].reset_index(drop=True)
