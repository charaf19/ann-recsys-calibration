"""Canonical k-core interaction filter (utils.preprocessing)."""
import pandas as pd
import pytest

from utils.preprocessing import (filter_min_user_interactions,
                                 DEFAULT_MIN_USER_INTERACTIONS)


def _df():
    rows = []
    # u1: 5, u2: 2, u3: 1, u4: 6
    for u, n in [("u1", 5), ("u2", 2), ("u3", 1), ("u4", 6)]:
        for i in range(n):
            rows.append({"user_id": u, "item_id": f"{u}i{i}", "timestamp": i})
    return pd.DataFrame(rows)


def test_default_is_five():
    assert DEFAULT_MIN_USER_INTERACTIONS == 5


def test_filter_keeps_only_qualifying_users():
    out = filter_min_user_interactions(_df(), 5)
    assert set(out["user_id"]) == {"u1", "u4"}
    assert len(out) == 11


def test_filter_min_two():
    out = filter_min_user_interactions(_df(), 2)
    assert set(out["user_id"]) == {"u1", "u2", "u4"}  # u3 (1) dropped


def test_filter_is_noop_below_two():
    df = _df()
    for m in (1, 0, -3):
        out = filter_min_user_interactions(df, m)
        assert len(out) == len(df)
        assert set(out["user_id"]) == set(df["user_id"])


def test_filter_is_deterministic_and_order_independent():
    df = _df()
    shuffled = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    a = filter_min_user_interactions(df, 5).sort_values(
        ["user_id", "item_id"]).reset_index(drop=True)
    b = filter_min_user_interactions(shuffled, 5).sort_values(
        ["user_id", "item_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_filter_requires_user_id_column():
    with pytest.raises(ValueError, match="user_id"):
        filter_min_user_interactions(pd.DataFrame({"item_id": [1, 2]}), 5)
