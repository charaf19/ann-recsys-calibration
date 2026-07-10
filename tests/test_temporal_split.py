"""Temporal leave-one-out protocol: the held-out positive is the
chronologically LAST interaction; earlier interactions form the history."""
import numpy as np
import pandas as pd

from utils.splits import temporal_leave_one_out, build_eval_cases


def toy_interactions():
    return pd.DataFrame({
        "user_id": ["u1", "u1", "u1", "u2", "u2", "u3"],
        "item_id": ["i1", "i2", "i3", "i9", "i4", "i7"],
        "timestamp": [10, 20, 30, 200, 100, 5],
    })


def test_last_interaction_is_held_out():
    train, test = temporal_leave_one_out(toy_interactions())
    held = dict(zip(test["user_id"], test["item_id"]))
    assert held["u1"] == "i3"      # timestamp 30 is chronologically last
    assert held["u2"] == "i9"      # timestamp 200 beats 100 despite row order
    assert "u3" not in held        # single interaction: cannot be evaluated


def test_earlier_interactions_form_training_history():
    train, test = temporal_leave_one_out(toy_interactions())
    u1_train = train[train["user_id"] == "u1"]["item_id"].tolist()
    assert sorted(u1_train) == ["i1", "i2"]
    assert "i3" not in u1_train
    # single-interaction user stays entirely in train
    assert train[train["user_id"] == "u3"]["item_id"].tolist() == ["i7"]


def test_timestamp_ties_broken_by_row_order_deterministically():
    df = pd.DataFrame({
        "user_id": ["u", "u", "u"],
        "item_id": ["a", "b", "c"],
        "timestamp": [1, 1, 1],
    })
    _, test1 = temporal_leave_one_out(df)
    _, test2 = temporal_leave_one_out(df.copy())
    assert test1["item_id"].iloc[0] == "c"          # stable mergesort: last row
    assert test1["item_id"].iloc[0] == test2["item_id"].iloc[0]


def test_build_eval_cases_alignment_and_subsample():
    train, test = temporal_leave_one_out(toy_interactions())
    id2idx = {f"i{k}": k for k in range(1, 10)}
    users, hists, test_idx = build_eval_cases(train, test, id2idx,
                                              max_queries="full", seed=42)
    by_user = dict(zip(users, test_idx))
    assert by_user["u1"] == 3 and by_user["u2"] == 9
    hist_u1 = hists[users.index("u1")]
    assert list(hist_u1) == [1, 2]                  # chronological history
    assert isinstance(test_idx, np.ndarray)
