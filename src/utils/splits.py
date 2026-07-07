"""Deterministic temporal splits for implicit-feedback interactions.

The canonical protocol used across the benchmark is the *temporal
leave-one-out* split: for every user with >= 2 interactions, the
chronologically last interaction is held out as the test positive and the
remainder forms the training history. Users with a single interaction stay
entirely in train (they cannot be evaluated).

Ties in timestamps are broken by original row order via a stable mergesort,
so the split is fully deterministic for a given input CSV.
"""
import numpy as np
import pandas as pd


def temporal_leave_one_out(df: pd.DataFrame):
    """Split interactions into (train_df, test_df).

    Args:
        df: DataFrame with columns user_id, item_id, timestamp.

    Returns:
        (train_df, test_df) — test_df has exactly one row per eligible user
        (the chronologically last interaction).
    """
    df = df.copy()
    if "timestamp" not in df.columns:
        df["timestamp"] = 0
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype(np.int64)
    df = df.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(drop=True)

    grp = df.groupby("user_id", sort=False)
    sizes = grp["item_id"].transform("size")
    # cumcount is 0-based position within the user's chronological history
    pos = grp.cumcount()
    is_last = pos == (sizes - 1)
    eligible = sizes >= 2

    test_df = df[is_last & eligible].reset_index(drop=True)
    train_df = df[~(is_last & eligible)].reset_index(drop=True)
    return train_df, test_df


def build_eval_cases(train_df: pd.DataFrame, test_df: pd.DataFrame, id2idx: dict,
                     max_queries=None, seed: int = 42):
    """Turn a temporal LOO split into evaluation cases.

    For each test user, returns:
      - user_id
      - train_item_idx: np.int32 array of the user's training items (index space)
      - test_item_idx:  int index of the held-out item

    Users whose held-out item or entire training history is missing from the
    embedding vocabulary are skipped. If max_queries is an int smaller than
    the number of eligible users, a deterministic subsample (seeded) is taken.
    max_queries=None or the string "full" keeps all users.
    """
    train_groups = {}
    for uid, g in train_df.groupby("user_id", sort=False):
        train_groups[uid] = g["item_id"].tolist()

    users, train_idx_list, test_idx_list = [], [], []
    for row in test_df.itertuples(index=False):
        uid = row.user_id
        t = id2idx.get(str(row.item_id))
        if t is None:
            continue
        hist = train_groups.get(uid, [])
        hist_idx = [id2idx.get(str(x)) for x in hist]
        hist_idx = np.array([int(h) for h in hist_idx if h is not None], dtype=np.int32)
        if hist_idx.size == 0:
            continue
        users.append(uid)
        train_idx_list.append(hist_idx)
        test_idx_list.append(int(t))

    if not users:
        raise ValueError("No valid evaluation cases could be built (id mapping mismatch?).")

    n = len(users)
    if max_queries not in (None, "full") and int(max_queries) < n:
        rng = np.random.default_rng(seed)
        sel = np.sort(rng.choice(n, size=int(max_queries), replace=False))
        users = [users[i] for i in sel]
        train_idx_list = [train_idx_list[i] for i in sel]
        test_idx_list = [test_idx_list[i] for i in sel]

    return users, train_idx_list, np.array(test_idx_list, dtype=np.int32)
