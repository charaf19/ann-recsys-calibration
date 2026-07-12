"""Shared deterministic query construction and validated compact caching."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.preprocessing import filter_min_user_interactions
from utils.result_io import write_json_atomic, write_npz_atomic
from utils.splits import temporal_leave_one_out, build_eval_cases


QUERY_CACHE_VERSION = 1


@dataclass
class QueryPopulation:
    query_vectors: np.ndarray
    query_ids: np.ndarray
    train_indices: list
    test_indices: np.ndarray
    pop_counts: np.ndarray
    modality: str
    metadata: dict

    @property
    def exclusions(self):
        if self.modality == "u2i":
            return [set(map(int, hist)) for hist in self.train_indices]
        return [{int(hist[-1])} for hist in self.train_indices]

    @property
    def positives(self):
        return [{int(value)} for value in self.test_indices]


def load_interactions(path):
    df = pd.read_csv(path, usecols=["user_id", "item_id", "timestamp"])
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["timestamp"] = pd.to_numeric(
        df["timestamp"], errors="coerce").fillna(0).astype(np.int64)
    return df


def item_id_map(item_vecs_path, n_items):
    path = Path(item_vecs_path).with_name("item_ids.npy")
    if path.is_file():
        ids = np.load(path, allow_pickle=True)
        if len(ids) != n_items:
            raise ValueError(f"item id/vector length mismatch: {len(ids)} != {n_items}")
        return {str(value): i for i, value in enumerate(ids)}
    return {str(i): i for i in range(n_items)}


def build_query_vectors(modality, item_vecs, train_indices, test_indices):
    """Return the historical evaluator's exact Q/exclusion/positive semantics."""
    modality = str(modality).lower()
    q = np.empty((len(train_indices), item_vecs.shape[1]), dtype=np.float32)
    exclusions = []
    for i, hist in enumerate(train_indices):
        hist = np.asarray(hist, dtype=np.int32)
        if modality == "u2i":
            q[i] = item_vecs[hist].mean(axis=0)
            exclusions.append(set(map(int, hist)))
        elif modality == "i2i":
            anchor = int(hist[-1])
            q[i] = item_vecs[anchor]
            exclusions.append({anchor})
        else:
            raise ValueError(f"unknown modality {modality}")
    positives = [{int(value)} for value in test_indices]
    return np.ascontiguousarray(q), exclusions, positives


def _popularity(train_df, id2idx, n_items):
    pop = np.zeros(n_items, dtype=np.int64)
    codes = train_df["item_id"].map(lambda x: id2idx.get(str(x), -1)).to_numpy()
    codes = codes[codes >= 0].astype(np.int64)
    np.add.at(pop, codes, 1)
    return pop


def build_query_population(*, interactions_path, item_vecs, id2idx, modality,
                           max_queries, seed, min_user_interactions, metadata):
    interactions = load_interactions(interactions_path)
    filtered = filter_min_user_interactions(interactions, min_user_interactions)
    train_df, test_df = temporal_leave_one_out(filtered)
    users, histories, tests = build_eval_cases(
        train_df, test_df, id2idx, max_queries=max_queries, seed=seed)
    q, _, _ = build_query_vectors(modality, item_vecs, histories, tests)
    doc = dict(metadata)
    doc.update({"cache_version": QUERY_CACHE_VERSION, "modality": modality,
                "seed": int(seed), "min_user_interactions": int(min_user_interactions),
                "max_queries": str(max_queries), "n_queries": int(len(users)),
                "n_items": int(item_vecs.shape[0]), "dim": int(item_vecs.shape[1]),
                "split_protocol": "temporal_leave_one_out_v1"})
    return QueryPopulation(q, np.asarray(users, dtype=str),
                           [np.asarray(h, dtype=np.int32) for h in histories],
                           np.asarray(tests, dtype=np.int32),
                           _popularity(train_df, id2idx, item_vecs.shape[0]),
                           modality, doc)


def _pack_histories(histories):
    offsets = np.zeros(len(histories) + 1, dtype=np.int64)
    for i, hist in enumerate(histories):
        offsets[i + 1] = offsets[i] + len(hist)
    values = (np.concatenate(histories).astype(np.int32, copy=False)
              if histories else np.empty(0, dtype=np.int32))
    return values, offsets


def write_query_cache(population, path, mode="replace"):
    path = Path(path)
    values, offsets = _pack_histories(population.train_indices)
    write_npz_atomic(path, mode=mode, query_vectors=population.query_vectors,
                     query_ids=population.query_ids, train_indices=values,
                     train_offsets=offsets, test_indices=population.test_indices,
                     pop_counts=population.pop_counts)
    write_json_atomic(population.metadata, path.with_suffix(".meta.json"), mode=mode)
    return path


def load_query_cache(path, expected_metadata=None):
    path = Path(path)
    meta_path = path.with_suffix(".meta.json")
    if not path.is_file() or not meta_path.is_file():
        raise ValueError(f"query cache is incomplete: {path} and {meta_path} are required")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid query-cache metadata {meta_path}: {exc}") from exc
    if meta.get("cache_version") != QUERY_CACHE_VERSION:
        raise ValueError(f"incompatible query-cache version in {meta_path}")
    expected_metadata = expected_metadata or {}
    mismatches = {key: {"expected": value, "found": meta.get(key)}
                  for key, value in expected_metadata.items()
                  if meta.get(key) != value}
    if mismatches:
        raise ValueError(f"incompatible query cache {path}: {mismatches}")
    with np.load(path, allow_pickle=False) as z:
        required = {"query_vectors", "query_ids", "train_indices", "train_offsets",
                    "test_indices", "pop_counts"}
        missing = required - set(z.files)
        if missing:
            raise ValueError(f"query cache {path} lacks arrays {sorted(missing)}")
        offsets = np.asarray(z["train_offsets"], dtype=np.int64)
        values = np.asarray(z["train_indices"], dtype=np.int32)
        histories = [values[offsets[i]:offsets[i + 1]].copy()
                     for i in range(len(offsets) - 1)]
        population = QueryPopulation(
            np.ascontiguousarray(z["query_vectors"], dtype=np.float32),
            np.asarray(z["query_ids"]).astype(str), histories,
            np.asarray(z["test_indices"], dtype=np.int32),
            np.asarray(z["pop_counts"], dtype=np.int64), meta["modality"], meta)
    n = len(population.query_ids)
    if not (population.query_vectors.shape[0] == len(histories) ==
            len(population.test_indices) == n):
        raise ValueError(f"unaligned arrays in query cache {path}")
    return population
