"""Point-in-time financial peer graphs with snapshot-persistence weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


@dataclass
class RelationGraph:
    source: np.ndarray
    target: np.ndarray
    weight: np.ndarray


@dataclass
class SnapshotGraph:
    relations: dict[str, RelationGraph]
    node_ids: list[str]


def _knn_edges(
    values: np.ndarray,
    local_ids: list[str],
    global_index: dict[str, int],
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, set[str]]]:
    if len(local_ids) < 2:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty.copy(), np.zeros(0, dtype=np.float32), {}
    n_neighbors = min(k + 1, len(local_ids))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(values)
    distances, indices = nn.kneighbors(values)
    src: list[int] = []
    dst: list[int] = []
    weights: list[float] = []
    sets: dict[str, set[str]] = {}
    for local_dst, (dists, neighbors) in enumerate(zip(distances, indices)):
        dst_id = local_ids[local_dst]
        peers: set[str] = set()
        for distance, local_src in zip(dists, neighbors):
            if local_src == local_dst:
                continue
            src_id = local_ids[int(local_src)]
            if src_id == dst_id:
                continue
            sim = float(np.clip(1.0 - distance, 0.01, 1.0))
            src.append(global_index[src_id])
            dst.append(global_index[dst_id])
            weights.append(sim)
            peers.add(src_id)
        sets[dst_id] = peers
    return np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64), np.asarray(weights, dtype=np.float32), sets


def _industry_edges(
    frame: pd.DataFrame,
    relation_column: str,
    global_index: dict[str, int],
    k: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, set[str]]]:
    rng = np.random.default_rng(seed)
    src: list[int] = []
    dst: list[int] = []
    sets: dict[str, set[str]] = {}
    grouped = frame.groupby(relation_column, dropna=False, sort=False)
    for _, group in grouped:
        ids = sorted(group.company_id.astype(str).unique())
        for dst_id in ids:
            candidates = [x for x in ids if x != dst_id]
            if len(candidates) > k:
                candidates = list(rng.choice(candidates, size=k, replace=False))
            sets[dst_id] = set(candidates)
            for src_id in candidates:
                src.append(global_index[src_id])
                dst.append(global_index[dst_id])
    return np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64), np.ones(len(src), dtype=np.float32), sets


def build_snapshot_graphs(
    frame: pd.DataFrame,
    scaled_features: np.ndarray,
    feature_groups: dict[str, list[str]],
    feature_columns: list[str],
    *,
    k: int = 10,
    seed: int = 42,
    persistence_weighting: bool = True,
) -> tuple[dict[int, SnapshotGraph], list[str]]:
    """Build graph snapshots using same-year observed covariates only."""
    if len(frame) != len(scaled_features):
        raise ValueError("Feature matrix row count must match frame row count.")
    ordered = frame.reset_index(drop=True)
    companies = sorted(ordered.company_id.astype(str).unique())
    global_index = {firm: idx for idx, firm in enumerate(companies)}
    column_positions = {name: idx for idx, name in enumerate(feature_columns)}
    previous: dict[str, dict[str, set[str]]] = {}
    snapshots: dict[int, SnapshotGraph] = {}
    category_cols = [c for c in ordered.columns if c.lower() in {"industry", "division", "majorgroup", "sic", "siccode", "sic_code"}]

    for year in sorted(ordered.fiscal_year.astype(int).unique()):
        row_ids = np.flatnonzero(ordered.fiscal_year.astype(int).to_numpy() == year)
        year_frame = ordered.iloc[row_ids].copy()
        relations: dict[str, RelationGraph] = {}
        current_sets: dict[str, dict[str, set[str]]] = {}
        local_ids = year_frame.company_id.astype(str).tolist()
        for relation, columns in feature_groups.items():
            cols = [column_positions[c] for c in columns]
            values = scaled_features[row_ids][:, cols]
            src, dst, weights, peers = _knn_edges(values, local_ids, global_index, k)
            relations[relation] = RelationGraph(src, dst, weights)
            current_sets[relation] = peers
        for cat in category_cols:
            relation = f"industry_{cat.lower()}"
            src, dst, weights, peers = _industry_edges(year_frame, cat, global_index, k, seed + year)
            relations[relation] = RelationGraph(src, dst, weights)
            current_sets[relation] = peers

        for relation, graph in relations.items():
            old_sets = previous.get(relation, {})
            peer_sets = current_sets.get(relation, {})
            adjusted = graph.weight.copy()
            if persistence_weighting and len(adjusted):
                by_target: dict[int, list[int]] = {}
                for edge_idx, target_idx in enumerate(graph.target):
                    by_target.setdefault(int(target_idx), []).append(edge_idx)
                for target_idx, edge_indices in by_target.items():
                    target_id = companies[target_idx]
                    old = old_sets.get(target_id, set())
                    current = peer_sets.get(target_id, set())
                    union = old | current
                    persistence = len(old & current) / len(union) if union else 0.0
                    # New or unstable links are retained, but carry lower confidence.
                    factor = 0.5 + 0.5 * persistence
                    adjusted[edge_indices] *= factor
            relations[relation] = RelationGraph(graph.source, graph.target, adjusted)
        snapshots[int(year)] = SnapshotGraph(relations, companies)
        previous = current_sets
    return snapshots, companies
