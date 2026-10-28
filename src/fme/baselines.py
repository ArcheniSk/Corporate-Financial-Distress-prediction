"""Strong tabular and conventional temporal-graph baselines."""

from __future__ import annotations

import torch
from torch import nn

from .graphs import SnapshotGraph


class TemporalOnly(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))

    def forward(self, sequences: torch.Tensor, graph: SnapshotGraph | None = None) -> torch.Tensor:
        _, (hidden, _) = self.encoder(sequences)
        return self.head(hidden[-1]).squeeze(-1)


class TemporalGCN(nn.Module):
    """Single-layer GCN/GraphSAGE-style comparator over merged relation edges."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, mode: str = "gcn", dropout: float = 0.2):
        super().__init__()
        if mode not in {"gcn", "sage"}:
            raise ValueError("mode must be gcn or sage")
        self.mode = mode
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.update = nn.Linear(hidden_dim * (2 if mode == "sage" else 1), hidden_dim)
        self.head = nn.Sequential(nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))

    def forward(self, sequences: torch.Tensor, graph: SnapshotGraph) -> torch.Tensor:
        _, (hidden, _) = self.encoder(sequences)
        x = hidden[-1]
        aggregate = torch.zeros_like(x)
        degree = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
        for relation_graph in graph.relations.values():
            if not relation_graph.source.size:
                continue
            source = torch.as_tensor(relation_graph.source, dtype=torch.long, device=x.device)
            target = torch.as_tensor(relation_graph.target, dtype=torch.long, device=x.device)
            weight = torch.as_tensor(relation_graph.weight, dtype=x.dtype, device=x.device).unsqueeze(1)
            aggregate.index_add_(0, target, x[source] * weight)
            degree.index_add_(0, target, weight)
        aggregate = aggregate / degree.clamp_min(1e-8)
        joined = torch.cat([x, aggregate], dim=-1) if self.mode == "sage" else x + aggregate
        return self.head(self.update(joined)).squeeze(-1)
