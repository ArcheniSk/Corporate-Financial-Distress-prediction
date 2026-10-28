"""Reliability-aware temporal peer graph model implemented with PyTorch."""

from __future__ import annotations

import torch
from torch import nn

from .graphs import SnapshotGraph


class WeightedGraphAttention(nn.Module):
    """Edge-weighted, target-normalized attention for one graph relation."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.source_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.target_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.source_score = nn.Linear(hidden_dim, 1, bias=False)
        self.target_score = nn.Linear(hidden_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, graph: SnapshotGraph, relation: str) -> torch.Tensor:
        edge = graph.relations.get(relation)
        aggregate = torch.zeros_like(values)
        if edge is None or edge.source.size == 0:
            return aggregate
        source = torch.as_tensor(edge.source, dtype=torch.long, device=values.device)
        target = torch.as_tensor(edge.target, dtype=torch.long, device=values.device)
        weight = torch.as_tensor(edge.weight, dtype=values.dtype, device=values.device).clamp_min(1e-8)
        projected_source = self.source_projection(values)
        projected_target = self.target_projection(values)
        logits = (
            self.source_score(projected_source[source]).squeeze(-1)
            + self.target_score(projected_target[target]).squeeze(-1)
            + torch.log(weight)
        )
        logits = torch.nn.functional.leaky_relu(logits, negative_slope=0.2)
        max_per_target = torch.full((values.shape[0],), -torch.inf, dtype=values.dtype, device=values.device)
        max_per_target.scatter_reduce_(0, target, logits, reduce="amax", include_self=True)
        attention = torch.exp(logits - max_per_target[target])
        denominator = torch.zeros((values.shape[0],), dtype=values.dtype, device=values.device)
        denominator.index_add_(0, target, attention)
        attention = attention / denominator[target].clamp_min(1e-8)
        aggregate.index_add_(0, target, self.dropout(projected_source[source]) * attention.unsqueeze(-1))
        return aggregate


class ReliabilityTemporalGNN(nn.Module):
    """LSTM encoder with relation attention and reliability-weighted graph attention."""

    def __init__(self, input_dim: int, hidden_dim: int, relations: list[str], dropout: float = 0.2):
        super().__init__()
        self.relations = relations
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.self_projection = nn.Linear(hidden_dim, hidden_dim)
        self.relation_attention = nn.ModuleDict({name: WeightedGraphAttention(hidden_dim, dropout) for name in relations})
        self.relation_gate = nn.Linear(hidden_dim, max(len(relations), 1))
        self.message_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))
        self.dropout = nn.Dropout(dropout)

    def encode(self, sequences: torch.Tensor, graph: SnapshotGraph) -> tuple[torch.Tensor, torch.Tensor]:
        _, (hidden, _) = self.encoder(sequences)
        temporal = hidden[-1]
        self_state = self.self_projection(temporal)
        if not self.relations:
            combined = self.fusion_norm(self_state)
        else:
            gates = torch.softmax(self.relation_gate(temporal), dim=-1)
            messages = torch.zeros_like(temporal)
            for relation_idx, relation in enumerate(self.relations):
                aggregate = self.relation_attention[relation](temporal, graph, relation)
                messages = messages + gates[:, relation_idx : relation_idx + 1] * aggregate
            fusion = torch.sigmoid(self.message_gate(torch.cat([self_state, messages], dim=-1)))
            combined = self.fusion_norm(self_state + fusion * self.dropout(messages))
        return combined, temporal

    def forward(self, sequences: torch.Tensor, graph: SnapshotGraph) -> torch.Tensor:
        combined, _ = self.encode(sequences, graph)
        return self.classifier(combined).squeeze(-1)


class PriorGuidedReliabilityTemporalGNN(ReliabilityTemporalGNN):
    """Gated LSTM-GNN residual correction over a pretrained tabular logit prior."""

    def __init__(self, input_dim: int, hidden_dim: int, relations: list[str], dropout: float = 0.2, adaptive_gate: bool = True, max_correction: float = 0.25):
        super().__init__(input_dim, hidden_dim, relations, dropout=dropout)
        self.adaptive_gate = adaptive_gate
        self.max_correction = max_correction
        correction_dim = hidden_dim * 2 + 2
        self.prior_residual = nn.Sequential(
            nn.Linear(correction_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.prior_gate = nn.Sequential(
            nn.Linear(correction_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.zeros_(self.prior_residual[-1].weight)
        nn.init.zeros_(self.prior_residual[-1].bias)
        nn.init.zeros_(self.prior_gate[-1].weight)
        nn.init.constant_(self.prior_gate[-1].bias, -2.0)
        self.global_amplitude_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(self, sequences: torch.Tensor, graph: SnapshotGraph, prior_logits: torch.Tensor | None = None) -> torch.Tensor:
        if prior_logits is None:
            raise ValueError("prior_logits are required for prior-guided LSTM-GNN inference.")
        combined, temporal = self.encode(sequences, graph)
        prior = prior_logits.reshape(-1, 1)
        prior_probability = torch.sigmoid(prior)
        uncertainty = 4.0 * prior_probability * (1.0 - prior_probability)
        correction_input = torch.cat([combined, temporal, prior, uncertainty], dim=-1)
        residual = self.prior_residual(correction_input)
        if self.adaptive_gate:
            gate = self.max_correction * torch.sigmoid(self.prior_gate(correction_input))
        else:
            gate = self.max_correction * torch.sigmoid(self.global_amplitude_logit).expand_as(residual)
        return prior.squeeze(-1) + (gate * residual).squeeze(-1)


class PriorGuidedTabularResidual(nn.Module):
    """Missingness-aware MLP residual comparator over a cross-fitted tree prior."""

    def __init__(self, input_dim: int, dropout: float = 0.2, adaptive_gate: bool = True, max_correction: float = 0.25):
        super().__init__()
        self.adaptive_gate = adaptive_gate
        self.max_correction = max_correction
        correction_dim = input_dim + 2
        self.residual = nn.Sequential(
            nn.Linear(correction_dim, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.amplitude = nn.Sequential(
            nn.Linear(correction_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        nn.init.zeros_(self.amplitude[-1].weight)
        nn.init.constant_(self.amplitude[-1].bias, -2.0)
        self.global_amplitude_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(self, sequences: torch.Tensor, graph: SnapshotGraph | None = None, prior_logits: torch.Tensor | None = None) -> torch.Tensor:
        if prior_logits is None:
            raise ValueError("prior_logits are required for prior-guided tabular residual inference.")
        values = sequences[:, -1, :] if sequences.ndim == 3 else sequences
        prior = prior_logits.reshape(-1, 1)
        prior_probability = torch.sigmoid(prior)
        uncertainty = 4.0 * prior_probability * (1.0 - prior_probability)
        correction_input = torch.cat([values, prior, uncertainty], dim=-1)
        residual = self.residual(correction_input)
        if self.adaptive_gate:
            amplitude = self.max_correction * torch.sigmoid(self.amplitude(correction_input))
        else:
            amplitude = self.max_correction * torch.sigmoid(self.global_amplitude_logit).expand_as(residual)
        return prior.squeeze(-1) + (amplitude * residual).squeeze(-1)


class TabularMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values).squeeze(-1)
