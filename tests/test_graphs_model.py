import numpy as np
import pandas as pd
import torch

from fme.graphs import build_snapshot_graphs
from fme.model import ReliabilityTemporalGNN


def test_snapshot_graph_has_bounded_peer_count_and_industry_relation():
    frame = pd.DataFrame({
        "company_id": ["a", "b", "c", "a", "b", "c"],
        "fiscal_year": [2000] * 3 + [2001] * 3,
        "industry": ["x", "x", "y"] * 2,
    })
    features = np.asarray([[0.0], [0.1], [4.0], [0.0], [0.2], [3.8]], dtype=np.float32)
    snapshots, companies = build_snapshot_graphs(frame, features, {"financial_similarity": ["f"]}, ["f"], k=1)
    assert companies == ["a", "b", "c"]
    assert "industry_industry" in snapshots[2000].relations
    assert len(snapshots[2000].relations["financial_similarity"].source) <= 3
    assert np.all(snapshots[2001].relations["financial_similarity"].weight <= 1.0)


def test_model_returns_one_logit_per_firm():
    frame = pd.DataFrame({"company_id": ["a", "b", "c"], "fiscal_year": [2000] * 3})
    features = np.asarray([[0.0, 1.0], [0.2, 0.9], [2.0, -1.0]], dtype=np.float32)
    graphs, companies = build_snapshot_graphs(frame, features, {"financial_similarity": ["x", "y"]}, ["x", "y"], k=1)
    model = ReliabilityTemporalGNN(input_dim=2, hidden_dim=8, relations=list(graphs[2000].relations))
    output = model(torch.zeros((len(companies), 3, 2)), graphs[2000])
    assert output.shape == (3,)
    assert torch.isfinite(output).all()
