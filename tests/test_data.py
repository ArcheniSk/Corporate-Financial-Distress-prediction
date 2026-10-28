import pandas as pd
import pytest

from fme.data import DatasetAuditError, load_panel
from fme.features import RobustFeatureTransform, build_sequences
from fme.labels import construct_horizon_labels


def test_constant_company_label_is_rejected(tmp_path):
    path = tmp_path / "panel.csv"
    pd.DataFrame({
        "company_id": ["a", "a", "b", "b"],
        "fiscal_year": [2000, 2001, 2000, 2001],
        "label": [1, 1, 0, 0],
        "x": [1.0, 2.0, 3.0, 4.0],
    }).to_csv(path, index=False)
    with pytest.raises(DatasetAuditError, match="constant within every company"):
        load_panel(path)


def test_valid_panel_audit_reports_transitions(tmp_path):
    path = tmp_path / "panel.csv"
    pd.DataFrame({
        "company_id": ["a", "a", "b", "b"],
        "fiscal_year": [2000, 2001, 2000, 2001],
        "label": [0, 1, 0, 0],
        "x": [1.0, 2.0, 3.0, 4.0],
    }).to_csv(path, index=False)
    panel, audit = load_panel(path)
    assert audit["firms_with_label_transitions"] == 1
    assert panel.feature_columns == ["x"]


def test_transform_uses_train_fit_parameters_only():
    train = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    test = pd.DataFrame({"x": [1000.0]})
    transform = RobustFeatureTransform(["x"]).fit(train)
    transformed = transform.transform(test)
    assert transformed.shape == (1, 1)
    assert float(transform.median[0]) == 1.0
    assert float(transformed[0, 0]) > 0.0


def test_sequences_are_causal_trailing_windows():
    frame = pd.DataFrame({
        "company_id": ["a", "a", "a"],
        "fiscal_year": [2000, 2001, 2002],
    })
    values = pd.DataFrame({"x": [1.0, 2.0, 3.0]}).to_numpy(dtype="float32")
    sequences, _, _ = build_sequences(frame, values, history_length=2)
    assert sequences[0, -1, 0] == 1.0
    assert sequences[1, :, 0].tolist() == [1.0, 2.0]
    assert sequences[2, :, 0].tolist() == [2.0, 3.0]


def test_horizon_labeling_drops_censored_rows_and_excludes_past_event():
    frame = pd.DataFrame({
        "company_id": ["a", "a", "b"],
        "available_date": ["2020-01-01", "2021-01-01", "2020-01-01"],
        "event_date": ["2020-06-01", "2020-06-01", None],
    })
    result = construct_horizon_labels(frame, observation_end_date="2022-12-31", horizon_days=365)
    assert result.company_id.tolist() == ["a", "b"]
    assert result.label.tolist() == [1, 0]
