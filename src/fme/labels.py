"""Point-in-time target construction with right-censoring."""

from __future__ import annotations

import pandas as pd

from .data import DatasetAuditError


def construct_horizon_labels(
    panel: pd.DataFrame,
    *,
    observation_end_date: str,
    horizon_days: int = 365,
    available_column: str = "available_date",
    event_column: str = "event_date",
) -> pd.DataFrame:
    """Label future events after predictor availability; drop already-failed/censored rows."""
    if available_column not in panel or event_column not in panel:
        raise DatasetAuditError(f"Event-label construction requires {available_column} and {event_column}.")
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    result = panel.copy()
    result[available_column] = pd.to_datetime(result[available_column], errors="coerce", utc=True)
    result[event_column] = pd.to_datetime(result[event_column], errors="coerce", utc=True)
    cutoff = pd.to_datetime(observation_end_date, utc=True)
    if result[available_column].isna().any():
        raise DatasetAuditError("available_date contains missing or invalid values.")
    horizon_end = result[available_column] + pd.to_timedelta(horizon_days, unit="D")
    event = result[event_column]
    prior_event = event.notna() & (event <= result[available_column])
    fully_observed = horizon_end <= cutoff
    result["label"] = ((event > result[available_column]) & (event <= horizon_end)).astype("int8")
    keep = ~prior_event & fully_observed
    result = result.loc[keep].copy().reset_index(drop=True)
    if result.empty:
        raise DatasetAuditError("No rows remain after excluding prior events and right-censored horizons.")
    result.attrs["label_horizon_days"] = int(horizon_days)
    result.attrs["observation_end_date"] = cutoff.isoformat()
    result.attrs["censored_rows_removed"] = int((~fully_observed).sum())
    result.attrs["prior_event_rows_removed"] = int(prior_event.sum())
    return result
