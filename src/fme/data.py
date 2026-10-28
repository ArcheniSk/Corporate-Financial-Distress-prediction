"""Data loading, target audits, and deterministic synthetic smoke data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ID_CANDIDATES = ("company_id", "company_name", "firm_id", "CIK")
YEAR_CANDIDATES = ("fiscal_year", "fyear", "year")
LABEL_CANDIDATES = ("label", "status_label", "financial_distress_label")


class DatasetAuditError(ValueError):
    """Raised when a panel cannot support the declared forecasting task."""


@dataclass
class Panel:
    frame: pd.DataFrame
    company_column: str
    year_column: str
    label_column: str
    feature_columns: list[str]
    categorical_columns: list[str]


def _first_present(frame: pd.DataFrame, candidates: tuple[str, ...], kind: str) -> str:
    for name in candidates:
        if name in frame.columns:
            return name
    raise DatasetAuditError(f"Could not identify {kind}; expected one of {candidates}.")


def load_panel(path: str | Path, *, require_temporal_labels: bool = True) -> tuple[Panel, dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise DatasetAuditError("Dataset contains no rows.")

    company = _first_present(frame, ID_CANDIDATES, "company identifier")
    year = _first_present(frame, YEAR_CANDIDATES, "fiscal year")
    label = _first_present(frame, LABEL_CANDIDATES, "label")
    frame = frame.rename(columns={company: "company_id", year: "fiscal_year", label: "label"})
    company, year, label = "company_id", "fiscal_year", "label"
    frame["fiscal_year"] = pd.to_numeric(frame[year], errors="coerce")
    if frame["fiscal_year"].isna().any():
        raise DatasetAuditError("Fiscal year contains missing or non-numeric values.")

    raw_labels = frame[label]
    if raw_labels.dtype == object:
        normalized = raw_labels.astype(str).str.strip().str.lower()
        mapping = {"alive": 0, "failed": 1, "healthy": 0, "distressed": 1, "false": 0, "true": 1}
        if normalized.isin(mapping).all():
            frame["label"] = normalized.map(mapping).astype(int)
        else:
            numeric = pd.to_numeric(raw_labels, errors="coerce")
            if numeric.isna().any():
                raise DatasetAuditError(f"Label values are not a recognized binary encoding: {sorted(normalized.unique())[:10]}")
            frame["label"] = numeric
    else:
        frame["label"] = pd.to_numeric(raw_labels, errors="coerce")
    if frame["label"].isna().any() or not set(frame["label"].unique()).issubset({0, 1}):
        raise DatasetAuditError("Labels must be binary values encoded as 0/1 or alive/failed.")
    frame["label"] = frame["label"].astype(int)
    frame = frame.sort_values([company, year], kind="stable").reset_index(drop=True)

    duplicate_count = int(frame.duplicated([company, year]).sum())
    if duplicate_count:
        raise DatasetAuditError(f"Found {duplicate_count} duplicate company-year rows.")
    transition_counts = frame.groupby(company, sort=False)["label"].nunique()
    transition_firms = int((transition_counts > 1).sum())
    failure_firms = int(frame.loc[frame.label.eq(1), company].nunique())
    id_like = {company, "fiscal_year", "label", "available_date", "event_date", "filing_date"}
    category_columns = [c for c in frame.columns if c.lower() in {"industry", "division", "majorgroup", "sic", "siccode", "sic_code"}]
    numeric_features = [c for c in frame.select_dtypes(include=[np.number]).columns if c not in id_like]
    numeric_features = [c for c in numeric_features if c not in category_columns]
    if not numeric_features:
        raise DatasetAuditError("No numeric predictor columns remain after excluding identifiers and labels.")

    audit = {
        "path": str(path),
        "rows": int(len(frame)),
        "companies": int(frame[company].nunique()),
        "year_min": int(frame[year].min()),
        "year_max": int(frame[year].max()),
        "feature_count": len(numeric_features),
        "features": numeric_features,
        "categorical_relations": category_columns,
        "label_counts": {str(int(k)): int(v) for k, v in frame.label.value_counts().sort_index().items()},
        "firms_with_positive_label": failure_firms,
        "firms_with_label_transitions": transition_firms,
        "duplicate_company_years": duplicate_count,
        "missing_cells": int(frame.isna().sum().sum()),
        "constant_within_company": transition_firms == 0,
    }
    if require_temporal_labels and transition_firms == 0:
        raise DatasetAuditError(
            "Target audit failed: labels are constant within every company. "
            "This does not establish the declared future-event horizon; recover event dates "
            "and construct point-in-time labels before training. Audit: "
            f"{audit}"
        )
    return Panel(frame, company, year, label, numeric_features, category_columns), audit


def make_smoke_panel(seed: int = 42, companies: int = 48, years: int = 8) -> pd.DataFrame:
    """Create a deterministic, event-aligned toy panel for software smoke checks only."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for firm in range(companies):
        risk = float(rng.normal())
        event_year = 1999 + int(rng.integers(3, years + 1)) if firm % 5 == 0 else None
        for offset in range(years):
            year = 1999 + offset
            rows.append({
                "company_id": f"SMOKE_{firm:03d}",
                "fiscal_year": year,
                "label": int(event_year is not None and year == event_year - 1),
                "available_date": f"{year + 1}-03-31",
                "liquidity": 1.5 - 0.08 * offset - 0.15 * risk + rng.normal(0, 0.15),
                "leverage": 0.35 + 0.04 * offset + 0.10 * risk + rng.normal(0, 0.08),
                "profitability": 0.08 - 0.02 * offset - 0.04 * risk + rng.normal(0, 0.03),
                "cash_flow": 0.12 - 0.015 * offset - 0.04 * risk + rng.normal(0, 0.04),
                "industry": f"sector_{firm % 4}",
            })
    return pd.DataFrame(rows)
