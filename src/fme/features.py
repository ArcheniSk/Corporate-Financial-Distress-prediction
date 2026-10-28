"""Leakage-aware feature preparation and longitudinal sequence construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RobustFeatureTransform:
    columns: list[str]
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    median: np.ndarray | None = None
    scale: np.ndarray | None = None

    def fit(self, train: pd.DataFrame) -> "RobustFeatureTransform":
        values = train[self.columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64, copy=True)
        values[~np.isfinite(values)] = np.nan
        with np.errstate(all="ignore"):
            self.lower = np.nanquantile(values, 0.01, axis=0)
            self.upper = np.nanquantile(values, 0.99, axis=0)
            self.median = np.nanmedian(values, axis=0)
        self.lower = np.nan_to_num(self.lower, nan=0.0, posinf=0.0, neginf=0.0)
        self.upper = np.nan_to_num(self.upper, nan=0.0, posinf=0.0, neginf=0.0)
        self.median = np.nan_to_num(self.median, nan=0.0, posinf=0.0, neginf=0.0)
        clipped = np.clip(values, self.lower, self.upper)
        with np.errstate(all="ignore"):
            q25 = np.nanquantile(clipped, 0.25, axis=0)
            q75 = np.nanquantile(clipped, 0.75, axis=0)
        self.scale = np.nan_to_num(q75 - q25, nan=1.0, posinf=1.0, neginf=1.0)
        self.scale[self.scale < 1e-8] = 1.0
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if any(v is None for v in (self.lower, self.upper, self.median, self.scale)):
            raise RuntimeError("Feature transform must be fitted before transform().")
        values = frame[self.columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64, copy=True)
        values[~np.isfinite(values)] = np.nan
        values = np.where(np.isnan(values), self.median, values)
        values = np.clip(values, self.lower, self.upper)
        return ((values - self.median) / self.scale).astype(np.float32)


def relation_feature_groups(columns: list[str]) -> dict[str, list[str]]:
    """Use named financial groups when possible; otherwise make one honest peer view."""
    groups = {
        "liquidity": [],
        "leverage": [],
        "profitability": [],
        "cash_flow": [],
    }
    rules = {
        "liquidity": ("current", "quick", "cash", "working_capital", "liquid"),
        "leverage": ("debt", "liabilit", "leverage", "equity", "solvency"),
        "profitability": ("profit", "income", "margin", "roa", "roe", "ebit"),
        "cash_flow": ("cash_flow", "cashflow", "operating_cash", "cfo"),
    }
    for col in columns:
        low = col.lower()
        for group, tokens in rules.items():
            if any(token in low for token in tokens):
                groups[group].append(col)
                break
    groups = {name: cols for name, cols in groups.items() if cols}
    if not groups or all(col.startswith("X") and col[1:].isdigit() for col in columns):
        return {"financial_similarity": columns}
    assigned = {c for cols in groups.values() for c in cols}
    remainder = [c for c in columns if c not in assigned]
    if remainder:
        groups["other_financial"] = remainder
    return groups


def build_sequences(
    frame: pd.DataFrame,
    features: np.ndarray,
    *,
    company_column: str = "company_id",
    year_column: str = "fiscal_year",
    history_length: int = 5,
) -> tuple[np.ndarray, dict[int, np.ndarray], list[str]]:
    """Build one padded trailing sequence per row plus global firm indices."""
    if len(frame) != len(features):
        raise ValueError("Feature matrix row count must match frame row count.")
    companies = sorted(frame[company_column].astype(str).unique())
    company_index = {company: i for i, company in enumerate(companies)}
    result = np.zeros((len(frame), history_length, features.shape[1]), dtype=np.float32)
    by_company = {
        company: group.sort_values(year_column, kind="stable").index.to_numpy()
        for company, group in frame.groupby(company_column, sort=False)
    }
    row_position = {idx: pos for pos, idx in enumerate(frame.index)}
    for company, indices in by_company.items():
        for end_pos, idx in enumerate(indices):
            begin = max(0, end_pos - history_length + 1)
            selected = indices[begin : end_pos + 1]
            vals = np.stack([features[row_position[int(i)]] for i in selected])
            result[row_position[int(idx)], -len(vals) :, :] = vals
    years = sorted(frame[year_column].astype(int).unique())
    histories: dict[int, np.ndarray] = {}
    for year in years:
        indices = np.flatnonzero(frame[year_column].astype(int).to_numpy() == year)
        snapshot = np.zeros((len(companies), history_length, features.shape[1]), dtype=np.float32)
        snapshot_company_positions: dict[str, int] = {}
        for row_idx in indices:
            company = str(frame.iloc[row_idx][company_column])
            snapshot_company_positions[company] = row_idx
        for company, row_idx in snapshot_company_positions.items():
            snapshot[company_index[company]] = result[row_idx]
        histories[year] = snapshot
    return result, histories, companies
