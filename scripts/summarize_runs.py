#!/usr/bin/env python3
"""Summarize repeated runs and bootstrap test PR-AUC intervals by company."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def bootstrap_company_ci(frame: pd.DataFrame, rng: np.random.Generator, n_bootstrap: int) -> dict:
    firms = frame.company_id.unique()
    pr_values, roc_values = [], []
    by_firm = {firm: group for firm, group in frame.groupby("company_id", sort=False)}
    for _ in range(n_bootstrap):
        sampled = rng.choice(firms, size=len(firms), replace=True)
        sample = pd.concat([by_firm[firm] for firm in sampled], ignore_index=True)
        if sample.label.nunique() < 2:
            continue
        pr_values.append(float(average_precision_score(sample.label, sample.score)))
        roc_values.append(float(roc_auc_score(sample.label, sample.score)))
    return {
        "pr_auc_95ci": [float(np.quantile(pr_values, 0.025)), float(np.quantile(pr_values, 0.975))] if pr_values else None,
        "roc_auc_95ci": [float(np.quantile(roc_values, 0.025)), float(np.quantile(roc_values, 0.975))] if roc_values else None,
        "bootstrap_replicates_used": len(pr_values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", help="Directories containing metrics.json and predictions.csv")
    parser.add_argument("--output", default="results/summary.json")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=420)
    args = parser.parse_args()
    records, predictions = [], []
    for run_dir in map(Path, args.run_dirs):
        metrics_path = run_dir / "metrics.json"
        prediction_path = run_dir / "predictions.csv"
        if not metrics_path.exists() or not prediction_path.exists():
            raise FileNotFoundError(f"Missing metrics or predictions in {run_dir}")
        metrics = json.loads(metrics_path.read_text())
        if metrics.get("synthetic_smoke_only"):
            raise ValueError("Synthetic smoke-test outputs cannot be summarized as research results.")
        records.append({"run_dir": str(run_dir), "seed": metrics["seed"], "models": metrics["models"]})
        predictions.append(pd.read_csv(prediction_path).assign(seed=metrics["seed"]))
    joined = pd.concat(predictions, ignore_index=True)
    # Average repeated-seed scores for each firm-year before resampling firms.
    pooled = joined.groupby(["model", "company_id", "year", "label"], as_index=False).agg(score=("score", "mean"), threshold=("threshold", "mean"))
    rng = np.random.default_rng(args.seed)
    summary = {"runs": records, "models": {}}
    for model_name, group in pooled.groupby("model", sort=True):
        run_values = [record["models"].get(model_name, {}).get("pr_auc") for record in records]
        run_values = [float(value) for value in run_values if value is not None]
        item = {
            "seed_count": len(run_values),
            "pr_auc_mean_over_runs": float(np.mean(run_values)) if run_values else None,
            "pr_auc_std_over_runs": float(np.std(run_values, ddof=1)) if len(run_values) > 1 else None,
            "pooled_score_pr_auc": float(average_precision_score(group.label, group.score)) if group.label.nunique() == 2 else None,
        }
        item.update(bootstrap_company_ci(group, rng, args.bootstrap))
        summary["models"][model_name] = item
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
