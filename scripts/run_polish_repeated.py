#!/usr/bin/env python3
"""Repeat the Polish tabular/graph benchmark across predeclared random seeds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SCORE_FIELDS = ("pr_auc", "roc_auc", "brier", "precision", "recall", "threshold", "fit_seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--tuned-config", default=None)
    parser.add_argument("--tune-configs-per-family", type=int, default=0, help="Run validation-only tuning separately inside each seed split; 0 reuses --tuned-config or fixed defaults.")
    parser.add_argument("--tuning-epochs", type=int, default=40)
    parser.add_argument("--tuning-patience", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seed values must be unique.")
    if args.tune_configs_per_family < 0:
        raise ValueError("--tune-configs-per-family cannot be negative.")
    if args.tuned_config and args.tune_configs_per_family:
        raise ValueError("Choose either one supplied --tuned-config or per-seed tuning, not both.")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reports: list[dict] = []
    prediction_files: list[pd.DataFrame] = []
    for seed in args.seeds:
        seed_dir = output / f"seed-{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        tuned_config = args.tuned_config
        if args.tune_configs_per_family:
            tuned_config = str(seed_dir / "tuned-config.json")
            tuning_device = "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
            tune_command = [
                sys.executable, str(ROOT / "scripts" / "tune_polish.py"),
                "--data_file", args.data_file,
                "--output_file", tuned_config,
                "--device", tuning_device,
                "--seed", str(seed),
                "--configs-per-family", str(args.tune_configs_per_family),
                "--epochs", str(args.tuning_epochs),
                "--patience", str(args.tuning_patience),
            ]
            subprocess.run(tune_command, check=True, cwd=ROOT)
        command = [
            sys.executable, str(ROOT / "scripts" / "run_polish_benchmark.py"),
            "--data_file", args.data_file, "--output_dir", str(seed_dir),
            "--device", args.device, "--seed", str(seed), "--epochs", str(args.epochs),
            "--patience", str(args.patience), "--neighbors", str(args.neighbors),
            "--hidden-dim", str(args.hidden_dim), "--dropout", str(args.dropout),
            "--learning-rate", str(args.learning_rate), "--weight-decay", str(args.weight_decay),
        ]
        if tuned_config:
            command.extend(["--tuned-config", tuned_config])
        subprocess.run(command, check=True, cwd=ROOT)
        report = json.loads((seed_dir / "metrics.json").read_text(encoding="utf-8"))
        reports.append(report)
        prediction_frame = pd.read_csv(seed_dir / "predictions.csv")
        prediction_frame["seed"] = seed
        prediction_files.append(prediction_frame)

    model_names = sorted(set.intersection(*(set(report["models"]) for report in reports)))
    summary_models: dict[str, dict] = {}
    for model_name in model_names:
        per_seed = {str(report["seed"]): report["models"][model_name] for report in reports}
        summary: dict[str, object] = {"per_seed": per_seed}
        for metric in SCORE_FIELDS:
            values = np.asarray([report["models"][model_name].get(metric) for report in reports], dtype=float)
            finite = values[np.isfinite(values)]
            if len(finite):
                summary[f"mean_{metric}"] = float(finite.mean())
                summary[f"std_{metric}"] = float(finite.std(ddof=1)) if len(finite) > 1 else 0.0
        summary_models[model_name] = summary

    combined = {
        "source": reports[0]["source"],
        "citation": reports[0]["citation"],
        "license": reports[0]["license"],
        "forecast_horizon": reports[0]["forecast_horizon"],
        "device_by_seed": {str(report["seed"]): report["device"] for report in reports},
        "seeds": list(args.seeds),
        "evaluation": "Five predeclared stratified grouped random holdouts. Each split uses validation for selection and early stopping, then refits final models on train plus validation before scoring the independent test partition. Means and standard deviations describe split variability, not confidence intervals.",
        "evaluation_scope": reports[0]["evaluation_scope"],
        "temporal_graph_evaluation": False,
        "limitations": reports[0]["limitations"],
        "systematic_hyperparameter_search": reports[0]["systematic_hyperparameter_search"],
        "selected_tuning_config": reports[0].get("selected_tuning_config"),
        "selected_tuning_config_by_seed": {str(report["seed"]): report.get("selected_tuning_config") for report in reports},
        "selected_prior_hybrid_by_seed": {str(report["seed"]): report.get("selected_prior_hybrid") for report in reports},
        "selected_csa_hybrid_by_seed": {str(report["seed"]): report.get("selected_csa_hybrid") for report in reports},
        "confidence_intervals": reports[0]["confidence_intervals"],
        "rows_per_seed": {str(report["seed"]): report["rows"] for report in reports},
        "class_counts_by_seed": {str(report["seed"]): report["class_counts"] for report in reports},
        "split_rows_by_seed": {str(report["seed"]): report["split_rows"] for report in reports},
        "split_positive_counts_by_seed": {str(report["seed"]): report["split_positive_counts"] for report in reports},
        "unavailable_optional_baselines": reports[0]["unavailable_optional_baselines"],
        "models": summary_models,
    }
    (output / "metrics.json").write_text(json.dumps(combined, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pd.concat(prediction_files, ignore_index=True).to_csv(output / "predictions.csv", index=False)
    print(f"Completed {len(reports)} seeds; wrote repeated-run summary to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
