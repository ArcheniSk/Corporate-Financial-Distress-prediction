#!/usr/bin/env python3
"""Nested 3-fold grouped random search for the GBDT, CatBoost, and LSTM-GNN families."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fme.model import ReliabilityTemporalGNN
from run_polish_benchmark import (
    CatBoostClassifier,
    LGBMClassifier,
    XGBClassifier,
    build_inductive_graph,
    financial_relation_groups,
    grouped_splits,
    read_arff,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--configs-per-family", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=5)
    return parser.parse_args()


def sample_tree_configs(rng: random.Random, family: str, count: int, ratio: float) -> list[dict]:
    configs = []
    for _ in range(count):
        if family == "xgboost":
            configs.append({
                "n_estimators": rng.choice([250, 400, 600, 800]),
                "max_depth": rng.choice([2, 3, 4, 5, 6]),
                "learning_rate": rng.choice([0.01, 0.02, 0.04, 0.07, 0.1]),
                "min_child_weight": rng.choice([1, 3, 5, 10]),
                "subsample": rng.choice([0.65, 0.8, 0.95, 1.0]),
                "colsample_bytree": rng.choice([0.6, 0.75, 0.9, 1.0]),
                "reg_lambda": rng.choice([0.25, 0.5, 1.0, 2.0, 5.0]),
                "reg_alpha": rng.choice([0.0, 0.01, 0.1, 1.0]),
                "scale_pos_weight_multiplier": rng.choice([0.5, 0.75, 1.0, 1.5, 2.0]),
            })
        elif family == "catboost":
            configs.append({
                "iterations": rng.choice([250, 400, 600, 800]),
                "depth": rng.choice([4, 5, 6, 7, 8]),
                "learning_rate": rng.choice([0.01, 0.02, 0.04, 0.07, 0.1]),
                "l2_leaf_reg": rng.choice([1.0, 3.0, 5.0, 7.0, 9.0]),
                "random_strength": rng.choice([0.0, 0.25, 0.5, 1.0, 2.0]),
                "scale_pos_weight_multiplier": rng.choice([0.5, 0.75, 1.0, 1.5, 2.0]),
            })
        else:
            configs.append({
                "n_estimators": rng.choice([250, 400, 600, 800]),
                "num_leaves": rng.choice([7, 15, 23, 31, 47]),
                "max_depth": rng.choice([-1, 3, 5, 7, 9]),
                "learning_rate": rng.choice([0.01, 0.02, 0.04, 0.07, 0.1]),
                "min_child_samples": rng.choice([10, 20, 30, 50, 80]),
                "subsample": rng.choice([0.65, 0.8, 0.95, 1.0]),
                "colsample_bytree": rng.choice([0.6, 0.75, 0.9, 1.0]),
                "reg_lambda": rng.choice([0.25, 0.5, 1.0, 2.0, 5.0]),
                "scale_pos_weight_multiplier": rng.choice([0.5, 0.75, 1.0, 1.5, 2.0]),
            })
    return configs


def grouped_cv_folds(raw: np.ndarray, labels: np.ndarray, development_ids: np.ndarray, seed: int, folds: int = 3) -> list[tuple[np.ndarray, np.ndarray]]:
    signature = pd.util.hash_pandas_object(pd.DataFrame(raw), index=False).to_numpy()
    groups = pd.factorize(signature, sort=True)[0]
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    return [
        (development_ids[fit_rel], development_ids[holdout_rel])
        for fit_rel, holdout_rel in splitter.split(raw[development_ids], labels[development_ids], groups[development_ids])
    ]


def tune_tree_family(family: str, configs: list[dict], values: np.ndarray, labels: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]], seed: int) -> dict:
    package = {"xgboost": XGBClassifier, "lightgbm": LGBMClassifier, "catboost": CatBoostClassifier}[family]
    if package is None:
        raise RuntimeError(f"{family} is required for the declared search but is not installed.")
    best_score, best_config = -1.0, None
    for index, config in enumerate(configs, start=1):
        fold_scores = []
        weight_multiplier = float(config["scale_pos_weight_multiplier"])
        for fold_index, (train_ids, val_ids) in enumerate(folds):
            y_train = labels[train_ids]
            positive = max(int(y_train.sum()), 1)
            negative = max(len(train_ids) - positive, 1)
            parameters = {key: value for key, value in config.items() if key != "scale_pos_weight_multiplier"}
            parameters["scale_pos_weight"] = (negative / positive) * weight_multiplier
            common = {"n_jobs": -1, "random_state": seed + index * 17 + fold_index}
            if family == "xgboost":
                model = XGBClassifier(**parameters, eval_metric="aucpr", tree_method="hist", **common)
            elif family == "lightgbm":
                model = LGBMClassifier(**parameters, objective="binary", verbosity=-1, **common)
            else:
                parameters["scale_pos_weight"] = (negative / positive) * weight_multiplier
                model = CatBoostClassifier(
                    **parameters, loss_function="Logloss", eval_metric="PRAUC", verbose=False,
                    allow_writing_files=False, thread_count=-1, random_seed=common["random_state"],
                )
            pipeline = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), RobustScaler(), model)
            pipeline.fit(values[train_ids], y_train)
            fold_scores.append(float(average_precision_score(labels[val_ids], pipeline.predict_proba(values[val_ids])[:, 1])))
        score = float(np.mean(fold_scores))
        if score > best_score:
            best_score, best_config = score, config
        if index % 10 == 0:
            print(f"{family} 3-fold grouped search {index}/{len(configs)} | best mean PR-AUC {best_score:.4f}", flush=True)
    return {"inner_cv_mean_pr_auc": best_score, "parameters": best_config, "configs_evaluated": len(configs), "inner_folds": len(folds)}


def tune_graph_family(configs: list[dict], raw: np.ndarray, y: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]], feature_columns: list[str], groups: dict[str, list[str]], device: torch.device, seed: int, epochs: int, patience: int) -> dict:
    relation_names = list(groups)
    graph_cache: dict[tuple[int, int], tuple] = {}
    fold_data = []
    all_labels = torch.as_tensor(y.astype(np.float32), dtype=torch.float32, device=device)
    for fold_index, (train_ids, val_ids) in enumerate(folds):
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        scaler = RobustScaler()
        x_all = scaler.fit_transform(imputer.fit_transform(raw[train_ids]))
        x_all = scaler.transform(imputer.transform(raw)).astype(np.float32)
        fold_data.append((
            train_ids, val_ids,
            torch.as_tensor(x_all[:, None, :], dtype=torch.float32, device=device),
            y[val_ids],
        ))
    best_score, best_config = -1.0, None
    for index, config in enumerate(configs, start=1):
        fold_scores = []
        for fold_index, (train_ids, val_ids, x, y_val) in enumerate(fold_data):
            k = int(config["neighbors"])
            cache_key = (fold_index, k)
            if cache_key not in graph_cache:
                x_all = x[:, 0, :].detach().cpu().numpy()
                graph_train = build_inductive_graph(x_all, train_ids, np.zeros(0, dtype=np.int64), feature_columns, groups, k=k, weighted=True)
                graph_val = build_inductive_graph(x_all, train_ids, val_ids, feature_columns, groups, k=k, weighted=True)
                graph_cache[cache_key] = graph_train, graph_val
            graph_train, graph_val = graph_cache[cache_key]
            torch.manual_seed(seed + index * 31 + fold_index)
            model = ReliabilityTemporalGNN(len(feature_columns), config["hidden_dim"], relation_names, dropout=config["dropout"]).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
            y_train = all_labels[torch.as_tensor(train_ids, dtype=torch.long, device=device)]
            pos = max(float(y_train.sum()), 1.0)
            neg = max(float(len(train_ids) - y_train.sum()), 1.0)
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((neg / pos) * config["positive_weight_multiplier"], dtype=torch.float32, device=device))
            train_index = torch.as_tensor(train_ids, dtype=torch.long, device=device)
            best_ap, stale = -1.0, 0
            for _epoch in range(1, epochs + 1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(x, graph_train)[train_index], all_labels[train_index])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    scores = torch.sigmoid(model(x, graph_val)).detach().cpu().numpy()[val_ids]
                ap = float(average_precision_score(y_val, scores))
                if ap > best_ap:
                    best_ap, stale = ap, 0
                else:
                    stale += 1
                if stale >= patience:
                    break
            fold_scores.append(best_ap)
        score = float(np.mean(fold_scores))
        if score > best_score:
            best_score, best_config = score, dict(config)
        if index % 10 == 0:
            print(f"LSTM-GNN 3-fold grouped search {index}/{len(configs)} | best mean PR-AUC {best_score:.4f}", flush=True)
    return {"inner_cv_mean_pr_auc": best_score, "parameters": best_config, "configs_evaluated": len(configs), "inner_folds": len(folds)}


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA tuning was requested but is unavailable.")
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    source = Path(args.data_file)
    frame, y, feature_columns = read_arff(source)
    raw = frame.to_numpy(dtype=np.float64, copy=True)
    raw[~np.isfinite(raw)] = np.nan
    train_ids, val_ids, _test_ids, _ = grouped_splits(raw, y, args.seed)
    development_ids = np.concatenate([train_ids, val_ids])
    folds = grouped_cv_folds(raw, y, development_ids, args.seed + 1001, folds=3)
    y_development = y[development_ids]
    pos_ratio = max(len(y_development) - int(y_development.sum()), 1) / max(int(y_development.sum()), 1)
    rng = random.Random(args.seed)
    tree_search = {family: sample_tree_configs(rng, family, args.configs_per_family, pos_ratio) for family in ("xgboost", "lightgbm", "catboost")}
    started = time.perf_counter()
    tree_results = {family: tune_tree_family(family, configs, raw, y, folds, args.seed) for family, configs in tree_search.items()}
    missingness_all = np.concatenate([raw, np.isnan(raw).astype(np.float32)], axis=1)
    missingness_configs = sample_tree_configs(rng, "lightgbm", args.configs_per_family, pos_ratio)
    tree_results["lightgbm_missingness_prior"] = tune_tree_family(
        "lightgbm", missingness_configs, missingness_all, y, folds, args.seed + 1,
    )

    groups = financial_relation_groups(feature_columns)
    graph_configs = []
    for _ in range(args.configs_per_family):
        graph_configs.append({
            "hidden_dim": rng.choice([32, 64, 96, 128]),
            "dropout": rng.choice([0.0, 0.1, 0.2, 0.35, 0.5]),
            "learning_rate": rng.choice([0.0002, 0.0005, 0.001, 0.002, 0.004]),
            "weight_decay": rng.choice([1e-6, 1e-5, 1e-4, 1e-3]),
            "neighbors": rng.choice([5, 10, 15, 20, 30]),
            "positive_weight_multiplier": rng.choice([0.5, 0.75, 1.0, 1.5, 2.0]),
        })
    graph_result = tune_graph_family(graph_configs, raw, y, folds, feature_columns, groups, device, args.seed, args.epochs, args.patience)
    report = {
        "source": str(source),
        "seed": args.seed,
        "device": str(device),
        "selection_metric": "mean 3-fold inner grouped cross-validation average precision",
        "test_accessed": False,
        "inner_cv": "3-fold stratified grouped cross-validation within the outer development partition (training plus validation); exact duplicate feature rows remain in one fold.",
        "configs_per_family": args.configs_per_family,
        "families": {**tree_results, "lstm_gnn": graph_result},
        "search_seconds": round(time.perf_counter() - started, 2),
        "warning": "Configurations use inner grouped cross-validation and do not inspect outer test labels; earlier exploratory test results informed architecture development, so the full research iteration remains exploratory.",
    }
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_file).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote 3-fold inner grouped-CV tuning results to {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
