#!/usr/bin/env python3
"""Train baselines and the proposed model, or run a synthetic software smoke test."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fme.baselines import TemporalGCN, TemporalOnly
from fme.data import DatasetAuditError, load_panel, make_smoke_panel
from fme.features import RobustFeatureTransform, build_sequences, relation_feature_groups
from fme.graphs import SnapshotGraph, build_snapshot_graphs
from fme.metrics import binary_metrics, choose_recall_threshold
from fme.model import ReliabilityTemporalGNN, TabularMLP
from fme.labels import construct_horizon_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--data_file", type=str, default=None)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--output_dir", type=str, default="/output")
    parser.add_argument("--history_length", type=int, default=5)
    parser.add_argument("--train_end_year", type=int, default=2011)
    parser.add_argument("--validation_end_year", type=int, default=2014)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recall_target", type=float, default=0.8)
    parser.add_argument("--observation_end_date", type=str, default=None)
    parser.add_argument("--horizon_days", type=int, default=365)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Training device; use cuda to require a GPU.")
    parser.add_argument("--ablations", action="store_true", help="Run leave-one-relation-out and no-persistence variants.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _year_rows(frame: pd.DataFrame) -> dict[int, np.ndarray]:
    years = frame.fiscal_year.astype(int).to_numpy()
    return {int(year): np.flatnonzero(years == year) for year in np.unique(years)}


def _evaluate_neural(model, histories, graphs, label_by_row, row_by_year, years, device):
    model.eval()
    y_true: list[int] = []
    y_score: list[float] = []
    row_ids_all: list[int] = []
    with torch.no_grad():
        for year in years:
            row_ids = row_by_year.get(int(year), np.zeros(0, dtype=np.int64))
            if not len(row_ids):
                continue
            seq = torch.as_tensor(histories[int(year)], dtype=torch.float32, device=device)
            graph = graphs[int(year)]
            logits = model(seq, graph)
            probs = torch.sigmoid(logits).cpu().numpy()
            # global graph company order is shared with panel order's unique company sort.
            y_true.extend(label_by_row[row_ids].tolist())
            y_score.extend(probs[row_ids_to_global(row_ids, graphs[int(year)].node_ids)].tolist())
            row_ids_all.extend(row_ids.tolist())
    return np.asarray(y_true, dtype=int), np.asarray(y_score, dtype=float), np.asarray(row_ids_all, dtype=int)


def row_ids_to_global(row_ids: np.ndarray, node_ids: list[str]) -> np.ndarray:
    """Map panel row indices to global firm-node indices for each row."""
    # `node_ids` is sorted; the panel frame ordering is separately supplied by caller below.
    # This placeholder is replaced by a precomputed mapping before evaluation.
    if not hasattr(row_ids_to_global, "company_index"):
        raise RuntimeError("Company index mapping has not been initialized.")
    companies = row_ids_to_global.row_companies
    return np.asarray([row_ids_to_global.company_index[str(companies[int(i)])] for i in row_ids], dtype=np.int64)


def fit_neural(model, frame, features, snapshots, histories, row_by_year, train_years, val_years, epochs, patience, learning_rate, device, history_length, seed):
    labels = frame.label.to_numpy(dtype=np.float32)
    company_index = {firm: i for i, firm in enumerate(sorted(frame.company_id.astype(str).unique()))}
    row_ids_to_global.company_index = company_index
    row_ids_to_global.row_companies = frame.company_id.astype(str).to_numpy()
    train_labels = labels[np.flatnonzero(frame.fiscal_year.astype(int).to_numpy() <= max(train_years))]
    positive = max(float(train_labels.sum()), 1.0)
    negative = max(float(len(train_labels) - train_labels.sum()), 1.0)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_score = -1.0
    best_state = None
    best_epoch = 0
    stale = 0
    history_log = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for year in train_years:
            row_ids = row_by_year.get(year, np.zeros(0, dtype=np.int64))
            if not len(row_ids):
                continue
            optimizer.zero_grad(set_to_none=True)
            seq = torch.as_tensor(histories[year], dtype=torch.float32, device=device)
            logits = model(seq, snapshots[year])
            idx = torch.as_tensor(row_ids_to_global(row_ids, snapshots[year].node_ids), dtype=torch.long, device=device)
            target = torch.as_tensor(labels[row_ids], dtype=torch.float32, device=device)
            loss = loss_fn(logits[idx], target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            losses.append(float(loss.item()))
        y_val, p_val, _ = _evaluate_neural(model, histories, snapshots, labels, row_by_year, val_years, device)
        score = float(average_precision_score(y_val, p_val)) if len(np.unique(y_val)) == 2 else -1.0
        epoch_loss = float(np.mean(losses)) if losses else float("nan")
        history_log.append({"epoch": epoch, "train_loss": epoch_loss, "validation_pr_auc": score})
        print(f"Epoch {epoch}/{epochs} | Loss: {epoch_loss:.4f} | validation PR-AUC: {score:.4f}")
        if score > best_score:
            best_score, best_state, best_epoch, stale = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, epoch, 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history_log, best_epoch


def evaluate_model(model, histories, snapshots, frame, row_by_year, years, device, threshold):
    y_true, scores, row_ids = _evaluate_neural(model, histories, snapshots, frame.label.to_numpy(dtype=int), row_by_year, years, device)
    return binary_metrics(y_true, scores, threshold), pd.DataFrame({"row_id": row_ids, "year": frame.iloc[row_ids].fiscal_year.to_numpy(), "company_id": frame.iloc[row_ids].company_id.to_numpy(), "label": y_true, "score": scores, "threshold": threshold})


def filter_graphs(graphs, relations):
    return {year: SnapshotGraph({name: graph.relations[name] for name in relations if name in graph.relations}, graph.node_ids) for year, graph in graphs.items()}


def _run_tabular_baselines(frame, scaled, sequences, row_by_year, train_years, val_years, test_years, recall_target, seed, device, epochs):
    year_values = frame.fiscal_year.astype(int).to_numpy()
    train_ids = np.flatnonzero(np.isin(year_values, train_years))
    val_ids = np.flatnonzero(np.isin(year_values, val_years))
    test_ids = np.flatnonzero(np.isin(year_values, test_years))
    y = frame.label.to_numpy(dtype=int)
    feature_views = {
        "last_year": scaled,
        "flattened_history": sequences.reshape(len(sequences), -1),
    }
    model_factories = {
        "logistic_regression": lambda: LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        "decision_tree": lambda: DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, class_weight="balanced", random_state=seed),
        "random_forest": lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, l2_regularization=1.0, random_state=seed),
    }
    results, predictions = {}, {}
    for view_name, matrix in feature_views.items():
        for model_name, factory in model_factories.items():
            name = f"{model_name}_{view_name}"
            model = factory()
            if model_name == "hist_gradient_boosting":
                model.fit(matrix[train_ids], y[train_ids], sample_weight=compute_sample_weight("balanced", y[train_ids]))
            else:
                model.fit(matrix[train_ids], y[train_ids])
            val_score = model.predict_proba(matrix[val_ids])[:, 1]
            threshold = choose_recall_threshold(y[val_ids], val_score, recall_target)
            test_score = model.predict_proba(matrix[test_ids])[:, 1]
            results[name] = binary_metrics(y[test_ids], test_score, threshold)
            validation_metrics = binary_metrics(y[val_ids], val_score, threshold)
            results[name].update({"validation_pr_auc": validation_metrics["pr_auc"], "validation_roc_auc": validation_metrics["roc_auc"]})
            predictions[name] = pd.DataFrame({"row_id": test_ids, "year": year_values[test_ids], "company_id": frame.iloc[test_ids].company_id.to_numpy(), "label": y[test_ids], "score": test_score, "threshold": threshold})

        mlp_name = f"mlp_{view_name}"
        mlp = TabularMLP(matrix.shape[1]).to(device)
        optimizer = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
        pos = max(int(y[train_ids].sum()), 1)
        neg = max(len(train_ids) - pos, 1)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / pos, dtype=torch.float32, device=device))
        x_train = torch.as_tensor(matrix[train_ids], dtype=torch.float32, device=device)
        y_train = torch.as_tensor(y[train_ids], dtype=torch.float32, device=device)
        rng = np.random.default_rng(seed)
        best_ap, best_state, stale = -1.0, None, 0
        for epoch in range(min(epochs, 40)):
            mlp.train()
            order = rng.permutation(len(train_ids))
            losses = []
            for start in range(0, len(order), 512):
                batch = torch.as_tensor(order[start : start + 512], dtype=torch.long, device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(mlp(x_train[batch]), y_train[batch])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mlp.parameters(), 2.0)
                optimizer.step()
                losses.append(float(loss.item()))
            mlp.eval()
            with torch.no_grad():
                val_tensor = torch.as_tensor(matrix[val_ids], dtype=torch.float32, device=device)
                val_score = torch.sigmoid(mlp(val_tensor)).cpu().numpy()
            ap = float(average_precision_score(y[val_ids], val_score)) if len(np.unique(y[val_ids])) == 2 else -1.0
            print(f"{mlp_name} epoch {epoch + 1}/{min(epochs, 40)} | Loss: {np.mean(losses):.4f} | validation PR-AUC: {ap:.4f}")
            if ap > best_ap:
                best_ap = ap
                best_state = {key: value.detach().cpu().clone() for key, value in mlp.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= 5:
                break
        if best_state is not None:
            mlp.load_state_dict(best_state)
        mlp.eval()
        with torch.no_grad():
            val_score = torch.sigmoid(mlp(torch.as_tensor(matrix[val_ids], dtype=torch.float32, device=device))).cpu().numpy()
            test_score = torch.sigmoid(mlp(torch.as_tensor(matrix[test_ids], dtype=torch.float32, device=device))).cpu().numpy()
        threshold = choose_recall_threshold(y[val_ids], val_score, recall_target)
        results[mlp_name] = binary_metrics(y[test_ids], test_score, threshold)
        validation_metrics = binary_metrics(y[val_ids], val_score, threshold)
        results[mlp_name].update({"validation_pr_auc": validation_metrics["pr_auc"], "validation_roc_auc": validation_metrics["roc_auc"]})
        predictions[mlp_name] = pd.DataFrame({"row_id": test_ids, "year": year_values[test_ids], "company_id": frame.iloc[test_ids].company_id.to_numpy(), "label": y[test_ids], "score": test_score, "threshold": threshold})
    return results, predictions


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    print(f"Data directory: {data_dir}")
    if not data_dir.exists():
        if not args.smoke_test:
            raise FileNotFoundError(f"Data dir not found: {data_dir}")
        data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Contents: {sorted(p.name for p in data_dir.iterdir())}")

    if args.smoke_test:
        frame = make_smoke_panel(seed=args.seed, companies=40, years=8)
        source_label = "synthetic_smoke_only"
        epochs = min(args.epochs, 2)
    else:
        data_path = Path(args.data_file) if args.data_file else data_dir / "panel.csv"
        try:
            panel, audit = load_panel(data_path, require_temporal_labels=False)
        except (DatasetAuditError, FileNotFoundError) as exc:
            (output_dir / "audit_failure.json").write_text(json.dumps({"error": str(exc)}, indent=2) + "\n")
            raise
        frame = panel.frame.reset_index(drop=True)
        label_metadata = {}
        if "event_date" in frame.columns:
            if not args.observation_end_date:
                raise DatasetAuditError("Pass --observation_end_date when constructing labels from event_date.")
            frame = construct_horizon_labels(frame, observation_end_date=args.observation_end_date, horizon_days=args.horizon_days)
            label_metadata = dict(frame.attrs)
        transition_firms = int((frame.groupby("company_id").label.nunique() > 1).sum())
        audit["firms_with_label_transitions"] = transition_firms
        audit["constant_within_company"] = transition_firms == 0
        audit["label_counts"] = {str(int(k)): int(v) for k, v in frame.label.value_counts().sort_index().items()}
        audit["firms_with_positive_label"] = int(frame.loc[frame.label.eq(1), "company_id"].nunique())
        audit["label_construction"] = label_metadata or {"source": "validated panel-provided binary label", "horizon_days": args.horizon_days}
        (output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        if audit["constant_within_company"]:
            message = ("Target audit failed: labels are constant within all companies. "
                       "This dataset is not a point-in-time event-horizon panel. "
                       f"See {output_dir / 'audit.json'} and docs/data-audit.md.")
            (output_dir / "audit_failure.json").write_text(json.dumps({"error": message, "audit": audit}, indent=2) + "\n")
            raise DatasetAuditError(message)
        if "available_date" not in frame.columns:
            message = "Point-in-time forecasting requires available_date; provide filing/publication dates for every company-year."
            (output_dir / "audit_failure.json").write_text(json.dumps({"error": message, "audit": audit}, indent=2) + "\n")
            raise DatasetAuditError(message)
        parsed_availability = pd.to_datetime(frame.available_date, errors="coerce", utc=True)
        if parsed_availability.isna().any():
            message = "available_date contains missing or invalid dates."
            (output_dir / "audit_failure.json").write_text(json.dumps({"error": message, "audit": audit}, indent=2) + "\n")
            raise DatasetAuditError(message)
        frame["available_date"] = parsed_availability
        source_label = str(data_path)
        epochs = args.epochs

    if "company_id" not in frame.columns or "fiscal_year" not in frame.columns or "label" not in frame.columns:
        raise DatasetAuditError("Panel requires company_id, fiscal_year, and label columns.")
    frame["company_id"] = frame.company_id.astype(str)
    frame = frame.sort_values(["company_id", "fiscal_year"], kind="stable").reset_index(drop=True)
    excluded = {"company_id", "fiscal_year", "label", "available_date", "event_date", "filing_date"}
    category_columns = [c for c in frame.columns if c.lower() in {"industry", "division", "majorgroup", "sic", "siccode", "sic_code"}]
    feature_columns = [c for c in frame.select_dtypes(include=[np.number]).columns if c not in excluded and c not in category_columns]
    if not feature_columns:
        raise DatasetAuditError("No numeric feature columns found.")
    year_values = frame.fiscal_year.astype(int).to_numpy()
    if args.smoke_test:
        max_year = int(year_values.max())
        train_end = max_year - 3
        validation_end = max_year - 1
    else:
        train_end, validation_end = args.train_end_year, args.validation_end_year
    train_years = [int(y) for y in np.unique(year_values) if y <= train_end]
    val_years = [int(y) for y in np.unique(year_values) if train_end < y <= validation_end]
    test_years = [int(y) for y in np.unique(year_values) if y > validation_end]
    if not train_years or not val_years or not test_years:
        raise DatasetAuditError(f"Need non-empty chronological train/validation/test periods; got {train_years}, {val_years}, {test_years}.")
    train_rows = np.flatnonzero(np.isin(year_values, train_years))
    transform = RobustFeatureTransform(feature_columns).fit(frame.iloc[train_rows])
    scaled = transform.transform(frame)
    seq_rows, histories, companies = build_sequences(frame, scaled, history_length=args.history_length)
    groups = relation_feature_groups(feature_columns)
    graphs, graph_companies = build_snapshot_graphs(frame, scaled, groups, feature_columns, k=args.neighbors, seed=args.seed)
    conventional_graphs, conventional_companies = build_snapshot_graphs(frame, scaled, groups, feature_columns, k=args.neighbors, seed=args.seed, persistence_weighting=False)
    if companies != graph_companies or companies != conventional_companies:
        raise RuntimeError("Sequence and graph company ordering mismatch.")
    relations = sorted({relation for graph in graphs.values() for relation in graph.relations})
    row_by_year = _year_rows(frame)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Enable a GPU accelerator for this run.")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    print(f"Device: {device}; feature views: {list(groups)}; graph relations: {relations}")
    print(f"Chronological years: train={train_years}, validation={val_years}, test={test_years}")

    tabular_results, prediction_tables = _run_tabular_baselines(frame, scaled, seq_rows, row_by_year, train_years, val_years, test_years, args.recall_target, args.seed, device, epochs)
    neural_models = {
        "lstm": TemporalOnly(len(feature_columns), args.hidden_dim),
        "gcn": TemporalGCN(len(feature_columns), args.hidden_dim, mode="gcn"),
        "graphsage": TemporalGCN(len(feature_columns), args.hidden_dim, mode="sage"),
        "reliability_temporal_gnn": ReliabilityTemporalGNN(len(feature_columns), args.hidden_dim, relations),
    }
    all_results = dict(tabular_results)
    histories_by_model = {}
    best_model_state = None
    model_specs = [
        ("lstm", neural_models["lstm"], conventional_graphs),
        ("gcn", neural_models["gcn"], conventional_graphs),
        ("graphsage", neural_models["graphsage"], conventional_graphs),
        ("reliability_temporal_gnn", neural_models["reliability_temporal_gnn"], graphs),
    ]
    if args.ablations and not args.smoke_test:
        for removed in relations:
            remaining = [r for r in relations if r != removed]
            if remaining:
                model_specs.append((f"reliability_without_{removed}", ReliabilityTemporalGNN(len(feature_columns), args.hidden_dim, remaining), filter_graphs(graphs, remaining)))
        no_persistence = conventional_graphs
        model_specs.append(("temporal_gnn_no_persistence", ReliabilityTemporalGNN(len(feature_columns), args.hidden_dim, relations), no_persistence))
    for name, model, model_graphs in model_specs:
        model = model.to(device)
        model, history_log, best_epoch = fit_neural(model, frame, seq_rows, model_graphs, histories, row_by_year, train_years, val_years, epochs, args.patience, args.learning_rate, device, args.history_length, args.seed)
        validation_y, validation_score, _ = _evaluate_neural(model, histories, model_graphs, frame.label.to_numpy(dtype=int), row_by_year, val_years, device)
        validation_threshold = choose_recall_threshold(validation_y, validation_score, args.recall_target)
        validation_metrics = binary_metrics(validation_y, validation_score, validation_threshold)
        all_results[name], pred = evaluate_model(model, histories, model_graphs, frame, row_by_year, test_years, device, validation_threshold)
        all_results[name].update({"validation_pr_auc": validation_metrics["pr_auc"], "validation_roc_auc": validation_metrics["roc_auc"]})
        prediction_tables[name] = pred
        histories_by_model[name] = history_log
        if name == "reliability_temporal_gnn":
            best_model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(f"{name} | test PR-AUC: {all_results[name]['pr_auc']}")
    predictions = pd.concat([table.assign(model=name) for name, table in prediction_tables.items()], ignore_index=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    metrics = {
        "source": source_label,
        "synthetic_smoke_only": bool(args.smoke_test),
        "seed": args.seed,
        "device": str(device),
        "feature_columns": feature_columns,
        "feature_groups": groups,
        "graph_relations": relations,
        "history_length": args.history_length,
        "splits": {"train_years": train_years, "validation_years": val_years, "test_years": test_years},
        "models": all_results,
        "training_curves": histories_by_model,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    if best_model_state is not None:
        torch.save(best_model_state, output_dir / "model.pt")
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, values in histories_by_model.items():
        ax.plot([x["epoch"] for x in values], [x["train_loss"] for x in values], label=name)
    ax.set(xlabel="Epoch", ylabel="Training loss", title="Training curves (smoke-test curves are synthetic)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png", dpi=160)
    plt.close(fig)
    print(f"Wrote outputs to {output_dir}")
    print(json.dumps({name: value["pr_auc"] for name, value in all_results.items()}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DatasetAuditError as exc:
        print(f"DATA AUDIT BLOCKED TRAINING: {exc}", file=sys.stderr)
        raise SystemExit(2)
