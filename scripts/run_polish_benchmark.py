#!/usr/bin/env python3
"""Run an exploratory tabular benchmark on UCI's Polish 5year ARFF subset.

This dataset has no company identifiers or fiscal-year keys, so this script uses
a stratified random holdout and does not train temporal or graph models.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import arff
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))

from fme.baselines import TemporalGCN, TemporalOnly
from fme.graphs import RelationGraph, SnapshotGraph
from fme.metrics import binary_metrics, choose_recall_threshold
from fme.model import PriorGuidedReliabilityTemporalGNN, PriorGuidedTabularResidual, ReliabilityTemporalGNN, TabularMLP

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

UCI_RELATION_ATTRIBUTE_IDS = {
    "liquidity": {3, 4, 5, 28, 37, 40, 46, 50, 55, 57},
    "solvency": {2, 8, 10, 15, 16, 17, 25, 26, 27, 30, 38, 41, 51, 53, 54, 59, 62, 63},
    "profitability": {1, 6, 7, 11, 12, 13, 14, 18, 19, 22, 23, 24, 31, 35, 39, 42, 45, 48, 49, 56, 58},
    "efficiency_scale": {9, 20, 21, 29, 32, 33, 34, 36, 43, 44, 47, 52, 60, 61, 64},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_file", required=True, help="Path to 5year.arff from the UCI Polish dataset.")
    parser.add_argument("--output_dir", default="/kaggle/working/fme-results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recall_target", type=float, default=0.8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--tuned-config", default=None, help="Validation-selected parameters from scripts/tune_polish.py.")
    return parser.parse_args()


def read_arff(path: Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    records, _ = arff.loadarff(path)
    frame = pd.DataFrame(records)
    if "class" not in frame:
        raise ValueError("Expected a binary 'class' attribute in the ARFF file.")
    labels = frame.pop("class")
    if labels.dtype == object:
        labels = labels.map(lambda value: value.decode("utf-8") if isinstance(value, bytes) else value)
    y = pd.to_numeric(labels, errors="coerce").to_numpy(dtype=np.int64)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Expected both 0 and 1 class labels in the ARFF file.")
    feature_columns = list(frame.columns)
    values = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64, copy=True)
    values[~np.isfinite(values)] = np.nan
    if not feature_columns:
        raise ValueError("No financial predictor attributes were found.")
    return frame, y, feature_columns


def grouped_splits(values: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Create approximately 60/20/20 stratified splits, grouping identical inputs."""
    signature = pd.util.hash_pandas_object(pd.DataFrame(values), index=False).to_numpy()
    groups = pd.factorize(signature, sort=True)[0]
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_val, test = next(outer.split(values, y, groups))
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed + 1)
    train_rel, val_rel = next(inner.split(values[train_val], y[train_val], groups[train_val]))
    train = train_val[train_rel]
    val = train_val[val_rel]
    if any(len(np.unique(y[ids])) != 2 for ids in (train, val, test)):
        raise ValueError("A split does not contain both target classes; try a different seed.")
    return train, val, test, int(len(np.unique(groups)))


def financial_relation_groups(feature_columns: list[str]) -> dict[str, list[str]]:
    """Map UCI's ordered Attr1..Attr64 ratios to broad financial relation channels."""
    groups: dict[str, list[str]] = {name: [] for name in UCI_RELATION_ATTRIBUTE_IDS}
    for column in feature_columns:
        try:
            attribute_id = int(column.removeprefix("Attr"))
        except ValueError:
            raise ValueError(f"Expected UCI attribute names Attr1..Attr64, got {column!r}.")
        match = [name for name, ids in UCI_RELATION_ATTRIBUTE_IDS.items() if attribute_id in ids]
        if len(match) != 1:
            raise ValueError(f"UCI attribute {column} is missing or assigned to multiple relation groups.")
        groups[match[0]].append(column)
    if sum(map(len, groups.values())) != len(feature_columns) or any(not cols for cols in groups.values()):
        raise ValueError("Financial relation groups do not cover the provided predictors.")
    return groups


def build_inductive_graph(values: np.ndarray, train_ids: np.ndarray, query_ids: np.ndarray, feature_columns: list[str], groups: dict[str, list[str]], *, k: int, weighted: bool) -> SnapshotGraph:
    """Build relation-specific cosine edges from training peers to train/query targets only."""
    n_nodes = len(values)
    node_ids = [str(i) for i in range(n_nodes)]
    positions = {name: index for index, name in enumerate(feature_columns)}
    train_set = set(map(int, train_ids))
    targets = np.unique(np.concatenate([train_ids, query_ids])).astype(np.int64)
    relation_graphs: dict[str, RelationGraph] = {}
    if len(train_ids) > 1:
        for relation, columns in groups.items():
            selected = [positions[column] for column in columns]
            relation_values = values[:, selected]
            neighbors = NearestNeighbors(n_neighbors=min(k + 1, len(train_ids)), metric="cosine", algorithm="brute", n_jobs=-1)
            neighbors.fit(relation_values[train_ids])
            distances, indices = neighbors.kneighbors(relation_values[targets])
            source_ids: list[int] = []
            target_ids: list[int] = []
            edge_weights: list[float] = []
            for target, dists, local_neighbors in zip(targets, distances, indices):
                peer_limit = min(k + 1 if int(target) in train_set else k, len(local_neighbors))
                for distance, local_source in zip(dists[:peer_limit], local_neighbors[:peer_limit]):
                    source = int(train_ids[int(local_source)])
                    if source == int(target):
                        continue
                    source_ids.append(source)
                    target_ids.append(int(target))
                    edge_weights.append(float(np.clip(1.0 - distance, 0.01, 1.0)) if weighted else 1.0)
            relation_graphs[relation] = RelationGraph(np.asarray(source_ids, dtype=np.int64), np.asarray(target_ids, dtype=np.int64), np.asarray(edge_weights, dtype=np.float32))
    return SnapshotGraph(relation_graphs, node_ids)


def filter_graph(graph: SnapshotGraph, relations: list[str]) -> SnapshotGraph:
    return SnapshotGraph({name: graph.relations[name] for name in relations if name in graph.relations}, graph.node_ids)


def crossfit_lightgbm_prior(raw_values: np.ndarray, labels: np.ndarray, train_ids: np.ndarray, base_parameters: dict[str, object], positive_weight_multiplier: float, seed: int) -> tuple[np.ndarray, np.ndarray, object]:
    """Fit a missingness-aware LightGBM prior and return grouped OOF/full-training logits."""
    signature = pd.util.hash_pandas_object(pd.DataFrame(raw_values), index=False).to_numpy()
    groups = pd.factorize(signature, sort=True)[0]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + 991)
    missingness = np.isnan(raw_values).astype(np.float32)
    features = np.concatenate([raw_values.astype(np.float32, copy=False), missingness], axis=1)
    positives = max(int(labels[train_ids].sum()), 1)
    negatives = max(len(train_ids) - int(labels[train_ids].sum()), 1)
    full_params = dict(base_parameters)
    full_params["scale_pos_weight"] = (negatives / positives) * positive_weight_multiplier
    full_model = LGBMClassifier(**full_params, objective="binary", verbosity=-1, n_jobs=-1, random_state=seed)
    full_model.fit(features[train_ids], labels[train_ids])
    probability = full_model.predict_proba(features)[:, 1].astype(np.float64, copy=True)
    oof_probability = np.full(len(train_ids), np.nan, dtype=np.float64)
    for fold, (fit_rel, holdout_rel) in enumerate(splitter.split(raw_values[train_ids], labels[train_ids], groups[train_ids])):
        fit_ids, holdout_ids = train_ids[fit_rel], train_ids[holdout_rel]
        positives = max(int(labels[fit_ids].sum()), 1)
        negatives = max(len(fit_ids) - int(labels[fit_ids].sum()), 1)
        fold_params = dict(base_parameters)
        fold_params["scale_pos_weight"] = (negatives / positives) * positive_weight_multiplier
        fold_model = LGBMClassifier(**fold_params, objective="binary", verbosity=-1, n_jobs=-1, random_state=seed + fold + 1)
        fold_model.fit(features[fit_ids], labels[fit_ids])
        oof_probability[holdout_rel] = fold_model.predict_proba(features[holdout_ids])[:, 1]
    if not np.isfinite(oof_probability).all():
        raise RuntimeError("Grouped out-of-fold LightGBM prior did not cover every training row.")
    probability[train_ids] = oof_probability
    probability = np.clip(probability, 1e-6, 1.0 - 1e-6)
    logits = np.log(probability / (1.0 - probability)).astype(np.float32)
    return logits, probability.astype(np.float32), full_model


def fit_graph_model(model, sequences, graph_train, graph_val, graph_test, labels, train_ids, val_ids, test_ids, device, recall_target=0.8, learning_rate=1e-3, weight_decay=1e-4, positive_weight_multiplier=1.0, epochs=50, patience=7, prior_logits=None, error_weight_alpha=0.0, error_weight_gamma=1.0, class_weight=True, cosine_schedule=False, gradient_clip=2.0, refit_ids=None, refit_sequences=None, refit_train_graph=None, refit_test_graph=None, refit_prior_logits=None, validation_loss_tiebreak=False, min_epochs=0):
    model = model.to(device)
    initial_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()} if refit_ids is not None else None
    labels_tensor = torch.as_tensor(np.asarray(labels, dtype=np.float32).copy(), dtype=torch.float32, device=device)
    positive = max(float(labels[train_ids].sum()), 1.0)
    negative = max(float(len(train_ids) - labels[train_ids].sum()), 1.0)
    pos_weight = (negative / positive) * positive_weight_multiplier if class_weight else 1.0
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device), reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1)) if cosine_schedule else None
    all_sequences = torch.as_tensor(sequences, dtype=torch.float32, device=device)
    train_index = torch.as_tensor(train_ids, dtype=torch.long, device=device)
    prior_tensor = torch.as_tensor(prior_logits, dtype=torch.float32, device=device) if prior_logits is not None else None
    if prior_tensor is not None:
        prior_probability = torch.sigmoid(prior_tensor[train_index]).detach()
        positive_labels = labels_tensor[train_index]
        sample_weights = 1.0 + error_weight_alpha * positive_labels * torch.pow(1.0 - prior_probability, error_weight_gamma)
        sample_weights = sample_weights / sample_weights.mean().clamp_min(1e-8)
    else:
        sample_weights = torch.ones(len(train_ids), dtype=torch.float32, device=device)

    def logits_for(graph):
        if prior_tensor is None:
            return model(all_sequences, graph)
        return model(all_sequences, graph, prior_logits=prior_tensor)

    val_tensor = torch.as_tensor(labels[val_ids], dtype=torch.float32, device="cpu").numpy()
    best_ap, best_val_loss, best_state, stale, best_epoch = -1.0, float("inf"), None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_logits = logits_for(graph_train)[train_index]
        loss = (loss_fn(train_logits, labels_tensor[train_index]) * sample_weights).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        model.eval()
        with torch.no_grad():
            val_logits = logits_for(graph_val)[val_ids]
            val_score = torch.sigmoid(val_logits).detach().cpu().numpy()
            val_loss = float(torch.nn.functional.binary_cross_entropy_with_logits(
                val_logits,
                labels_tensor[torch.as_tensor(val_ids, dtype=torch.long, device=device)],
            ).item()) if validation_loss_tiebreak else float("inf")
        ap = float(average_precision_score(val_tensor, val_score))
        ap_improved = ap > best_ap + 1e-7
        loss_tiebreak_improved = validation_loss_tiebreak and abs(ap - best_ap) <= 1e-7 and val_loss < best_val_loss - 1e-7
        if ap_improved or loss_tiebreak_improved:
            best_ap = max(best_ap, ap)
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= patience and epoch >= min_epochs:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.selected_epoch = best_epoch
    model.eval()
    with torch.no_grad():
        val_score = torch.sigmoid(logits_for(graph_val)).detach().cpu().numpy()[val_ids]
        test_score = torch.sigmoid(logits_for(graph_test)).detach().cpu().numpy()[test_ids]
    threshold = choose_recall_threshold(val_tensor, val_score, recall_target)

    if refit_ids is not None:
        if refit_sequences is None or refit_train_graph is None or refit_test_graph is None or initial_state is None:
            raise ValueError("Full train-plus-validation refitting requires sequences and train/test refit graphs.")
        model.load_state_dict(initial_state)
        refit_index = torch.as_tensor(refit_ids, dtype=torch.long, device=device)
        refit_sequences_tensor = torch.as_tensor(refit_sequences, dtype=torch.float32, device=device)
        refit_prior_tensor = torch.as_tensor(refit_prior_logits, dtype=torch.float32, device=device) if refit_prior_logits is not None else None
        refit_positive = max(float(labels[refit_ids].sum()), 1.0)
        refit_negative = max(float(len(refit_ids) - labels[refit_ids].sum()), 1.0)
        refit_pos_weight = (refit_negative / refit_positive) * positive_weight_multiplier if class_weight else 1.0
        refit_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(refit_pos_weight, dtype=torch.float32, device=device), reduction="none")
        if refit_prior_tensor is not None:
            prior_probability = torch.sigmoid(refit_prior_tensor[refit_index]).detach()
            positive_labels = labels_tensor[refit_index]
            refit_sample_weights = 1.0 + error_weight_alpha * positive_labels * torch.pow(1.0 - prior_probability, error_weight_gamma)
            refit_sample_weights = refit_sample_weights / refit_sample_weights.mean().clamp_min(1e-8)
        else:
            refit_sample_weights = torch.ones(len(refit_ids), dtype=torch.float32, device=device)

        refit_optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        refit_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(refit_optimizer, T_max=max(best_epoch, 1)) if cosine_schedule else None
        for _epoch in range(best_epoch):
            model.train()
            refit_optimizer.zero_grad(set_to_none=True)
            if refit_prior_tensor is None:
                refit_logits = model(refit_sequences_tensor, refit_train_graph)
            else:
                refit_logits = model(refit_sequences_tensor, refit_train_graph, prior_logits=refit_prior_tensor)
            refit_loss = (refit_loss_fn(refit_logits[refit_index], labels_tensor[refit_index]) * refit_sample_weights).mean()
            refit_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            refit_optimizer.step()
            if refit_scheduler is not None:
                refit_scheduler.step()
        model.eval()
        with torch.no_grad():
            if refit_prior_tensor is None:
                refit_test_logits = model(refit_sequences_tensor, refit_test_graph)
            else:
                refit_test_logits = model(refit_sequences_tensor, refit_test_graph, prior_logits=refit_prior_tensor)
            test_score = torch.sigmoid(refit_test_logits).detach().cpu().numpy()[test_ids]
    return val_score, test_score, threshold


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no GPU is available.")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    tuning = json.loads(Path(args.tuned_config).read_text(encoding="utf-8")) if args.tuned_config else None
    tuned_families = tuning.get("families", {}) if tuning else {}

    data_path = Path(args.data_file)
    frame, y, feature_columns = read_arff(data_path)
    raw_values = frame.to_numpy(dtype=np.float64, copy=True)
    raw_values[~np.isfinite(raw_values)] = np.nan
    train_ids, val_ids, test_ids, unique_feature_groups = grouped_splits(raw_values, y, args.seed)
    refit_ids = np.concatenate([train_ids, val_ids])
    feature_groups = financial_relation_groups(feature_columns)

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = RobustScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(raw_values[train_ids])).astype(np.float32)
    x_val = scaler.transform(imputer.transform(raw_values[val_ids])).astype(np.float32)
    x_test = scaler.transform(imputer.transform(raw_values[test_ids])).astype(np.float32)
    x_all = scaler.transform(imputer.transform(raw_values)).astype(np.float32)
    refit_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    refit_scaler = RobustScaler()
    x_trainval_refit = refit_scaler.fit_transform(refit_imputer.fit_transform(raw_values[refit_ids])).astype(np.float32)
    x_all_refit = refit_scaler.transform(refit_imputer.transform(raw_values)).astype(np.float32)

    model_factories = {
        "logistic_regression": lambda: LogisticRegression(max_iter=3000, solver="liblinear", class_weight="balanced", random_state=args.seed),
        "linear_discriminant_analysis": lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "decision_tree": lambda: DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=args.seed),
        "random_forest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=args.seed),
        "extra_trees": lambda: ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=args.seed),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, l2_regularization=1.0, random_state=args.seed),
        "rbf_svm": lambda: CalibratedClassifierCV(SVC(C=1.0, kernel="rbf", class_weight="balanced"), method="sigmoid", cv=3),
    }
    train_positive = max(int(y[train_ids].sum()), 1)
    train_negative = max(int(len(train_ids) - y[train_ids].sum()), 1)
    lgbm_base_parameters: dict[str, object] | None = None
    lgbm_positive_weight_multiplier = 1.0
    lgbm_prior_base_parameters: dict[str, object] | None = None
    lgbm_prior_positive_weight_multiplier = 1.0
    if XGBClassifier is not None:
        xgb_params = dict(tuned_families.get("xgboost", {}).get("parameters") or {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.04, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 2.0, "reg_alpha": 0.0, "min_child_weight": 1})
        xgb_params["scale_pos_weight"] = (train_negative / train_positive) * float(xgb_params.pop("scale_pos_weight_multiplier", 1.0))
        model_factories["xgboost"] = lambda: XGBClassifier(**xgb_params, eval_metric="aucpr", tree_method="hist", n_jobs=-1, random_state=args.seed)
    if LGBMClassifier is not None:
        lgbm_params = dict(tuned_families.get("lightgbm", {}).get("parameters") or {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 15})
        lgbm_positive_weight_multiplier = float(lgbm_params.pop("scale_pos_weight_multiplier", 1.0))
        lgbm_base_parameters = dict(lgbm_params)
        lgbm_params["scale_pos_weight"] = (train_negative / train_positive) * lgbm_positive_weight_multiplier
        model_factories["lightgbm"] = lambda: LGBMClassifier(**lgbm_params, objective="binary", verbosity=-1, n_jobs=-1, random_state=args.seed)
        prior_tuning = tuned_families.get("lightgbm_missingness_prior", {}).get("parameters")
        if prior_tuning:
            prior_params = dict(prior_tuning)
            lgbm_prior_positive_weight_multiplier = float(prior_params.pop("scale_pos_weight_multiplier", 1.0))
            lgbm_prior_base_parameters = prior_params
        else:
            lgbm_prior_base_parameters = dict(lgbm_base_parameters)
            lgbm_prior_positive_weight_multiplier = lgbm_positive_weight_multiplier
    if CatBoostClassifier is not None:
        catboost_params = dict(tuned_families.get("catboost", {}).get("parameters") or {
            "iterations": 500, "depth": 6, "learning_rate": 0.04, "l2_leaf_reg": 5.0,
            "random_strength": 0.5, "scale_pos_weight_multiplier": 1.0,
        })
        catboost_weight_multiplier = float(catboost_params.pop("scale_pos_weight_multiplier", 1.0))
        catboost_params["scale_pos_weight"] = (train_negative / train_positive) * catboost_weight_multiplier
        model_factories["catboost"] = lambda: CatBoostClassifier(
            **catboost_params, loss_function="Logloss", eval_metric="PRAUC", verbose=False,
            allow_writing_files=False, thread_count=-1, random_seed=args.seed,
        )
    results: dict[str, dict] = {}
    predictions: list[pd.DataFrame] = []
    fitted_models: dict[str, object] = {}
    model_scores: dict[str, dict[str, np.ndarray]] = {}
    y_train, y_val, y_test = y[train_ids], y[val_ids], y[test_ids]

    for name, factory in model_factories.items():
        start = time.perf_counter()
        model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), RobustScaler(), factory())
        if name == "hist_gradient_boosting":
            model.fit(raw_values[train_ids], y_train, histgradientboostingclassifier__sample_weight=compute_sample_weight("balanced", y_train))
        else:
            model.fit(raw_values[train_ids], y_train)
        val_score = model.predict_proba(raw_values[val_ids])[:, 1]
        threshold = choose_recall_threshold(y_val, val_score, args.recall_target)
        refit_labels = y[refit_ids]
        if name == "hist_gradient_boosting":
            model.fit(raw_values[refit_ids], refit_labels, histgradientboostingclassifier__sample_weight=compute_sample_weight("balanced", refit_labels))
        else:
            model.fit(raw_values[refit_ids], refit_labels)
        test_score = model.predict_proba(raw_values[test_ids])[:, 1]
        fitted_models[name] = model
        model_scores[name] = {"validation": val_score, "test": test_score}
        results[name] = binary_metrics(y_test, test_score, threshold)
        val_metrics = binary_metrics(y_val, val_score, threshold)
        results[name].update({"validation_pr_auc": val_metrics["pr_auc"], "validation_roc_auc": val_metrics["roc_auc"], "fit_seconds": round(time.perf_counter() - start, 3)})
        predictions.append(pd.DataFrame({"row_id": test_ids, "label": y_test, "score": test_score, "threshold": threshold, "model": name}))
        print(f"{name} | validation PR-AUC: {val_metrics['pr_auc']} | test PR-AUC: {results[name]['pr_auc']}")

    mlp = TabularMLP(x_train.shape[1]).to(device)
    initial_mlp_state = {key: value.detach().cpu().clone() for key, value in mlp.state_dict().items()}
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    positive = max(int(y_train.sum()), 1)
    negative = max(len(y_train) - positive, 1)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive, dtype=torch.float32, device=device))
    tx_train = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    ty_train = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    tx_val = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    tx_test = torch.as_tensor(x_test, dtype=torch.float32, device=device)
    best_ap, best_state, stale, best_epoch = -1.0, None, 0, 0
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        mlp.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(mlp(tx_train), ty_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mlp.parameters(), 2.0)
        optimizer.step()
        mlp.eval()
        with torch.no_grad():
            val_score = torch.sigmoid(mlp(tx_val)).cpu().numpy()
        ap = float(average_precision_score(y_val, val_score))
        print(f"mlp epoch {epoch}/{args.epochs} | loss: {loss.item():.4f} | validation PR-AUC: {ap:.4f}")
        if ap > best_ap:
            best_ap = ap
            best_state = {key: value.detach().cpu().clone() for key, value in mlp.state_dict().items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is not None:
        mlp.load_state_dict(best_state)
    mlp.eval()
    with torch.no_grad():
        val_score = torch.sigmoid(mlp(tx_val)).cpu().numpy()
    threshold = choose_recall_threshold(y_val, val_score, args.recall_target)
    mlp.load_state_dict(initial_mlp_state)
    ty_trainval_refit = torch.as_tensor(y[refit_ids], dtype=torch.float32, device=device)
    tx_trainval_refit = torch.as_tensor(x_trainval_refit, dtype=torch.float32, device=device)
    tx_test_refit = torch.as_tensor(x_all_refit[test_ids], dtype=torch.float32, device=device)
    positive_refit = max(int(y[refit_ids].sum()), 1)
    negative_refit = max(len(refit_ids) - positive_refit, 1)
    refit_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative_refit / positive_refit, dtype=torch.float32, device=device))
    refit_optimizer = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    for _epoch in range(best_epoch):
        mlp.train()
        refit_optimizer.zero_grad(set_to_none=True)
        refit_loss = refit_loss_fn(mlp(tx_trainval_refit), ty_trainval_refit)
        refit_loss.backward()
        torch.nn.utils.clip_grad_norm_(mlp.parameters(), 2.0)
        refit_optimizer.step()
    mlp.eval()
    with torch.no_grad():
        test_score = torch.sigmoid(mlp(tx_test_refit)).cpu().numpy()
    results["mlp"] = binary_metrics(y_test, test_score, threshold)
    val_metrics = binary_metrics(y_val, val_score, threshold)
    results["mlp"].update({"validation_pr_auc": val_metrics["pr_auc"], "validation_roc_auc": val_metrics["roc_auc"], "fit_seconds": round(time.perf_counter() - start, 3)})
    predictions.append(pd.DataFrame({"row_id": test_ids, "label": y_test, "score": test_score, "threshold": threshold, "model": "mlp"}))

    # Single-snapshot graph baselines. Edges originate only from training nodes;
    # validation/test nodes query training peers, so holdout nodes never message train nodes.
    gnn_params = tuned_families.get("lstm_gnn", {}).get("parameters") or {}
    gnn_neighbors = int(gnn_params.get("neighbors", args.neighbors))
    gnn_hidden = int(gnn_params.get("hidden_dim", args.hidden_dim))
    gnn_dropout = float(gnn_params.get("dropout", args.dropout))
    graph_train = build_inductive_graph(x_all, train_ids, np.zeros(0, dtype=np.int64), feature_columns, feature_groups, k=gnn_neighbors, weighted=True)
    graph_val = build_inductive_graph(x_all, train_ids, val_ids, feature_columns, feature_groups, k=gnn_neighbors, weighted=True)
    graph_test = build_inductive_graph(x_all, train_ids, test_ids, feature_columns, feature_groups, k=gnn_neighbors, weighted=True)
    graph_train_unweighted = build_inductive_graph(x_all, train_ids, np.zeros(0, dtype=np.int64), feature_columns, feature_groups, k=gnn_neighbors, weighted=False)
    graph_val_unweighted = build_inductive_graph(x_all, train_ids, val_ids, feature_columns, feature_groups, k=gnn_neighbors, weighted=False)
    graph_test_unweighted = build_inductive_graph(x_all, train_ids, test_ids, feature_columns, feature_groups, k=gnn_neighbors, weighted=False)
    graph_train_k5 = build_inductive_graph(x_all, train_ids, np.zeros(0, dtype=np.int64), feature_columns, feature_groups, k=5, weighted=True)
    graph_val_k5 = build_inductive_graph(x_all, train_ids, val_ids, feature_columns, feature_groups, k=5, weighted=True)
    graph_test_k5 = build_inductive_graph(x_all, train_ids, test_ids, feature_columns, feature_groups, k=5, weighted=True)
    graph_train_refit = build_inductive_graph(x_all_refit, refit_ids, np.zeros(0, dtype=np.int64), feature_columns, feature_groups, k=gnn_neighbors, weighted=True)
    graph_test_refit = build_inductive_graph(x_all_refit, refit_ids, test_ids, feature_columns, feature_groups, k=gnn_neighbors, weighted=True)
    graph_train_unweighted_refit = build_inductive_graph(x_all_refit, refit_ids, np.zeros(0, dtype=np.int64), feature_columns, feature_groups, k=gnn_neighbors, weighted=False)
    graph_test_unweighted_refit = build_inductive_graph(x_all_refit, refit_ids, test_ids, feature_columns, feature_groups, k=gnn_neighbors, weighted=False)
    graph_train_k5_refit = build_inductive_graph(x_all_refit, refit_ids, np.zeros(0, dtype=np.int64), feature_columns, feature_groups, k=5, weighted=True)
    graph_test_k5_refit = build_inductive_graph(x_all_refit, refit_ids, test_ids, feature_columns, feature_groups, k=5, weighted=True)
    sequences = x_all[:, None, :]
    sequences_refit = x_all_refit[:, None, :]
    missingness = np.isnan(raw_values).astype(np.float32)[:, None, :]
    prior_sequences = np.concatenate([sequences, missingness], axis=-1)
    prior_sequences_refit = np.concatenate([sequences_refit, missingness], axis=-1)
    relations = list(feature_groups)
    graph_specs = [
        ("lstm_only", lambda: TemporalOnly(len(feature_columns), args.hidden_dim, dropout=args.dropout), graph_train, graph_val, graph_test, graph_train_refit, graph_test_refit),
        ("gcn", lambda: TemporalGCN(len(feature_columns), args.hidden_dim, mode="gcn", dropout=args.dropout), graph_train, graph_val, graph_test, graph_train_refit, graph_test_refit),
        ("graphsage", lambda: TemporalGCN(len(feature_columns), args.hidden_dim, mode="sage", dropout=args.dropout), graph_train, graph_val, graph_test, graph_train_refit, graph_test_refit),
        ("lstm_gnn_relation_attention", lambda: ReliabilityTemporalGNN(len(feature_columns), gnn_hidden, relations, dropout=gnn_dropout), graph_train, graph_val, graph_test, graph_train_refit, graph_test_refit),
        ("ablation_unweighted_edges", lambda: ReliabilityTemporalGNN(len(feature_columns), gnn_hidden, relations, dropout=gnn_dropout), graph_train_unweighted, graph_val_unweighted, graph_test_unweighted, graph_train_unweighted_refit, graph_test_unweighted_refit),
        ("ablation_k5_neighbors", lambda: ReliabilityTemporalGNN(len(feature_columns), gnn_hidden, relations, dropout=gnn_dropout), graph_train_k5, graph_val_k5, graph_test_k5, graph_train_k5_refit, graph_test_k5_refit),
    ]
    for relation in relations:
        remaining = [name for name in relations if name != relation]
        graph_specs.append((f"ablation_without_{relation}", lambda remaining=remaining: ReliabilityTemporalGNN(len(feature_columns), gnn_hidden, remaining, dropout=gnn_dropout), filter_graph(graph_train, remaining), filter_graph(graph_val, remaining), filter_graph(graph_test, remaining), filter_graph(graph_train_refit, remaining), filter_graph(graph_test_refit, remaining)))
    for offset, (name, factory, train_graph, val_graph, test_graph, train_graph_refit, test_graph_refit) in enumerate(graph_specs, start=1):
        torch.manual_seed(args.seed + offset)
        start = time.perf_counter()
        val_score, test_score, threshold = fit_graph_model(
            factory(), sequences, train_graph, val_graph, test_graph, y, train_ids, val_ids, test_ids, device,
            recall_target=args.recall_target,
            learning_rate=float(gnn_params.get("learning_rate", args.learning_rate)),
            weight_decay=float(gnn_params.get("weight_decay", args.weight_decay)),
            positive_weight_multiplier=float(gnn_params.get("positive_weight_multiplier", 1.0)),
            epochs=args.epochs, patience=args.patience,
            refit_ids=refit_ids, refit_sequences=sequences_refit,
            refit_train_graph=train_graph_refit, refit_test_graph=test_graph_refit,
        )
        results[name] = binary_metrics(y_test, test_score, threshold)
        val_metrics = binary_metrics(y_val, val_score, threshold)
        results[name].update({"validation_pr_auc": val_metrics["pr_auc"], "validation_roc_auc": val_metrics["roc_auc"], "fit_seconds": round(time.perf_counter() - start, 3)})
        predictions.append(pd.DataFrame({"row_id": test_ids, "label": y_test, "score": test_score, "threshold": threshold, "model": name}))
        print(f"{name} | validation PR-AUC: {val_metrics['pr_auc']} | test PR-AUC: {results[name]['pr_auc']}")

    selected_prior_hybrid: dict[str, object] | None = None
    selected_csa_hybrid: dict[str, object] | None = None
    if LGBMClassifier is not None and "lightgbm" in fitted_models and lgbm_prior_base_parameters is not None:
        prior_fit_started = time.perf_counter()
        prior_logits, prior_probabilities, _prior_model = crossfit_lightgbm_prior(
            raw_values, y, train_ids, lgbm_prior_base_parameters,
            lgbm_prior_positive_weight_multiplier, args.seed,
        )
        prior_refit_logits, prior_refit_probabilities, _prior_refit_model = crossfit_lightgbm_prior(
            raw_values, y, refit_ids, lgbm_prior_base_parameters,
            lgbm_prior_positive_weight_multiplier, args.seed + 5000,
        )
        prior_val_score = prior_probabilities[val_ids]
        prior_test_score = prior_refit_probabilities[test_ids]
        prior_threshold = choose_recall_threshold(y_val, prior_val_score, args.recall_target)
        results["lightgbm_missingness_prior"] = binary_metrics(y_test, prior_test_score, prior_threshold)
        prior_val_metrics = binary_metrics(y_val, prior_val_score, prior_threshold)
        results["lightgbm_missingness_prior"].update({
            "validation_pr_auc": prior_val_metrics["pr_auc"],
            "validation_roc_auc": prior_val_metrics["roc_auc"],
            "fit_seconds": round(time.perf_counter() - prior_fit_started, 3),
        })
        predictions.append(pd.DataFrame({"row_id": test_ids, "label": y_test, "score": prior_test_score, "threshold": prior_threshold, "model": "lightgbm_missingness_prior"}))

        if "xgboost" in model_scores:
            xgb_val_score = model_scores["xgboost"]["validation"]
            xgb_test_score = model_scores["xgboost"]["test"]
            blend_candidates = []
            for prior_weight in np.linspace(0.0, 1.0, 21):
                val_score = prior_weight * prior_val_score + (1.0 - prior_weight) * xgb_val_score
                blend_candidates.append((float(average_precision_score(y_val, val_score)), float(prior_weight), val_score))
            _, selected_prior_weight, blend_val_score = max(blend_candidates, key=lambda candidate: candidate[0])
            blend_test_score = selected_prior_weight * prior_test_score + (1.0 - selected_prior_weight) * xgb_test_score
            blend_threshold = choose_recall_threshold(y_val, blend_val_score, args.recall_target)
            results["missingness_prior_xgboost_blend"] = binary_metrics(y_test, blend_test_score, blend_threshold)
            blend_val_metrics = binary_metrics(y_val, blend_val_score, blend_threshold)
            results["missingness_prior_xgboost_blend"].update({
                "validation_pr_auc": blend_val_metrics["pr_auc"],
                "validation_roc_auc": blend_val_metrics["roc_auc"],
                "selected_prior_weight": selected_prior_weight,
            })
            predictions.append(pd.DataFrame({"row_id": test_ids, "label": y_test, "score": blend_test_score, "threshold": blend_threshold, "model": "missingness_prior_xgboost_blend"}))
        if "catboost" in model_scores:
            cat_val_score = model_scores["catboost"]["validation"]
            cat_test_score = model_scores["catboost"]["test"]
            blend_candidates = []
            for prior_weight in np.linspace(0.0, 1.0, 21):
                val_score = prior_weight * prior_val_score + (1.0 - prior_weight) * cat_val_score
                blend_candidates.append((float(average_precision_score(y_val, val_score)), float(prior_weight), val_score))
            _, selected_cat_prior_weight, cat_blend_val_score = max(blend_candidates, key=lambda candidate: candidate[0])
            cat_blend_test_score = selected_cat_prior_weight * prior_test_score + (1.0 - selected_cat_prior_weight) * cat_test_score
            cat_blend_threshold = choose_recall_threshold(y_val, cat_blend_val_score, args.recall_target)
            results["missingness_prior_catboost_blend"] = binary_metrics(y_test, cat_blend_test_score, cat_blend_threshold)
            cat_blend_val_metrics = binary_metrics(y_val, cat_blend_val_score, cat_blend_threshold)
            results["missingness_prior_catboost_blend"].update({
                "validation_pr_auc": cat_blend_val_metrics["pr_auc"],
                "validation_roc_auc": cat_blend_val_metrics["roc_auc"],
                "selected_prior_weight": selected_cat_prior_weight,
            })
            predictions.append(pd.DataFrame({"row_id": test_ids, "label": y_test, "score": cat_blend_test_score, "threshold": cat_blend_threshold, "model": "missingness_prior_catboost_blend"}))
        prior_candidates: dict[tuple[float, float, float], tuple[np.ndarray, np.ndarray, float]] = {}
        for max_correction in (0.25, 0.5):
            for alpha in (1.0, 2.0, 4.0):
                for learning_rate in (3e-4, 1e-3):
                    candidate = (max_correction, alpha, learning_rate)
                    torch.manual_seed(args.seed + 500 + len(prior_candidates))
                    model = PriorGuidedReliabilityTemporalGNN(
                        len(feature_columns) * 2, gnn_hidden, relations, dropout=gnn_dropout,
                        adaptive_gate=True, max_correction=max_correction,
                    )
                    val_score, test_score, threshold = fit_graph_model(
                        model, prior_sequences, graph_train, graph_val, graph_test, y, train_ids, val_ids, test_ids, device,
                        recall_target=args.recall_target,
                        learning_rate=learning_rate,
                        weight_decay=1e-4,
                        epochs=min(args.epochs, 30), patience=min(args.patience, 5),
                        prior_logits=prior_logits, error_weight_alpha=alpha, error_weight_gamma=1.0,
                        class_weight=False, cosine_schedule=True, gradient_clip=5.0,
                        validation_loss_tiebreak=True, min_epochs=5,
                    )
                    candidate_ap = float(average_precision_score(y_val, val_score))
                    prior_candidates[candidate] = (val_score, test_score, threshold)
                    print(f"prior-guided LSTM-GNN max_amp={max_correction:g} alpha={alpha:g} lr={learning_rate:g} | validation PR-AUC: {candidate_ap:.4f}")

        selected_config = max(
            prior_candidates,
            key=lambda config: (
                round(float(average_precision_score(y_val, prior_candidates[config][0])), 7),
                -float(np.mean((y_val - prior_candidates[config][0]) ** 2)),
            ),
        )
        selected_max_correction, selected_alpha, selected_learning_rate = selected_config
        selected_val, selected_test, selected_threshold = prior_candidates[selected_config]
        selected_candidate_index = list(prior_candidates).index(selected_config)
        torch.manual_seed(args.seed + 500 + selected_candidate_index)
        selected_prior_model = PriorGuidedReliabilityTemporalGNN(
            len(feature_columns) * 2, gnn_hidden, relations, dropout=gnn_dropout,
            adaptive_gate=True, max_correction=selected_max_correction,
        )
        _, selected_test, _ = fit_graph_model(
            selected_prior_model, prior_sequences, graph_train, graph_val, graph_test,
            y, train_ids, val_ids, test_ids, device,
            recall_target=args.recall_target, learning_rate=selected_learning_rate,
            weight_decay=1e-4, epochs=min(args.epochs, 30), patience=min(args.patience, 5),
            prior_logits=prior_logits, error_weight_alpha=selected_alpha, error_weight_gamma=1.0,
            class_weight=False, cosine_schedule=True, gradient_clip=5.0,
            validation_loss_tiebreak=True, min_epochs=5,
            refit_ids=refit_ids, refit_sequences=prior_sequences_refit,
            refit_train_graph=graph_train_refit, refit_test_graph=graph_test_refit,
            refit_prior_logits=prior_refit_logits,
        )
        selected_prior_hybrid = {
            "error_weight_alpha": selected_alpha,
            "max_correction": selected_max_correction,
            "learning_rate": selected_learning_rate,
            "epochs_max": min(args.epochs, 30),
            "selected_epoch": getattr(selected_prior_model, "selected_epoch", None),
            "validation_pr_auc": float(average_precision_score(y_val, selected_val)),
            "prior_model": "LightGBM with grouped out-of-fold training logits",
        }
        results["lgbm_prior_lstm_gnn"] = binary_metrics(y_test, selected_test, selected_threshold)
        selected_val_metrics = binary_metrics(y_val, selected_val, selected_threshold)
        results["lgbm_prior_lstm_gnn"].update({"validation_pr_auc": selected_val_metrics["pr_auc"], "validation_roc_auc": selected_val_metrics["roc_auc"]})
        predictions.append(pd.DataFrame({"row_id": test_ids, "label": y_test, "score": selected_test, "threshold": selected_threshold, "model": "lgbm_prior_lstm_gnn"}))

        alpha0_model = PriorGuidedReliabilityTemporalGNN(
            len(feature_columns) * 2, gnn_hidden, relations, dropout=gnn_dropout,
            adaptive_gate=True, max_correction=selected_max_correction,
        )
        alpha0_val, alpha0_test, alpha0_threshold = fit_graph_model(
            alpha0_model, prior_sequences, graph_train, graph_val, graph_test, y, train_ids, val_ids, test_ids, device,
            recall_target=args.recall_target, learning_rate=selected_learning_rate, weight_decay=1e-4,
            epochs=min(args.epochs, 30), patience=min(args.patience, 5), prior_logits=prior_logits,
            error_weight_alpha=0.0, class_weight=False,
            cosine_schedule=True, gradient_clip=5.0,
            refit_ids=refit_ids, refit_sequences=prior_sequences_refit,
            refit_train_graph=graph_train_refit, refit_test_graph=graph_test_refit,
            refit_prior_logits=prior_refit_logits,
        )
        results["ablation_prior_no_cost_reweight"] = binary_metrics(y_test, alpha0_test, alpha0_threshold)
        alpha0_val_metrics = binary_metrics(y_val, alpha0_val, alpha0_threshold)
        results["ablation_prior_no_cost_reweight"].update({"validation_pr_auc": alpha0_val_metrics["pr_auc"], "validation_roc_auc": alpha0_val_metrics["roc_auc"]})
        predictions.append(pd.DataFrame({"row_id": test_ids, "label": y_test, "score": alpha0_test, "threshold": alpha0_threshold, "model": "ablation_prior_no_cost_reweight"}))

        for name, adaptive_gate, use_graph, use_missingness in (
            ("ablation_prior_fixed_gate", False, True, True),
            ("ablation_prior_lstm_only", True, False, True),
            ("ablation_prior_no_missingness", True, True, False),
        ):
            torch.manual_seed(args.seed + 600 + len(results))
            prior_relations = relations if use_graph else []
            prior_input = prior_sequences if use_missingness else sequences
            prior_input_dim = len(feature_columns) * (2 if use_missingness else 1)
            prior_model = PriorGuidedReliabilityTemporalGNN(
                prior_input_dim, gnn_hidden, prior_relations, dropout=gnn_dropout,
                adaptive_gate=adaptive_gate, max_correction=selected_max_correction,
            )
            val_score, test_score, threshold = fit_graph_model(
                prior_model, prior_input,
                graph_train if use_graph else filter_graph(graph_train, []),
                graph_val if use_graph else filter_graph(graph_val, []),
                graph_test if use_graph else filter_graph(graph_test, []),
                y, train_ids, val_ids, test_ids, device,
                recall_target=args.recall_target,
                learning_rate=selected_learning_rate,
                weight_decay=1e-4,
                epochs=min(args.epochs, 30), patience=min(args.patience, 5), prior_logits=prior_logits,
                error_weight_alpha=selected_alpha,
                class_weight=False, cosine_schedule=True, gradient_clip=5.0,
                validation_loss_tiebreak=True, min_epochs=5,
                refit_ids=refit_ids,
                refit_sequences=prior_sequences_refit if use_missingness else sequences_refit,
                refit_train_graph=graph_train_refit if use_graph else filter_graph(graph_train_refit, []),
                refit_test_graph=graph_test_refit if use_graph else filter_graph(graph_test_refit, []),
                refit_prior_logits=prior_refit_logits,
            )
            results[name] = binary_metrics(y_test, test_score, threshold)
            val_metrics = binary_metrics(y_val, val_score, threshold)
            results[name].update({"validation_pr_auc": val_metrics["pr_auc"], "validation_roc_auc": val_metrics["roc_auc"]})
            predictions.append(pd.DataFrame({"row_id": test_ids, "label": y_test, "score": test_score, "threshold": threshold, "model": name}))
            print(f"{name} | validation PR-AUC: {val_metrics['pr_auc']} | test PR-AUC: {results[name]['pr_auc']}")

        csa_candidates: dict[tuple[float, float, float], tuple[np.ndarray, np.ndarray, float]] = {}
        for max_correction in (0.25, 0.5):
            for alpha in (1.0, 2.0, 4.0):
                for learning_rate in (3e-4, 1e-3):
                    candidate = (max_correction, alpha, learning_rate)
                    torch.manual_seed(args.seed + 800 + len(csa_candidates))
                    model = PriorGuidedTabularResidual(
                        len(feature_columns) * 2, dropout=0.2,
                        adaptive_gate=True, max_correction=max_correction,
                    )
                    val_score, test_score, threshold = fit_graph_model(
                        model, prior_sequences, graph_train, graph_val, graph_test,
                        y, train_ids, val_ids, test_ids, device,
                        recall_target=args.recall_target, learning_rate=learning_rate,
                        weight_decay=1e-4, epochs=min(args.epochs, 30), patience=min(args.patience, 5),
                        prior_logits=prior_logits, error_weight_alpha=alpha, error_weight_gamma=1.0,
                        class_weight=False, cosine_schedule=True, gradient_clip=5.0,
                        validation_loss_tiebreak=True, min_epochs=5,
                    )
                    candidate_ap = float(average_precision_score(y_val, val_score))
                    csa_candidates[candidate] = (val_score, test_score, threshold)
                    print(f"prior-guided MLP max_amp={max_correction:g} alpha={alpha:g} lr={learning_rate:g} | validation PR-AUC: {candidate_ap:.4f}")

        selected_csa_config = max(
            csa_candidates,
            key=lambda config: (
                round(float(average_precision_score(y_val, csa_candidates[config][0])), 7),
                -float(np.mean((y_val - csa_candidates[config][0]) ** 2)),
            ),
        )
        selected_csa_amplitude, selected_csa_alpha, selected_csa_learning_rate = selected_csa_config
        csa_val, csa_test, csa_threshold = csa_candidates[selected_csa_config]
        selected_csa_candidate_index = list(csa_candidates).index(selected_csa_config)
        torch.manual_seed(args.seed + 800 + selected_csa_candidate_index)
        selected_csa_model = PriorGuidedTabularResidual(
            len(feature_columns) * 2, dropout=0.2,
            adaptive_gate=True, max_correction=selected_csa_amplitude,
        )
        _, csa_test, _ = fit_graph_model(
            selected_csa_model, prior_sequences, graph_train, graph_val, graph_test,
            y, train_ids, val_ids, test_ids, device,
            recall_target=args.recall_target, learning_rate=selected_csa_learning_rate,
            weight_decay=1e-4, epochs=min(args.epochs, 30), patience=min(args.patience, 5),
            prior_logits=prior_logits, error_weight_alpha=selected_csa_alpha, error_weight_gamma=1.0,
            class_weight=False, cosine_schedule=True, gradient_clip=5.0,
            validation_loss_tiebreak=True, min_epochs=5,
            refit_ids=refit_ids, refit_sequences=prior_sequences_refit,
            refit_train_graph=graph_train_refit, refit_test_graph=graph_test_refit,
            refit_prior_logits=prior_refit_logits,
        )
        selected_csa_hybrid = {
            "error_weight_alpha": selected_csa_alpha,
            "max_correction": selected_csa_amplitude,
            "learning_rate": selected_csa_learning_rate,
            "epochs_max": min(args.epochs, 30),
            "selected_epoch": getattr(selected_csa_model, "selected_epoch", None),
            "validation_pr_auc": float(average_precision_score(y_val, csa_val)),
            "prior_model": "Missingness-aware LightGBM with grouped out-of-fold training logits",
        }
        results["csa_style_lgbm_residual"] = binary_metrics(y_test, csa_test, csa_threshold)
        csa_val_metrics = binary_metrics(y_val, csa_val, csa_threshold)
        results["csa_style_lgbm_residual"].update({"validation_pr_auc": csa_val_metrics["pr_auc"], "validation_roc_auc": csa_val_metrics["roc_auc"]})
        predictions.append(pd.DataFrame({"row_id": test_ids, "label": y_test, "score": csa_test, "threshold": csa_threshold, "model": "csa_style_lgbm_residual"}))

        for name, adaptive_gate, use_missingness, alpha in (
            ("ablation_csa_no_cost", True, True, 0.0),
            ("ablation_csa_fixed_gate", False, True, selected_csa_alpha),
            ("ablation_csa_no_missingness", True, False, selected_csa_alpha),
        ):
            torch.manual_seed(args.seed + 900 + len(results))
            model_input = prior_sequences if use_missingness else sequences
            input_dim = len(feature_columns) * (2 if use_missingness else 1)
            model = PriorGuidedTabularResidual(
                input_dim, dropout=0.2, adaptive_gate=adaptive_gate,
                max_correction=selected_csa_amplitude,
            )
            val_score, test_score, threshold = fit_graph_model(
                model, model_input, graph_train, graph_val, graph_test,
                y, train_ids, val_ids, test_ids, device,
                recall_target=args.recall_target, learning_rate=selected_csa_learning_rate,
                weight_decay=1e-4, epochs=min(args.epochs, 30), patience=min(args.patience, 5),
                prior_logits=prior_logits, error_weight_alpha=alpha, error_weight_gamma=1.0,
                class_weight=False, cosine_schedule=True, gradient_clip=5.0,
                validation_loss_tiebreak=True, min_epochs=5,
                refit_ids=refit_ids,
                refit_sequences=prior_sequences_refit if use_missingness else sequences_refit,
                refit_train_graph=graph_train_refit, refit_test_graph=graph_test_refit,
                refit_prior_logits=prior_refit_logits,
            )
            results[name] = binary_metrics(y_test, test_score, threshold)
            val_metrics = binary_metrics(y_val, val_score, threshold)
            results[name].update({"validation_pr_auc": val_metrics["pr_auc"], "validation_roc_auc": val_metrics["roc_auc"]})
            predictions.append(pd.DataFrame({"row_id": test_ids, "label": y_test, "score": test_score, "threshold": threshold, "model": name}))
            print(f"{name} | validation PR-AUC: {val_metrics['pr_auc']} | test PR-AUC: {results[name]['pr_auc']}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(predictions, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)
    report = {
        "source": "UCI Polish Companies Bankruptcy, 5year.arff",
        "citation": "Tomczak, S. (2016). Polish Companies Bankruptcy. UCI Machine Learning Repository. https://doi.org/10.24432/C5F600",
        "license": "CC BY 4.0",
        "forecast_horizon": "One-year-ahead subset (5year.arff)",
        "evaluation": "Stratified grouped 60/20/20 holdout; identical feature rows are kept within one split. Model selection and early stopping use validation average precision; final models refit on train plus validation rows before test scoring.",
        "temporal_graph_evaluation": False,
        "evaluation_scope": "The LSTM-GNN is evaluated as a one-snapshot inductive peer graph model. Its LSTM receives one observed vector, so this experiment does not test temporal modeling.",
        "limitations": ["The supplied files contain no company identifiers or fiscal-year keys, so they cannot support longitudinal sequences, chronological splits, or company-clustered intervals.", "Graph ablations test edge attention, unweighted edges, neighbor count, and relation channels; temporal persistence is not identifiable from this single-snapshot dataset.", "When per-seed tuning is enabled, candidate selection uses only that split's validation partition; otherwise the supplied tuned config is reused. The selected models refit on train plus validation rows after selection. Earlier exploratory benchmark test records informed architecture development, so these results are not a confirmatory blinded evaluation. Seed variability is not a confidence interval and does not establish state-of-the-art performance."],
        "systematic_hyperparameter_search": tuning is not None,
        "confidence_intervals": "Not estimated; no company identifiers are available for clustered resampling.",
        "seed": args.seed,
        "device": str(device),
        "rows": int(len(y)),
        "features": feature_columns,
        "relation_feature_groups": feature_groups,
        "selected_tuning_config": tuning,
        "selected_prior_hybrid": selected_prior_hybrid,
        "selected_csa_hybrid": selected_csa_hybrid,
        "unique_feature_groups": unique_feature_groups,
        "class_counts": {str(int(k)): int(v) for k, v in pd.Series(y).value_counts().sort_index().items()},
        "split_rows": {"train": int(len(train_ids)), "validation": int(len(val_ids)), "test": int(len(test_ids))},
        "split_positive_counts": {"train": int(y_train.sum()), "validation": int(y_val.sum()), "test": int(y_test.sum())},
        "models": results,
        "unavailable_optional_baselines": [name for name, package in (("xgboost", XGBClassifier), ("lightgbm", LGBMClassifier), ("catboost", CatBoostClassifier)) if package is None],
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Device: {device}; wrote benchmark outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
