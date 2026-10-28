# Baselines, rationale, and paper success gate

## Research claim being tested

The proposed method is a relation-specific temporal GNN whose edges represent financial similarity and whose edge weights combine current similarity with neighborhood persistence. The claim is that this representation improves point-in-time future distress prediction beyond strong traditional classifiers and ordinary temporal/graph methods. This is a hypothesis, not a result. GNNs for bankruptcy prediction and temporal relational GNNs for distress already exist, so “we used an LSTM and GNN together” is not a novelty claim. [ComRisk (2022)](https://arxiv.org/abs/2202.03874), [RA-HTGNN (2026)](https://www.sciencedirect.com/science/article/pii/S1877050926027055)

## Why test temporal and peer information

| Component | Reason it may add predictive information | How we test that reason |
|---|---|---|
| Five-year financial sequence | Deterioration, volatility, and recovery can be more informative than the latest snapshot | Compare current-year baselines to the same models on flattened five-year features, then compare flat history to LSTM/GRU |
| Peer graph | Financially similar firms may share sector, credit, or macroeconomic exposures, giving an individual firm context from peers | Compare LSTM-only to GCN, GraphSAGE, GAT, and relation-specific graph models with identical nodes, covariates, years, and splits |
| Relation-specific channels | Liquidity, leverage, profitability, and cash-flow peer sets may encode different financial states | Compare separate channels to one merged graph and remove one channel at a time |
| Persistence-based edge reliability | A peer relationship that recurs may be less sensitive to noisy features than a one-year nearest-neighbor link | Compare no-persistence, similarity-only, persistence-only, and combined weighting; perturb edges and features |

Every rationale can fail. Boosted trees may capture the useful signal already; a graph may duplicate industry or macroeconomic information; persistence may encode firm size or stable sector membership; and message passing may oversmooth firms. A feature-similarity edge is an observational predictive relation, not evidence of causal contagion. Only mechanisms that pass their ablation and robustness tests can be described as supported.

## Required comparison ladder

The literature makes tree ensembles a high bar: a U.S. public-firm bankruptcy benchmark reports gradient-boosted trees, including XGBoost and LightGBM, as stronger than the other evaluated methods for one-year-ahead prediction. These models are therefore mandatory comparators, not optional baselines. [Alanis, Chava, and Shah](https://arxiv.org/abs/2212.12051)

| Tier | Methods | Input views | What it controls |
|---|---|---|---|
| Classic financial scores | Altman Z-score where its required inputs are available; optionally Ohlson O-score under a documented variable mapping | Published score inputs, no learned graph | Whether complexity beats a recognized financial scoring reference |
| Statistical models | Regularized logistic regression, linear discriminant analysis, penalized discrete-time hazard/logit if the outcome timing supports it | Current-year ratios and a declared raw-item feature set | Linear and classical risk modeling |
| Traditional ML | Decision tree, linear/RBF SVM, random forest, ExtraTrees, histogram gradient boosting, **XGBoost and LightGBM**; CatBoost if categorical fields are used | Current-year features and the same trailing five years flattened | Nonlinear interaction and state-of-practice tabular prediction; XGBoost/LightGBM are required |
| Survival models, only with valid event/censor data | Cox or discrete-time hazard model; random survival forest | Event times, censoring status, point-in-time predictors | Whether explicit time-to-event modeling is better suited than fixed-horizon classification |
| Neural tabular | MLP on current-year and flattened five-year inputs | Same values available to the classical models | Neural capacity without sequence or graph structure |
| Temporal | LSTM, GRU; temporal convolution if feasible | Five-year sequence plus missing-history mask | Value of temporal encoding versus current-year and flat-history controls |
| Conventional graph | GCN, GraphSAGE, GAT, dynamic graph with no persistence, and relation-specific graph without persistence | Same temporal node states and contemporaneous peer graph | Value of standard graph aggregation and relation channels |
| Proposed and mechanism ablations | Reliability-aware relation-specific temporal GNN; no-persistence, similarity-only, persistence-only, merged-relation, leave-one-relation-out, no-message variants | Same inputs and snapshots as the graph controls | Whether the proposed edge-reliability mechanism earns its added complexity |

Survival methods are not forced into an ordinary binary task: they require a valid event time and censoring indicator. A published score on a different dataset or random split is context only. Direct numerical comparison to ComRisk or RA-HTGNN is permitted only if the population, features, target horizon, and test protocol can be reproduced on a common dataset.

## Fair comparison rules

1. **Same cohort and horizon:** every primary model uses the same eligible rows and event definition. If a baseline cannot use a particular field, use a common feature subset and disclose it.
2. **Matched historical information:** all tabular models run on both year-*t* values and the trailing five years flattened with a fixed missing-history mask. Sequence models use the same five years and mask. A graph win must not result merely from having more years of inputs.
3. **Point-in-time construction:** features and graph snapshots use only data available by each forecast date. Filing dates, event dates, status labels, and post-event information are never predictors. Fit imputation, clipping, scaling, and feature selection on training years only.
4. **Class imbalance:** default to training-set class weights. Any undersampling or synthetic resampling is confined to training folds and is a separate reported condition. No validation/test resampling.
5. **Chronological split:** train/validation/test are ordered in time; use expanding-window validation for tuning. Add at least two future test windows if positive-event counts support it. A company-disjoint analysis is additional and does not replace the primary future holdout.
6. **Equal tuning opportunity:** publish search spaces and use at least 50 validation configurations per required GBDT and proposed family, with the same expanding validation folds and a declared compute cap. If resources force a smaller cap, apply it to all models before test access. Choose strongest comparators on validation PR-AUC and freeze their identities before test evaluation.
7. **Reproducibility:** use at least five predeclared seeds for neural/graph models; store configs, checksums, split years, transforms, validation/test predictions, environment versions, and logs. Report every baseline, not only the strongest.

## Metrics and uncertainty

- **Primary metric: PR-AUC / average precision.** Distress is a minority class, so positive-class precision-recall behavior is central.
- **Co-required ranking metric: ROC-AUC.** The project requires improvement here as well; a PR-AUC-only win does not pass.
- **Operating point:** precision at a threshold selected on validation to target 80% recall; report achieved test precision and recall. Freeze the threshold before test evaluation.
- **Probability quality:** Brier score and calibration plot; report calibration slope/intercept when estimable.
- **Context:** class prevalence/counts, confusion matrix, and per-window/per-seed results. Accuracy is supplemental only. If decision-cost analysis is included, declare costs before inspecting test results.

For proposed-versus-comparator differences, compute paired company-clustered 95% confidence intervals: resample firms and retain all their test-year observations together, with at least 10,000 valid bootstrap replicates. Report each seed and future window so pooling cannot hide a failed period. Correct additional confirmatory pairwise tests using a declared procedure; label other analyses exploratory.

## Hard success gate

The paper counts as successful under the user's criterion only if all conditions hold on a valid, audited, point-in-time event-horizon panel:

1. The proposed method's untouched-test PR-AUC exceeds both (a) the strongest traditional model, including a tuned XGBoost or LightGBM model, and (b) the strongest temporal/conventional graph baseline. Baseline identities are frozen using validation performance.
2. For both comparators, the paired company-clustered 95% intervals for PR-AUC differences are entirely above zero; ROC-AUC differences must also have intervals entirely above zero.
3. Both gains retain the same direction across all predeclared seeds and adequately powered future test windows.
4. At the validation-selected operating point, test precision at target recall is not below either comparator. The upper 95% confidence bound for the proposed-minus-comparator Brier score difference must be below +0.01 absolute, and calibration diagnostics must satisfy their prespecified non-deterioration rule.
5. There is no target leakage, test tuning, post hoc baseline removal, or selective reporting.

If any condition fails, the method has not met the required success criterion. Keep the results, diagnose the failure, and improve the method or data before making a success claim. Do not weaken the metric gate, omit a strong baseline, or call a mixed result successful. Passing this empirical gate still does not guarantee acceptance at an A* workshop.

## Current implementation and data status

The chronological panel runner supports logistic regression, decision tree, random forest, histogram gradient boosting, MLP, LSTM, GCN, GraphSAGE, and the proposed relation-aware LSTM-GNN. The separate UCI Polish runner adds shrinkage LDA, ExtraTrees, calibrated RBF SVM, XGBoost, LightGBM, a missingness-aware LightGBM prior, a CSA-style residual MLP, static GCN/GraphSAGE, and single-snapshot edge-attention LSTM-GNN, plus edge, neighbor-count, relation, prior, gate, cost-weight, and missingness ablations. Its current results are documented in [`polish-benchmark.md`](polish-benchmark.md).

The latest completed Polish run (version 11; five fresh grouped holdouts with three-fold grouped inner tuning for the major tree/prior/neural families) reached 0.8148 ± 0.0246 test PR-AUC with the missingness-aware LightGBM prior and CSA-style residual, which were numerically tied. The prior-guided LSTM-GNN scored 0.8147 ± 0.0245; the standalone one-step LSTM-GNN scored 0.3577 ± 0.0588. CatBoost, XGBoost, LightGBM, classical models, graph controls, and mechanism ablations are reported. A published CSA-Net benchmark reports 0.824 ± 0.018 for the one-year horizon under a different nested five-fold protocol, so our experiment does not establish state of the art. The dataset has no company identifiers or year keys, so it cannot test multi-year sequences, chronological splits, or graph persistence. See [`polish-benchmark.md`](polish-benchmark.md) for results and limitations.

The candidate American Stock Market CSV has firm-level labels constant across years and lacks the verified event horizon required for this task. No baseline scores from it count as forecasting evidence, and the success gate cannot yet be evaluated. See [`data-audit.md`](data-audit.md) and [`polish-benchmark.md`](polish-benchmark.md).
