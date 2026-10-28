# Reliability-Aware Temporal Peer Graphs for Corporate Distress Prediction

**Authors:** Atharv Shanker, Vedang Pandey, Shoumik Mitra
**Status:** Research scaffold with exploratory cross-sectional benchmark results; temporal success criterion not met.

## Abstract

Corporate distress prediction combines a firm's financial history with information about related firms, but similarity graphs can change over time and may contain unreliable links. This project studies whether relation-specific, reliability-weighted temporal peer graphs improve future-period bankruptcy prediction beyond strong tabular and temporal baselines. The implemented model has a recurrent encoder and relation-specific peer graph attention, while the longer-term design targets multiple annual snapshots and persistence-weighted edges. An exploratory UCI Polish one-year benchmark is reported, but its records lack firm identifiers and years; that experiment therefore evaluates only a one-step cross-sectional model and cannot test temporal learning. **The benchmark does not establish that the proposed graph architecture improves on its tree prior or achieves state-of-the-art performance.**

## 1. Introduction

Financial distress develops over time and may co-vary among firms with related financial profiles or industry exposure. The project asks whether those peer relationships add measurable predictive value under realistic out-of-time evaluation. The contribution, if supported after a literature audit, is the explicit modeling and stress-testing of peer-edge reliability over temporal graph snapshots—not the general combination of recurrent and graph neural networks.

## 2. Data and prediction task

The candidate U.S. public-company dataset spans 1999–2018 and includes 18 financial variables and broad industry fields. Its released target is constant within each firm across all years, and its CSV firm count differs from the paper's reported count. It cannot support a one-year-ahead forecasting evaluation as provided. Separately, the UCI Polish `5year.arff` subset contains 5,910 rows, 64 financial attributes, and 410 positive labels. It is a one-year-ahead cross-sectional task, but the supplied files contain no company identifiers or fiscal-year keys, so its repeated grouped holdouts cannot support chronological or firm-disjoint evaluation. A valid longitudinal panel with verified event dates, feature availability dates, and complete negative follow-up is still needed for the paper's temporal claim.

## 3. Method

For each firm and prediction year, an LSTM encodes the preceding five fiscal years. Snapshot graphs represent financial-peer relations by feature group and optional industry membership. Edge confidence combines feature similarity with neighborhood persistence between consecutive snapshots. Relation-specific message passing is fused with the firm's temporal state and a self-only representation for binary classification.

## 4. Evaluation protocol

For a valid longitudinal panel, use chronological train, validation, and test periods, plus a company-disjoint generalization analysis. Fit every transformation on training data only. Select checkpoints, calibration, and operating thresholds using validation data. On the Polish subset, use five duplicate-grouped random holdouts and describe the model as a one-snapshot peer graph. The rationale is that distress may be better characterized by multiyear changes and that financially similar peers may provide context; persistence weighting is intended to reduce reliance on transient links. These are hypotheses, not established benefits.

Compare regularized logistic regression, linear discriminant analysis, decision tree, SVM, random forest, ExtraTrees, histogram gradient boosting, XGBoost, and LightGBM on current-year features and flattened five-year histories; include Altman Z-score when fields permit and survival/hazard methods only when event and censoring data support them. Compare MLP on both feature views, LSTM/GRU, GCN, GraphSAGE, GAT, and dynamic relation-specific graph variants. All models receive the same underlying history and point-in-time information. Report PR-AUC and ROC-AUC, precision at a validation-selected 80%-recall operating point, Brier score, calibration, and confusion matrices. Use paired company-clustered confidence intervals and repeated-seed results. Ablate temporal history, relation channels, persistence weighting, and message passing; stress-test edge corruption and graph density.

The project defines success in advance: the proposed method must outperform both the strongest traditional baseline (including a tuned XGBoost or LightGBM model) and the strongest temporal/graph baseline on held-out PR-AUC and ROC-AUC, with paired company-clustered 95% confidence intervals above zero, consistent direction across declared seeds and evaluable future windows, and no failure of operating-point or calibration guardrails. If this gate fails, the method has not met the project's success criterion; the gate will not be relaxed after seeing test results. See `docs/comparison-framework.md` for the complete protocol and current implementation gaps.

## 5. Results and limitations

On Kaggle T4, version 11's best mean test PR-AUC was 0.8148 ± 0.0246 for both the missingness-aware LightGBM prior and a CSA-style LightGBM residual. The prior-guided LSTM-GNN scored 0.8147 ± 0.0245, while the standalone one-step LSTM-GNN scored 0.3577 ± 0.0588. The run used five fresh duplicate-grouped holdouts and three-fold grouped inner tuning for XGBoost, LightGBM, CatBoost, the prior, and standalone LSTM-GNN; the residual and pairwise blend were selected on validation. Means and standard deviations describe split variability, not confidence intervals. Residual and graph mechanisms were effectively tied with the tree prior, so the results do not support a graph or neural correction gain. A published CSA-Net benchmark reports 0.824 ± 0.018 on the one-year Polish horizon under a different nested five-fold protocol ([Wang, Lan & Jie, *Scientific Reports*, 2026](https://www.nature.com/articles/s41598-026-68005-8)); this project's best mean is about 0.009 lower and not directly comparable. The UCI files do not identify firms or years, preventing company-clustered inference and any claim of temporal benefit. Similarity relationships are observational and do not establish risk contagion or causal influence. The A* workshop target is not yet selected; see [`docs/polish-benchmark.md`](../docs/polish-benchmark.md) for the full benchmark and limitations.

## References

References and venue-specific formatting will be completed after the novelty audit and dataset choice are finalized.
