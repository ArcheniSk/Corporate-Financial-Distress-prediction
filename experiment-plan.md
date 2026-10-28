# FME Project: Experimental Plan

## 1. Objective and non-negotiable success criterion

**Research question:** For a firm-year with verified point-in-time financial information, does a time-aware graph model improve prediction of a documented future distress event over the strongest established financial-risk models, and does the proposed edge-reliability mechanism add value beyond an ordinary temporal GNN?

The goal is not to show that a GNN can be trained or that graph features sometimes help. The user's required success condition is empirical superiority. The paper counts as successful under this project only if the proposed model beats the strongest established traditional ML baseline **and** the strongest temporal/graph baseline on untouched out-of-time data, on PR-AUC and ROC-AUC, with paired company-clustered 95% confidence intervals above zero; gains must persist across declared seeds and available future windows; and calibration and operating-point guardrails must pass. The exact selection, inference, and reporting rules are in [`docs/comparison-framework.md`](docs/comparison-framework.md). The gate cannot be weakened after test results are seen.

The current public dataset candidate fails the target audit. No result, superiority claim, or successful-paper claim is currently possible. Recovering a valid event-horizon panel is the first hard dependency, not a paperwork step.

## 2. Why a temporal peer GNN is worth testing

Financial distress can develop through changing liquidity, leverage, profitability, and cash flow. A model using only year *t* may miss deterioration rate, volatility, recovery, or combinations of changes across years; a five-year history could capture these dynamics. Separately, firms with similar financial profiles may share sector conditions, credit-market exposure, or macroeconomic shocks. Their contemporaneous financial state could supply context that the focal firm's history alone does not contain. A graph makes that context explicit and allows it to be aggregated selectively.

The proposed edge reliability idea addresses a particular weakness in feature-similarity graphs: a nearest-neighbor relation can change because of measurement noise, small feature movements, missing values, or the chosen neighbor count. If a peer remains similar across adjacent snapshots, its relation may be more dependable than a link that appears for one year. The proposed mechanism therefore tests whether neighborhood persistence adds useful confidence information to current similarity.

These are plausible mechanisms, not proof that our model should win. Strong boosting models may already capture the predictive information in the features. LSTM may capture all useful temporal signal. Similarity peers may be noisy or redundant. Persistent neighborhoods may simply encode size or industry. A GNN may overfit or smooth away a firm's distinctive distress signal. Each possibility has a control below. Similarity edges are predictive associations; they do not show causal contagion.

| Proposed information source | What it might contribute | Required control that could show it is unnecessary |
|---|---|---|
| Trailing financial sequence | Trends, volatility, persistence, and recovery patterns unavailable from a single snapshot | Current-year models versus matched five-year flattened models, then flattened-history models versus LSTM |
| Peer messages | Context from firms with similar current financial states, including common but unobserved exposures | LSTM-only versus GCN/GraphSAGE/GAT and relation-specific graph model on identical firms, years, features, and splits |
| Financial relation channels | Different peer sets may represent distinct liquidity, leverage, profitability, and cash-flow regimes | One merged graph versus separate relation graphs; leave-one-relation-out tests |
| Link persistence | Downweight transient nearest-neighbor links that may be artifacts of noisy features | Same graph and same model with similarity-only weights, persistence-only weights, combined weights, and no persistence |
| Robust graph aggregation | Reduce sensitivity to incorrect, missing, or unstable peers | Edge deletion/addition, feature perturbation, randomized matched-degree graphs, and graph-density sensitivity |

## 3. Existing evidence and implications for comparisons

Prior bankruptcy work has already compared many traditional statistical and ML classifiers. A broad U.S. bankruptcy benchmark includes penalized hazard/logistic specifications, random forests, survival random forests, XGBoost, LightGBM, and neural networks; it reports gradient-boosted trees as strongest for its one-year-ahead task. Therefore XGBoost/LightGBM-class models are required comparators here, not optional afterthoughts. [Alanis, Chava, and Shah, *Benchmarking Machine Learning Models to Predict Corporate Bankruptcy*](https://arxiv.org/abs/2212.12051)

Graph-based enterprise bankruptcy prediction also predates this project. ComRisk combines firm-level risk with heterogeneous relations and hypergraphs. More recent work describes relation-specific graph convolution, relation attention, recurrent temporal encoding, and one-year-ahead distress prediction. Thus neither “GNN for bankruptcy” nor “LSTM plus GNN” is a defensible novelty claim. The candidate contribution is narrower: whether persistence-based reliability for changing feature-similarity peer edges helps under matched comparisons and graph-noise stress tests. It remains a candidate until equation-level literature review and experiments support it. [ComRisk](https://arxiv.org/abs/2202.03874), [RA-HTGNN (2026)](https://www.sciencedirect.com/science/article/pii/S1877050926027055)

These prior datasets and relation types are not necessarily transferable to the anonymized U.S. accounting panel. Use them to determine the baseline and novelty bar; make direct numerical comparisons only when task, event horizon, population, features, and evaluation split match.

## 4. Dataset and point-in-time target gate

The first candidate is the public American Stock Market company-year dataset (1999–2018). It is excluded from Git and retained locally under `data/raw/`. Its source paper and downloadable release must be checked against each other. The current audit found 78,682 rows and 8,971 firms in the CSV, while the paper reports 8,262 firms; all 609 firms marked failed have that label in every observed year, with no within-firm label changes. The released label therefore cannot be used as a one-year-ahead event label. See [`docs/data-audit.md`](docs/data-audit.md) and the [dataset paper](https://www.mdpi.com/1999-5903/14/8/244).

Before training, the replacement or repaired panel must document: (1) the exact event, such as bankruptcy filing versus another distress definition; (2) the forecast horizon and index date; (3) the date each feature became public, not just fiscal year-end; (4) event dates and fully observed negative follow-up; (5) right-censoring and firms with multiple observations; (6) identifier continuity; (7) license and redistribution constraints. For a one-year classification target, an eligible row at time *t* is positive only if its event occurs after the information cutoff and within the declared horizon, and negative only if the full horizon is observed without that event. Exclude rows with prior events or incomplete follow-up. Do not infer anonymized firm identities, industry, or event dates without a defensible join source.

If only valid time-to-event and censoring information exists, add a survival/hazard analysis as a separate task; do not silently convert censored firms to negatives. If only a valid preconstructed one-year label exists, it can support fixed-horizon classifiers but not survival claims.

## 5. Baseline and model comparison ladder

The comparison is designed to find out whether each added information source earns its complexity. Every baseline is trained on the same target cohort and point-in-time data. The current code implements only part of this ladder; missing required baselines and their documented tuning must be completed before the final test run.

| Tier | Required methods | Input representation | Purpose |
|---|---|---|---|
| Established score/statistical models | Altman Z-score as a published fixed score where its inputs are available; regularized logistic regression; linear discriminant analysis; penalized discrete-time hazard/logit when event timing supports it | Current-year financial ratios, plus a prespecified raw-feature version | Verify that the method beats interpretable and long-standing financial-risk references, not just weak ML models |
| Classical ML | Decision tree; RBF or linear SVM; random forest; ExtraTrees; histogram gradient boosting; **XGBoost and LightGBM** (at least one must be present, both preferred); CatBoost if categorical features are used | Current-year snapshot and matched five-year flattened features | Benchmark nonlinear interactions and the strongest widely used tree ensembles; these are mandatory challengers |
| Survival/risk models, conditional on data | Cox/discrete-time hazard model and random survival forest | Event times, censoring, and point-in-time covariates | Compare event-time risk modeling when valid event and censoring data exist; do not force these models into an invalid binary-label task |
| Neural tabular controls | MLP; optional FT-Transformer/TabNet only if a fair compute budget is available | Current-year and matched five-year flattened features | Test whether any gain is from neural capacity rather than temporal or graph structure |
| Temporal models | LSTM, GRU, and a temporal convolution model if time permits | Same five-year sequence and masks for missing history | Isolate temporal encoding from graph message passing |
| Conventional graph models | GCN, GraphSAGE, and GAT; one relation-agnostic dynamic peer graph; relation-specific GNN without persistence | Same LSTM node states, node cohort, yearly graph snapshots, and labels | Compare the proposed method to standard graph aggregation and relation modeling |
| Proposed method and ablations | Relation-specific temporal GNN with current similarity and neighbor-set persistence; no-persistence, similarity-only, persistence-only, merged-relation, leave-one-relation-out, and no-message variants | Same temporal sequences and snapshots as graph controls | Identify whether the persistence reliability mechanism itself improves results |

**Feature-view matching is mandatory.** For each classical and tabular model, report (A) current-year features and (B) the same trailing five years flattened with a fixed missing-history mask. The LSTM and GNN receive those same values and masks as sequences. This prevents a misleading result where the GNN wins merely because it sees more history. Add ratios and raw items only as a declared feature-family comparison, applying identical availability rules to every model. Map variables into liquidity, leverage, profitability, and cash-flow relations only when source definitions justify it; opaque or anonymized feature names must remain in an honestly named general financial-similarity relation.

**Strongest comparator selection.** Tune the specified baselines on training/validation data only. Before test evaluation, freeze the strongest traditional challenger by validation PR-AUC from the required statistical/classical ML models, and the strongest temporal/graph challenger by validation PR-AUC from the temporal and conventional graph tier. The GNN must beat both frozen challengers. Also report every baseline, not only winners. Do not compare against a baseline with a deliberately weak configuration.

If a method cannot be implemented or its original inputs are unavailable, record the reason and mark it as unavailable; do not call the resulting comparison exhaustive. A published score from a different dataset or random split is context, not an apples-to-apples baseline.

## 6. Fair training, tuning, and leakage controls

- **One cohort:** use the same eligible firm-years and label horizon for all methods. State any method-specific exclusions and rerun the shared comparison on the intersection cohort.
- **Point-in-time features:** feature values must have been public by the forecast cutoff. Fiscal-year labels alone do not prove availability. Graphs for year *t* use only features available by *t*; no future labels or future features enter graph construction.
- **Preprocessing:** fit imputation, clipping, scaling, feature selection, and any class weights on training years only. Apply the frozen transforms to validation and test.
- **Class imbalance:** use training-fold weighting as the default; report the exact positive weighting. If sampling or SMOTE is explored, apply it inside the training fold only and report it as a separate condition. Never oversample validation/test or select a threshold on test.
- **Chronology:** preserve a chronological train/validation/test split. Use expanding-window validation folds for model selection and a future holdout that is untouched until configurations are frozen. Where event counts allow, use at least two distinct future test windows. Report a separate company-disjoint evaluation as a generalization check; do not substitute it for the time-forward primary test.
- **Tuning budget:** publish model-specific search spaces and use the same number of validation trials per model family (target at least 50 trials for the required GBDT baselines and proposed model), with a prespecified per-trial or total compute cap. Use the same expanding validation folds. If resource limits force a lower budget, reduce it for all candidates and state the limit before test evaluation.
- **Randomness:** train at least five predeclared seeds for neural and graph methods and repeat randomized baselines with the same seed set. Keep data split and candidate search protocol fixed across seeds.
- **Calibration/thresholds:** calibrate only on validation data if calibration is used. Choose the operating threshold on validation data targeting 80% recall; freeze it before the test. Report precision and recall achieved on the test.
- **Graph construction:** preregister similarity metric, feature groups, *k*, directed/symmetric handling, self-loop rules, persistence formula, and missing-feature behavior. Tune graph density only on validation data. Compare graph variants with matched node states and training budgets.
- **Run integrity:** save configuration, source/data checksums, split-year lists, random seeds, software versions, selected checkpoints, validation predictions, test predictions, and logs. Keep raw data and unpublished results out of the public repo unless redistribution rights are confirmed.

## 7. Metrics and statistical evidence

**Primary ranking metric: PR-AUC / average precision**, because distress is rare and performance on the positive class matters. **ROC-AUC is a co-required success metric** to check ranking across thresholds. A model does not pass by improving only one metric.

Also report precision at a validation-selected threshold targeting 80% recall, achieved test precision and recall, Brier score, calibration plot/slope/intercept when estimable, confusion matrix, positive prevalence, class counts, and per-window results. Report accuracy only as a supplemental number, never as the principal success measure. If possible, add a decision-cost or expected-loss analysis with cost assumptions fixed before looking at test results.

For the proposed model versus each frozen strongest comparator, calculate paired 95% confidence intervals for differences in PR-AUC and ROC-AUC using a company-cluster bootstrap (resample firm clusters, retaining all their test-year rows together; at least 10,000 valid replicates). Report seed-level and future-window results rather than pooling away failures. Use a predefined correction for any additional confirmatory pairwise tests; mark exploratory comparisons as exploratory. Publish all metric tables and intervals, including unfavorable results.

## 8. Hypotheses and explicit falsification tests

- **H1 — temporal signal:** five-year sequence methods outperform current-year-only controls. Falsified if the matched-history baselines and LSTM do not improve over current-year models on future holdouts.
- **H2 — peer signal:** graph methods improve over the LSTM with identical temporal input. Falsified if the graph models do not beat LSTM or the improvement is not stable over seeds/windows.
- **H3 — persistence reliability:** combined similarity-plus-persistence edges improve over similarity-only dynamic graphs and ordinary GCN/GraphSAGE/GAT. Falsified if persistence ablation is indistinguishable or worse, or if random/matched-degree edges perform similarly.
- **H4 — robustness:** the proposed method degrades less under plausible edge deletion, addition, and feature-noise perturbation than conventional graph baselines. Falsified if gains vanish under small perturbations or depend on one arbitrary *k* setting.
- **H5 — value beyond tabular state:** the proposed model beats the strongest well-tuned boosting and statistical baselines on the common out-of-time sample. Falsified if XGBoost/LightGBM or another validated traditional model matches or beats it.

Only hypotheses that pass their predeclared tests may appear as supported contributions. No causal interpretation follows from predictive gains.

## 9. Hard pass/fail gate for a successful paper

All conditions below must hold on an audited event-horizon dataset; otherwise, under the user's criterion, the paper has **not** succeeded:

1. The proposed model has higher test PR-AUC than the frozen strongest traditional baseline, which includes a properly tuned XGBoost or LightGBM comparator, and than the frozen strongest temporal/graph baseline.
2. For both comparisons, paired company-clustered 95% confidence intervals for the PR-AUC difference are entirely above zero. The same is true for ROC-AUC differences.
3. Both ranking gains have the same direction across all declared seeds and each adequately powered future test window. One lucky seed or one favorable period is not enough.
4. At the frozen validation-selected operating point, test precision at the target recall does not fall below either comparator. The upper 95% confidence bound for the proposed-minus-comparator Brier score difference must be below +0.01 absolute; calibration diagnostics must also satisfy their prespecified non-deterioration rule.
5. Point-in-time leakage, target leakage, test tuning, post hoc baseline removal, and selective reporting are absent.

If any condition fails, report the failure plainly, preserve the experiment, and improve the data or method before making a successful-paper claim. Do not loosen the threshold, switch headline metrics, omit a strong baseline, or relabel a negative result as success. Acceptance at an A* workshop is never guaranteed even if this empirical gate passes.

## 10. Execution schedule and deliverables

| Weeks | Work | Exit condition |
|---|---|---|
| 1–2 | Literature and data audit; secure a valid event-year/availability panel; write protocol before modeling | Event definition, censoring, license, point-in-time cutoffs, and baseline table approved internally; if the target cannot be recovered, stop empirical claims and find another source |
| 3–4 | Implement audited data pipeline and classical/statistical benchmark suite, including tuned GBDT models; create validation-only model-selection report | All required conventional methods train on the same folds; leakage and split audit is reproducible |
| 5–6 | Add five-year feature views, MLP, temporal LSTM/GRU, and conventional GNN baselines | Matched-input comparison table runs; baseline configs and compute are frozen |
| 7–8 | Implement proposed graph reliability method and prespecified ablations | Persistence, similarity-only, persistence-only, and ordinary graph models are isolated; unit-level graph checks pass |
| 9–10 | Run seeds, future windows, company-disjoint analysis, and robustness experiments | Complete outputs and paired cluster-bootstrap CIs are generated without test tuning |
| 11–12 | Apply hard success gate; repeat/fix only protocol-approved issues; complete figures and error analysis | Gate outcome documented; only evidence-supported claims retained |
| 13–14 | Write and format workshop manuscript; verify venue fit and current deadline; release code/configs and permitted artifacts | Reproduction instructions and final paper match committed evidence; no unverified score is included |

**Current implementation status:** the repository currently runs logistic regression, decision tree, random forest, histogram gradient boosting, and MLP on current-year and flattened-history views, plus LSTM, GCN, GraphSAGE, and the proposed model. XGBoost/LightGBM, SVM, classic financial scores, hazard/survival comparators, systematic tuning, multi-window runs, and the paired cluster-bootstrap report are still required before the benchmark is strong enough to evaluate the user's gate. The dataset target audit remains the immediate blocker. No cloud experiment is configured or launched.

## References informing the comparison plan

1. Alanis, Chava, and Shah. *Benchmarking Machine Learning Models to Predict Corporate Bankruptcy*. [arXiv:2212.12051](https://arxiv.org/abs/2212.12051).
2. Zhao et al. *Combining Intra-Risk and Contagion Risk for Enterprise Bankruptcy Prediction Using Graph Neural Networks*. [arXiv:2202.03874](https://arxiv.org/abs/2202.03874).
3. Sun, Shao, and Meng. *Network-Aware Financial Distress Prediction via Hierarchical Attention on Dynamic Heterogeneous Corporate Networks*. [Procedia Computer Science (2026)](https://www.sciencedirect.com/science/article/pii/S1877050926027055).
4. Liang et al. *Machine Learning Models and Bankruptcy Prediction*. [Expert Systems with Applications (2018)](https://www.sciencedirect.com/science/article/pii/S0957417417302415).
