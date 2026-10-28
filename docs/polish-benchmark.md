# UCI Polish Companies Bankruptcy benchmark

## Dataset and task

The archive is the UCI Polish Companies Bankruptcy dataset by Sebastian Tomczak (2016), DOI [10.24432/C5F600](https://doi.org/10.24432/C5F600), licensed CC BY 4.0. The `5year.arff` subset has 5,910 rows, 64 financial attributes, 410 bankruptcies, and a one-year forecast horizon.

The supplied files contain no company identifier or fiscal-year key. This is a cross-sectional benchmark; it cannot provide per-company histories, chronological splits, company-disjoint evaluation, or company-clustered confidence intervals. The LSTM receives one vector at one time step here. The results therefore do not evaluate temporal learning or graph persistence.

## Architecture and evaluation

The proposed model uses an LSTM encoder, relation-specific edge-weighted graph attention over financial peer groups, a learned relation gate, and a gated residual fusion of self and neighbor representations. For the Polish experiment, similarity graphs are built inductively: validation and test rows query training peers, and held-out rows do not send messages back to training rows.

The prior-guided variant adds a missingness-aware LightGBM prior trained with grouped out-of-fold logits on the outer training partition. A capped residual branch uses the prior logit, its uncertainty, and learned company/peer representations. A CSA-style MLP residual comparator uses the same prior and missingness indicators without graph messages. Positive-label error weights, adaptive versus fixed gates, graph removal, and missingness removal are ablated.

Version 6 used five predeclared grouped random holdouts (seeds 401–405). Exact duplicate feature rows stay in one split. Imputation and scaling fit on training rows only. Twelve random-search configurations per family were selected using each seed's validation partition; the held-out test partition was used only for final metrics. Thresholds target 80% validation recall. Reported standard deviations describe split variability, not confidence intervals.

Comparators include logistic regression, shrinkage LDA, decision tree, random forest, ExtraTrees, histogram gradient boosting, calibrated RBF SVM, XGBoost, LightGBM, MLP, LSTM-only, GCN, and GraphSAGE. The graph suite also includes unweighted edges, 5-neighbor graphs, and leave-one-financial-relation-out ablations.

## Version 11 results

Version 11 added CatBoost as a standard comparator and used fresh grouped holdouts (seeds 901–905) on Kaggle CUDA/T4. XGBoost, LightGBM, CatBoost, the missingness-aware LightGBM prior, and standalone LSTM-GNN each received 12 candidates with three-fold grouped inner selection within the outer development partition. The outer test partition was not used by this tuning stage. Final models were refit on train plus validation. The prior-guided residual candidates and validation-selected two-model blends were selected on the outer validation split. Earlier benchmark iterations informed architecture development, so the combined study remains exploratory rather than blinded confirmation.

| Model | Test PR-AUC | Test ROC-AUC |
|---|---:|---:|
| CSA-style LightGBM residual | 0.8148 ± 0.0246 | 0.9595 ± 0.0064 |
| Missingness-aware LightGBM prior | 0.8148 ± 0.0246 | 0.9595 ± 0.0064 |
| LightGBM-prior LSTM-GNN | 0.8147 ± 0.0245 | 0.9595 ± 0.0064 |
| Missingness-prior/XGBoost blend | 0.8128 ± 0.0287 | 0.9588 ± 0.0072 |
| Missingness-prior/CatBoost blend | 0.8112 ± 0.0298 | 0.9588 ± 0.0053 |
| LightGBM | 0.7861 ± 0.0293 | 0.9532 ± 0.0027 |
| XGBoost | 0.7815 ± 0.0323 | 0.9492 ± 0.0060 |
| Histogram gradient boosting | 0.7781 ± 0.0311 | 0.9510 ± 0.0039 |
| CatBoost | 0.7601 ± 0.0360 | 0.9462 ± 0.0055 |
| Random forest | 0.4782 ± 0.0485 | 0.8972 ± 0.0115 |
| Decision tree | 0.4118 ± 0.0624 | 0.7216 ± 0.0408 |
| ExtraTrees | 0.3559 ± 0.0518 | 0.8493 ± 0.0160 |
| Logistic regression | 0.3479 ± 0.0441 | 0.8150 ± 0.0439 |
| Shrinkage LDA | 0.2794 ± 0.0351 | 0.7748 ± 0.0168 |
| Calibrated RBF SVM | 0.2428 ± 0.0262 | 0.7453 ± 0.0129 |
| MLP | 0.2639 ± 0.0456 | 0.7442 ± 0.0419 |
| LSTM-GNN relation attention, one step | 0.3577 ± 0.0588 | 0.8162 ± 0.0277 |
| LSTM-only, one step | 0.3225 ± 0.0627 | 0.7670 ± 0.0113 |
| GCN | 0.3508 ± 0.0519 | 0.7948 ± 0.0214 |
| GraphSAGE | 0.3362 ± 0.0682 | 0.7797 ± 0.0179 |

The prior-guided LSTM-GNN and CSA-style residual remain effectively tied with the tree prior. Across the prior, gate, cost-weight, and missingness ablations, mean PR-AUC changed by at most 0.0001. The standalone one-step LSTM-GNN and conventional graph models are far below the boosted trees.

| LSTM-GNN mechanism ablation | Test PR-AUC |
|---|---:|
| Unweighted edges | 0.3601 ± 0.0522 |
| 5-neighbor graph | 0.3569 ± 0.0480 |
| Without liquidity relation | 0.3608 ± 0.0458 |
| Without solvency relation | 0.3674 ± 0.0451 |
| Without profitability relation | 0.3472 ± 0.0710 |
| Without efficiency/scale relation | 0.3608 ± 0.0688 |

The best mean is **0.0092 below** the published CSA-Net one-year result of 0.824 ± 0.018. The benchmark protocols differ, so this gap is descriptive; our holdout SD is not a confidence interval and does not support a formal superiority test. The LSTM-GNN has not shown a consistent gain or state-of-the-art performance.

## Version 10 results

Version 10 evaluated a residual-checkpoint selection fix on fresh holdouts (seeds 801–805). When PR-AUC tied, validation loss selected the checkpoint; residual candidate ties used validation Brier score. The best validation-selected LightGBM/XGBoost blend averaged 0.8146 ± 0.0360 PR-AUC, while the missingness-aware prior averaged 0.8142 ± 0.0372. The prior-guided residual remained effectively tied with the prior, so this change did not produce a useful performance gain.

## Version 9 results

Version 9 used five fresh grouped holdouts (seeds 701–705) on Kaggle CUDA/T4. For each seed, 12 candidates per family were selected with three-fold grouped inner cross-validation on the outer development partition for XGBoost, LightGBM, the missingness-aware LightGBM prior, and the standalone LSTM-GNN. The outer test set was not read by that tuner. Models selected on the outer validation split were refit on train plus validation before the independent test score. The prior-guided neural residual and the prior/XGBoost blend were selected on the outer validation split. Because earlier exploratory test results informed architecture iteration, this remains exploratory rather than a blinded confirmatory study.

| Model | Test PR-AUC | Test ROC-AUC |
|---|---:|---:|
| Missingness-prior/XGBoost blend | 0.8123 ± 0.0192 | 0.9592 ± 0.0056 |
| Missingness-aware LightGBM prior | 0.8117 ± 0.0207 | 0.9592 ± 0.0056 |
| CSA-style LightGBM residual | 0.8117 ± 0.0207 | 0.9592 ± 0.0056 |
| LightGBM-prior LSTM-GNN | 0.8116 ± 0.0207 | 0.9592 ± 0.0056 |
| Histogram gradient boosting | 0.7996 ± 0.0381 | 0.9584 ± 0.0036 |
| XGBoost | 0.7967 ± 0.0186 | 0.9534 ± 0.0074 |
| LightGBM | 0.7959 ± 0.0403 | 0.9565 ± 0.0053 |
| Random forest | 0.5330 ± 0.0449 | 0.9186 ± 0.0112 |
| Decision tree | 0.4499 ± 0.0431 | 0.7589 ± 0.0297 |
| ExtraTrees | 0.3796 ± 0.0430 | 0.8736 ± 0.0106 |
| LSTM-GNN relation attention, one step | 0.3842 ± 0.0553 | 0.8313 ± 0.0279 |
| LSTM-only, one step | 0.3851 ± 0.0371 | 0.7924 ± 0.0156 |
| GCN | 0.3744 ± 0.0421 | 0.8039 ± 0.0204 |
| GraphSAGE | 0.3742 ± 0.0411 | 0.8133 ± 0.0255 |
| Logistic regression | 0.3657 ± 0.0792 | 0.8432 ± 0.0189 |
| Shrinkage LDA | 0.3119 ± 0.0820 | 0.8100 ± 0.0294 |
| MLP | 0.3138 ± 0.0799 | 0.7881 ± 0.0256 |
| Calibrated RBF SVM | 0.2314 ± 0.0407 | 0.7510 ± 0.0324 |

The average precision gain of the best blend over the missingness-aware LightGBM prior was only 0.0006. The prior-guided LSTM-GNN and MLP residual were numerically tied with the prior; their graph, gate, cost-weight, and missingness ablations changed mean PR-AUC by at most about 0.0002. The single-snapshot LSTM-GNN scored 0.3842 ± 0.0553, close to the one-step LSTM-only result and far below boosted trees. This provides no evidence that the proposed graph architecture adds predictive value on this dataset.

| LSTM-GNN mechanism ablation | Test PR-AUC |
|---|---:|
| Unweighted edges | 0.3775 ± 0.0403 |
| 5-neighbor graph | 0.3925 ± 0.0337 |
| Without liquidity relation | 0.3858 ± 0.0567 |
| Without solvency relation | 0.3739 ± 0.0506 |
| Without profitability relation | 0.3770 ± 0.0652 |
| Without efficiency/scale relation | 0.3851 ± 0.0448 |

The prior-guided hybrid's test Brier score was 0.0298; the selected blend's was 0.0286. Threshold precision and recall were computed at a validation-selected threshold targeting 80% recall; they are not used to select these PR-AUC rankings. Means and standard deviations are over five different holdout splits, not confidence intervals. No paired company-level confidence interval can be calculated because the data contain no company IDs.

CSA-Net reports **0.824 ± 0.018** for the one-year Polish horizon under five outer grouped folds and three-fold inner selection ([Wang, Lan & Jie, *Scientific Reports*, 2026](https://www.nature.com/articles/s41598-026-68005-8)). Version 9's best mean is lower by about 0.012 PR-AUC. The protocols differ, so this numerical gap is descriptive rather than a formal significance comparison. The paper also evaluates the five Polish horizons separately; it does not treat them as longitudinal records of the same firms.

## Version 8 results

Version 8 used five grouped holdouts (seeds 601–605), validation selection, and a final train-plus-validation refit. Its missingness-aware LightGBM prior scored 0.8042 ± 0.0298 PR-AUC; the prior-guided LSTM-GNN and CSA-style residual were effectively tied with it. Version 9 added per-seed three-fold inner grouped tuning for the major tree/prior/standalone-LSTM-GNN families.

## Version 7 results

Version 7 evaluated five fresh grouped holdouts (seeds 501–505) on Kaggle CUDA, with 24 validation-only configurations per family. It tuned the LightGBM prior on value-plus-missingness inputs and selected a prior/XGBoost blend weight on validation data. The best average was the missingness-prior/XGBoost blend at **0.7866 ± 0.0312 PR-AUC**; the missingness-aware prior alone averaged 0.7845 ± 0.0397. The prior-guided LSTM-GNN scored 0.7842 ± 0.0401, and the CSA-style residual scored 0.7844 ± 0.0396. Those residuals again track the prior closely. The score difference between the blend and the prior is small and has not been tested with a paired confidence interval.

| Model | Test PR-AUC | Test ROC-AUC |
|---|---:|---:|
| Missingness-prior/XGBoost blend | 0.7866 ± 0.0312 | 0.9511 ± 0.0089 |
| Missingness-aware LightGBM prior | 0.7845 ± 0.0397 | 0.9512 ± 0.0103 |
| CSA-style LightGBM residual | 0.7844 ± 0.0396 | 0.9512 ± 0.0103 |
| LightGBM-prior LSTM-GNN | 0.7842 ± 0.0401 | 0.9512 ± 0.0103 |
| XGBoost | 0.7622 ± 0.0248 | 0.9459 ± 0.0088 |
| LightGBM | 0.7620 ± 0.0394 | 0.9463 ± 0.0127 |
| Histogram gradient boosting | 0.7464 ± 0.0402 | 0.9446 ± 0.0125 |
| Random forest | 0.4926 ± 0.0374 | 0.9045 ± 0.0085 |
| ExtraTrees | 0.3922 ± 0.0456 | 0.8675 ± 0.0198 |
| GCN | 0.3657 ± 0.0221 | 0.8153 ± 0.0249 |
| LSTM-GNN relation attention, one step | 0.3633 ± 0.0181 | 0.8213 ± 0.0196 |
| LSTM-only, one step | 0.3545 ± 0.0223 | 0.7930 ± 0.0141 |
| GraphSAGE | 0.3521 ± 0.0378 | 0.7992 ± 0.0217 |
| Logistic regression | 0.3595 ± 0.0438 | 0.8417 ± 0.0285 |
| Shrinkage LDA | 0.3111 ± 0.0218 | 0.8056 ± 0.0176 |
| MLP | 0.2921 ± 0.0522 | 0.7729 ± 0.0394 |
| Calibrated RBF SVM | 0.2369 ± 0.0252 | 0.7559 ± 0.0301 |

All five seeds used CUDA. The blend's selected prior weights were 0.25, 0.90, 0.65, 1.00, and 0.45 across seeds, showing that validation sometimes favored XGBoost and sometimes the LightGBM prior. Version 7 does not change the core conclusion: boosted trees lead, and the LSTM-GNN has not shown a consistent gain.

## Version 6 results

All five seeds ran on Kaggle CUDA. The strongest result was the missingness-aware LightGBM prior at **0.8061 ± 0.0334 PR-AUC**. The prior-guided LSTM-GNN reached 0.8062 ± 0.0335, and the CSA-style MLP residual reached 0.8062 ± 0.0334. Their per-seed gains over the prior were below 0.0003 PR-AUC, so this run does not establish a useful residual or graph contribution.

| Model | Test PR-AUC | Test ROC-AUC |
|---|---:|---:|
| CSA-style LightGBM residual | 0.8062 ± 0.0334 | 0.9598 ± 0.0122 |
| LightGBM-prior LSTM-GNN | 0.8062 ± 0.0335 | 0.9599 ± 0.0122 |
| Missingness-aware LightGBM prior | 0.8061 ± 0.0334 | 0.9598 ± 0.0122 |
| LightGBM | 0.7835 ± 0.0243 | 0.9551 ± 0.0117 |
| XGBoost | 0.7718 ± 0.0270 | 0.9517 ± 0.0102 |
| Histogram gradient boosting | 0.7605 ± 0.0335 | 0.9476 ± 0.0154 |
| Random forest | 0.4850 ± 0.0500 | 0.9080 ± 0.0150 |
| GCN | 0.4180 ± 0.0470 | 0.8370 ± 0.0260 |
| GraphSAGE | 0.3940 ± 0.0450 | 0.8280 ± 0.0180 |
| 5-neighbor LSTM-GNN ablation | 0.3870 ± 0.0350 | 0.8270 ± 0.0190 |
| LSTM-only, one step | 0.3785 ± 0.0473 | 0.8158 ± 0.0212 |
| ExtraTrees | 0.3790 ± 0.0340 | 0.8650 ± 0.0230 |
| LSTM-GNN relation attention, one step | 0.3582 ± 0.0398 | 0.8243 ± 0.0152 |
| Unweighted-edge LSTM-GNN ablation | 0.3580 ± 0.0510 | 0.8170 ± 0.0180 |
| Decision tree | 0.3350 ± 0.0380 | 0.7370 ± 0.0440 |
| Logistic regression | 0.3290 ± 0.0340 | 0.8240 ± 0.0320 |
| Shrinkage LDA | 0.3190 ± 0.0280 | 0.8210 ± 0.0290 |
| MLP | 0.2850 ± 0.0120 | 0.7950 ± 0.0210 |
| Calibrated RBF SVM | 0.2360 ± 0.0290 | 0.7760 ± 0.0130 |

The single-snapshot LSTM-GNN underperformed the tree baselines. Its prior-guided variant mostly reproduced the prior's ranking, and the cost, gate, graph, and missingness ablations were nearly tied. The clearest gain in this run came from adding missingness indicators to the LightGBM prior, not from the novel graph component.

## State-of-the-art comparison

CSA-Net reports 1-year PR-AUC of 0.824 ± 0.018 on the Polish bankruptcy task, using nested grouped 5-fold/3-fold inner CV with a missingness-aware LightGBM prior and adaptive cost-sensitive residual ([Wang, Lan & Jie, *Scientific Reports*, 2026](https://www.nature.com/articles/s41598-026-68005-8)). Version 11's best mean is lower by about 0.009 PR-AUC and uses repeated grouped holdouts rather than the paper's five-fold outer protocol. The protocols are not directly interchangeable, and this work does **not** establish state-of-the-art performance. The UCI subset also cannot assess actual temporal learning because it lacks firm IDs and year keys.

## Artifacts

- [Kaggle T4 benchmark kernel](https://www.kaggle.com/code/atharvshanker/fme-polish-bankruptcy-t4-nested-benchmark)
- [Version 11 aggregate metrics](../results/kaggle-polish-version11/fme-results/metrics.json)
- [Version 11 combined predictions](../results/kaggle-polish-version11/fme-results/predictions.csv)
- [Version 9 aggregate metrics](../results/kaggle-polish-version9/fme-results/metrics.json)
- [Version 9 combined predictions](../results/kaggle-polish-version9/fme-results/predictions.csv)
- [Version 7 aggregate metrics](../results/kaggle-polish-version7/fme-results/metrics.json)
- [Version 7 combined predictions](../results/kaggle-polish-version7/fme-results/predictions.csv)
- [Version 6 aggregate metrics](../results/kaggle-polish-version6/fme-results/metrics.json)
- [Benchmark runner](../scripts/run_polish_benchmark.py)
- [Per-seed nested tuning runner](../scripts/tune_polish.py)
- [Repeated-seed runner](../scripts/run_polish_repeated.py)
