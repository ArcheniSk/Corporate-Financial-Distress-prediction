# FME Project: Time-Aware Graph Learning for Corporate Distress

Research software for testing whether reliable, time-varying peer relationships improve corporate distress prediction. The central model combines a five-year LSTM with relation-specific graph message passing over financial similarity and, when available, industry peer links.

The original project synopsis is preserved at [`synopsis.pdf`](synopsis.pdf).

## Research status

The initial public U.S. bankruptcy dataset has been audited, and its `status_label` is constant for every company across all observed years. Its CSV also reports 8,971 firms while the paper reports 8,262. It therefore does **not** provide the point-in-time next-year labels required by this project as released. The training command rejects it for forecasting. No performance claims or results are reported.

The project accepts a prepared, licensed `panel.csv` with:

| Column | Meaning |
|---|---|
| `company_id` | Stable firm identifier |
| `fiscal_year` | Financial reporting year |
| `label` | Binary event label for the documented future horizon |
| numeric columns | Financial predictors available by the prediction date |
| optional `available_date` | Date those predictors became public |
| optional `industry` | Licensed industry group for a separate graph relation |

Labels must reflect a known future horizon and survive the label-transition audit. Do not use company name, filing date, or future-derived status as predictors.

The supplied UCI Polish Companies Bankruptcy archive is used for a cross-sectional one-year-ahead benchmark. It has 64 financial attributes but no company IDs or fiscal-year keys, so it cannot evaluate longitudinal LSTM-GNN behavior, persistence weighting, chronological splits, or company-clustered uncertainty. The comparison suite includes tabular, boosted-tree, missingness-aware prior, residual, and single-snapshot graph models. Run the repeated-seed grouped random-holdout suite with:

```bash
python scripts/run_polish_repeated.py --data_file data/raw/polish-bankruptcy/5year.arff --output_dir results/polish-local --device cuda
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the prepared panel at `data/raw/panel.csv`, or pass another path with `--data_file`.

## Kaggle CLI (T4)

See [`kaggle/README.md`](kaggle/README.md) for preparing a self-contained Kaggle script kernel and pushing it with the T4 accelerator. Kaggle execution requires an authenticated CLI account and a valid panel dataset attached to the kernel. The CLI helper accepts a Kaggle `owner/kernel-slug` or numeric kernel ID; API tokens are credentials and should be configured locally as environment variables, never committed or pasted into project files.

The private UCI Polish archive has been benchmarked on Kaggle T4 with standard tabular models (including XGBoost, LightGBM, and CatBoost), missingness-aware priors, validation-selected blends, single-snapshot graph models, and mechanism ablations. The latest completed run is version 11; aggregate metrics and predictions are in `results/kaggle-polish-version11/fme-results/` and the full results are documented in [`docs/polish-benchmark.md`](docs/polish-benchmark.md). The prior-guided LSTM-GNN did not materially improve its LightGBM prior and remains below the published CSA-Net one-year result; these experiments do not establish state of the art. Because the UCI files have no firm identifiers or year keys, this benchmark cannot assess longitudinal LSTM learning or graph persistence.

## Audit and run

```bash
python scripts/audit_dataset.py --data_file data/raw/american_bankruptcy_dataset.csv
python train.py --data_dir data/raw --output_dir results/run-001
python train.py --data_file data/raw/panel.csv --output_dir results/run-001
python train.py --smoke_test --output_dir results/smoke
```

The audit command is read-only. Training is chronological: it selects the checkpoint and operating threshold on validation data, then evaluates the untouched test period. Outputs are written only to `--output_dir`. Smoke mode uses a deterministic synthetic panel to exercise the pipeline; its metrics are software checks, not research results.

## Data and license

The original candidate benchmark is available at [sowide/bankruptcy_dataset](https://github.com/sowide/bankruptcy_dataset) and carries its own CC BY 4.0 license. The exact audited upstream commit and file checksums are recorded in `docs/data-audit.md`. The raw CSV is ignored by Git and must not be redistributed without preserving its attribution and license. The synopsis PDF is not covered by this repository's MIT license; see `docs/NOTICE.md`.

## Reproducibility and reporting

Default split years, model settings, and seed are in `configs/default.json`. The output includes `metrics.json`, `predictions.csv`, `loss_curve.png`, and `model.pt`. The runnable comparison suite includes logistic regression, decision tree, random forest, histogram gradient boosting, and MLP controls on both current-year and flattened five-year features, as well as LSTM, GCN, GraphSAGE, and the reliability-aware temporal GNN. The separate Polish suite adds XGBoost, LightGBM, missingness-aware LightGBM, a CSA-style residual comparator, and graph ablations. Conventional graph baselines omit persistence weighting to isolate the proposed mechanism.

The paper's hard success criterion is to beat the strongest traditional and temporal/graph baselines on held-out PR-AUC and ROC-AUC with paired company-clustered confidence intervals above zero, consistent gains across seeds and future windows, and no failure of the operating-point or calibration guardrails. A negative or mixed result does not count as success under this project's criterion. The Polish experiments remain exploratory and do not satisfy the full temporal success gate because the archive lacks company identifiers and longitudinal observations; survival baselines and company-clustered confidence intervals also remain unavailable. See [`experiment-plan.md`](experiment-plan.md) and [`docs/comparison-framework.md`](docs/comparison-framework.md) for the rationale, baseline list, and pass/fail rules. Do not treat feature-similarity edges as causal links or evidence of contagion.

## Paper

See `paper/paper.md` for the claim-safe manuscript scaffold, `experiment-plan.md` for the study plan, and `docs/comparison-framework.md` for the baseline matrix and non-negotiable pass/fail criterion. The workshop venue remains unselected. The Polish cross-sectional benchmark has exploratory results, but it cannot meet the full temporal success criterion because it has no firm identifiers or year keys.
