# Run FME on Kaggle

The CLI runs a Python script kernel with the T4 accelerator. The project is bundled into the generated script, so Kaggle only needs the entrypoint and metadata.

## Prepare a kernel

Install and authenticate the Kaggle CLI on the machine that will push the kernel. Then prepare a kernel using the Kaggle kernel slug and a dataset containing a valid `panel.csv`:

```powershell
python scripts/prepare_kaggle_kernel.py `
  --kernel-id owner/kernel-slug `
  --data-source owner/panel-dataset `
  --output-dir kaggle/kernel
kaggle kernels push -p kaggle/kernel --accelerator NvidiaTeslaT4
```

For an existing kernel identified by Kaggle's numeric ID, pass the number to `--kernel-id`. Kaggle kernel identifiers are not arbitrary UUIDs. Review `kaggle/kernel/kernel-metadata.json` before pushing; the helper keeps the kernel private and selects the T4 machine shape.

The runner finds the first `/kaggle/input/**/panel.csv` and requires CUDA. Without an attached panel dataset, it stops with an actionable error. For a synthetic software run only, prepare with `--smoke-test`; those metrics are not research results. Add `--enable-internet` only when the kernel needs network access to install dependencies.

The panel must include `company_id`, `fiscal_year`, `label`, and `available_date`, and pass the repository's target transition audit. The known public candidate dataset is not valid for this forecast task because its labels are constant within each company. The kernel writes artifacts to `/kaggle/working/fme-results` for Kaggle output collection.

## Polish Companies Bankruptcy archive

The supplied archive is the UCI Polish Companies Bankruptcy dataset, licensed CC BY 4.0. A private copy is available on Kaggle as `atharvshanker/polish-companies-bankruptcy-fme`. `5year.arff` is used for its one-year-ahead label. Since the files have no company IDs or fiscal-year keys, this path evaluates tabular baselines and a single-snapshot LSTM-GNN/GCN/GraphSAGE under five stratified, grouped random holdouts. Its LSTM receives only one feature vector, so the experiment does not evaluate temporal sequence learning, graph persistence, chronological splits, or company-clustered confidence intervals.

To rebuild and push the private T4 benchmark kernel:

```powershell
python scripts/prepare_kaggle_kernel.py `
  --kernel-id atharvshanker/fme-polish-bankruptcy-t4-nested-benchmark `
  --title "FME Polish Bankruptcy T4 Nested Benchmark" `
  --data-source atharvshanker/polish-companies-bankruptcy-fme `
  --polish-benchmark `
  --enable-internet `
  --output-dir kaggle/polish-kernel
kaggle kernels push -p kaggle/polish-kernel --accelerator NvidiaTeslaT4
```

For a protocol-matched five-fold outer evaluation, prepare a separate kernel with `--outer-cv` and a distinct kernel slug. The same shuffled `StratifiedGroupKFold` partition is used for all five folds, so each row is scored once in an untouched outer fold. Each fold tunes only inside its outer development data and reports a fold-specific result:

```powershell
python scripts/prepare_kaggle_kernel.py `
  --kernel-id atharvshanker/fme-polish-bankruptcy-t4-outer-cv `
  --title "FME Polish Bankruptcy T4 Outer CV" `
  --data-source atharvshanker/polish-companies-bankruptcy-fme `
  --polish-benchmark `
  --outer-cv `
  --enable-internet `
  --output-dir kaggle/polish-kernel-outer-cv
kaggle kernels push -p kaggle/polish-kernel-outer-cv --accelerator NvidiaTeslaT4
```

The current kernel is [FME Polish Bankruptcy T4 Nested Benchmark](https://www.kaggle.com/code/atharvshanker/fme-polish-bankruptcy-t4-nested-benchmark). Version 11 completed on Kaggle T4; aggregate metrics, predictions, per-seed results, and tuned configurations are in `results/kaggle-polish-version11/fme-results/`. The run includes CatBoost alongside XGBoost, LightGBM, classical tabular models, graph baselines, and architecture ablations. See [`../docs/polish-benchmark.md`](../docs/polish-benchmark.md) for the protocol and results.

## CLI limitations

The CLI must be installed and authenticated separately (`kaggle` command and account API token). This repository does not contain account credentials. Kaggle API kernel slugs use `owner/kernel-slug`, or the API accepts a numeric kernel ID; an API credential is not a kernel ID.
