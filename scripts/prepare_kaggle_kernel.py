#!/usr/bin/env python3
"""Package this repository as a self-contained Kaggle Python kernel."""

from __future__ import annotations

import argparse
import base64
import json
import re
import zipfile
from io import BytesIO
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = [Path("train.py"), Path("requirements.txt"), Path("scripts/run_polish_benchmark.py"), Path("scripts/run_polish_repeated.py"), Path("scripts/tune_polish.py"), *sorted(Path("src/fme").glob("*.py"))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-id", required=True, help="Existing Kaggle kernel: owner/kernel-slug, or its numeric ID.")
    parser.add_argument("--title", default="FME temporal graph training")
    parser.add_argument("--output-dir", default="kaggle/kernel")
    parser.add_argument("--data-source", action="append", default=[], help="Kaggle dataset slug containing panel.csv; may be repeated.")
    parser.add_argument("--enable-internet", action="store_true", help="Allow the kernel to install project dependencies.")
    parser.add_argument("--smoke-test", action="store_true", help="Run the synthetic software check instead of requiring panel.csv.")
    parser.add_argument("--polish-benchmark", action="store_true", help="Run the tabular-only one-year-ahead UCI Polish benchmark from an attached dataset.")
    parser.add_argument("--tuning-configs", type=int, default=50, help="Validation-only random search configs per family inside each outer split.")
    parser.add_argument("--tuning-epochs", type=int, default=40, help="Maximum epochs per LSTM-GNN tuning candidate.")
    parser.add_argument("--tuning-patience", type=int, default=5, help="Early-stopping patience for LSTM-GNN tuning.")
    parser.add_argument("--benchmark-seeds", nargs="+", type=int, default=[101, 102, 103, 104, 105], help="Fresh holdout seeds for the Polish benchmark run.")
    parser.add_argument("--outer-cv", action="store_true", help="Run every fold of one stratified grouped 5-fold outer evaluation.")
    parser.add_argument("--outer-cv-seed", type=int, default=42, help="Seed for the shared 5-fold outer partition.")
    parser.add_argument("--train-arg", action="append", default=[], help="Extra train.py argument; repeat for each argument.")
    return parser.parse_args()


def validate_kernel_id(value: str) -> tuple[str | None, int | None]:
    if value.isdecimal():
        return None, int(value)
    parts = value.split("/")
    if len(parts) == 2 and all(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", part) for part in parts):
        return value, None
    raise ValueError("Kaggle kernel ID must be owner/kernel-slug or a numeric kernel ID.")


def make_bundle() -> str:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in SOURCE_FILES:
            path = ROOT / relative
            if not path.is_file():
                raise FileNotFoundError(f"Required project source is missing: {path}")
            archive.write(path, relative.as_posix())
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def main() -> int:
    args = parse_args()
    if args.smoke_test and args.polish_benchmark:
        raise SystemExit("Choose either --smoke-test or --polish-benchmark, not both.")
    kernel_slug, kernel_number = validate_kernel_id(args.kernel_id)
    destination = (ROOT / args.output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    bundle = make_bundle()
    smoke = args.smoke_test
    extra_args = list(args.train_arg)
    if smoke:
        extra_args.insert(0, "--smoke_test")
    runner = f'''#!/usr/bin/env python3
"""Generated self-contained Kaggle entrypoint for the FME project."""
import base64, io, os, subprocess, sys, zipfile
from pathlib import Path
import importlib.util

PROJECT_BUNDLE = {bundle!r}
root = Path("/kaggle/working/fme-project")
root.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PROJECT_BUNDLE))) as archive:
    archive.extractall(root)
os.chdir(root)
if {args.enable_internet!r}:
    missing = [package for package in ("xgboost", "lightgbm", "catboost") if importlib.util.find_spec(package) is None]
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", *missing], check=True)
if {smoke!r}:
    run_args = {extra_args!r}
elif {args.polish_benchmark!r}:
    candidates = sorted(Path("/kaggle/input").rglob("5year.arff"))
    if not candidates:
        raise SystemExit("No 5year.arff found under /kaggle/input. Attach the private Polish bankruptcy dataset.")
    run_args = ["--data_file", str(candidates[0]), "--device", "cuda", "--output_dir", "/kaggle/working/fme-results", *{extra_args!r}]
    run_args += ["--tune-configs-per-family", str({args.tuning_configs}), "--tuning-epochs", str({args.tuning_epochs}), "--tuning-patience", str({args.tuning_patience})]
    if {args.outer_cv!r}:
        run_args += ["--outer-cv", "--outer-cv-seed", str({args.outer_cv_seed})]
    else:
        run_args += ["--seeds", *[str(seed) for seed in {args.benchmark_seeds!r}]]
    subprocess.run([sys.executable, "scripts/run_polish_repeated.py", *run_args], check=True)
    raise SystemExit(0)
else:
    data_root = Path("/kaggle/input")
    panels = sorted(data_root.rglob("panel.csv")) if data_root.exists() else []
    if not panels:
        raise SystemExit("No panel.csv found under /kaggle/input. Attach a Kaggle dataset containing a valid point-in-time panel.csv, or build with --smoke-test for a synthetic software check.")
    run_args = ["--data_file", str(panels[0])] + {extra_args!r}
if "--device" not in run_args:
    run_args += ["--device", "cuda"]
run_args += ["--output_dir", "/kaggle/working/fme-results"]
subprocess.run([sys.executable, "train.py", *run_args], check=True)
'''
    (destination / "fme_kaggle.py").write_text(runner, encoding="utf-8")
    metadata: dict[str, object] = {
        "title": args.title,
        "code_file": "fme_kaggle.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true" if args.enable_internet else "false",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": args.data_source,
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    if kernel_slug:
        metadata["id"] = kernel_slug
    else:
        metadata["id_no"] = kernel_number
    (destination / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared Kaggle kernel folder: {destination}")
    print("Review kernel-metadata.json, then push with: kaggle kernels push -p " + str(destination) + " --accelerator NvidiaTeslaT4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
