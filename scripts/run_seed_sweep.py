#!/usr/bin/env python3
"""Run the same experiment under declared seeds; does not alter the test protocol."""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--output_root", default="results/seed-sweep")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    for seed in args.seeds:
        destination = Path(args.output_root) / f"seed-{seed}"
        command = [sys.executable, str(root / "train.py"), "--data_file", args.data_file,
                   "--output_dir", str(destination), "--seed", str(seed), "--epochs", str(args.epochs), "--ablations"]
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, cwd=root, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
