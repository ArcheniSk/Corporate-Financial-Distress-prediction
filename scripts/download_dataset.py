#!/usr/bin/env python3
"""Download the original candidate benchmark locally (never into version control)."""

import argparse
from pathlib import Path
from urllib.request import urlopen


SOURCE_COMMIT = "8bcd8db3b432b1fa6d65e26753b5c9fc6567438d"
URL = f"https://raw.githubusercontent.com/sowide/bankruptcy_dataset/{SOURCE_COMMIT}/american_bankruptcy_dataset.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/raw/american_bankruptcy_dataset.csv")
    args = parser.parse_args()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(URL, timeout=60) as response, destination.open("wb") as out:
        while chunk := response.read(1024 * 1024):
            out.write(chunk)
    print(f"Downloaded pinned source {SOURCE_COMMIT} to {destination}")
    print("Run scripts/audit_dataset.py before training. The raw file is excluded from Git.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
