#!/usr/bin/env python3
"""Audit a candidate panel before any modeling or result generation."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fme.data import DatasetAuditError, load_panel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_file", required=True)
    args = parser.parse_args()
    try:
        _, audit = load_panel(args.data_file, require_temporal_labels=False)
    except (DatasetAuditError, FileNotFoundError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(audit, indent=2))
    if audit["constant_within_company"]:
        print("BLOCKED: target is company-level constant, not a verified event-horizon label.", file=sys.stderr)
        return 2
    print("PASS: labels vary within companies; review event timing and censoring before modeling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
