#!/usr/bin/env python3
"""Generate summary report with visualizations from pipeline marts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.report.generate_summary_report import generate_summary_report


def main() -> int:
    path = generate_summary_report()
    print(f"Summary report written to: {path}")
    print(f"Charts saved to: {ROOT / 'assets' / 'report'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
