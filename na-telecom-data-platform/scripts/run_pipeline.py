#!/usr/bin/env python3
"""Run full NA telecom pipeline end-to-end (local mode)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingest.crtc_ingest import ingest_crtc_mobile
from src.ingest.fcc_ingest import ingest_fcc_county
from src.ingest.generate_subscriptions import generate_subscriptions
from src.quality.expectations import all_passed, run_quality_checks, write_quality_report
from src.report.generate_summary_report import generate_summary_report
from src.transform.local_warehouse import build_local_warehouse, export_mart_samples


def main() -> int:
    print("=== NA Telecom Pipeline ===\n")

    print("1/4 Ingesting sources...")
    ingest_crtc_mobile(skip_download=True)
    ingest_fcc_county(skip_download=True, use_sample=True)
    generate_subscriptions()

    print("2/4 Building local warehouse...")
    db = build_local_warehouse()
    print(f"     {db}")

    print("3/4 Running quality checks...")
    results = run_quality_checks(db)
    report = write_quality_report(results)
    for r in results:
        print(f"     {'PASS' if r.passed else 'FAIL'}: {r.name}")

    print("4/5 Exporting mart samples...")
    samples = export_mart_samples()
    for name, df in samples.items():
        print(f"     {name}: {len(df)} rows")

    print("5/5 Generating summary report...")
    report_md = generate_summary_report(db)
    print(f"     {report_md}")

    print(f"\nQuality report: {report}")
    if not all_passed(results):
        print("\nSome quality checks failed.", file=sys.stderr)
        return 1
    print("\nPipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
