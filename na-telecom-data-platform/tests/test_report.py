"""Tests for summary report generation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.report.generate_summary_report import REPORT_PATH, generate_summary_report


def test_summary_report_generates():
    path = generate_summary_report()
    assert path.exists()
    text = path.read_text()
    assert "## Conclusion" in text
    assert "Executive summary" in text
    assets = ROOT / "assets" / "report"
    for chart in (
        "national_subscriber_share.png",
        "provincial_share_latest.png",
        "subscriber_growth.png",
        "churn_comparison.png",
        "us_border_broadband.png",
        "cross_border_summary.png",
    ):
        assert (assets / chart).exists(), f"Missing chart: {chart}"
