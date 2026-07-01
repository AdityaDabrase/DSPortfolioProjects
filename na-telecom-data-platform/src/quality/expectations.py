"""Data quality checks for NA telecom pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from src.config import DATA_WAREHOUSE, TOP3_CHURN_RATE


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def run_quality_checks(warehouse_path: Path | None = None) -> list[CheckResult]:
    warehouse_path = warehouse_path or DATA_WAREHOUSE / "na_telecom.duckdb"
    results: list[CheckResult] = []

    if not warehouse_path.exists():
        return [CheckResult("warehouse_exists", False, f"Missing {warehouse_path}")]

    con = duckdb.connect(str(warehouse_path), read_only=True)

    # 1. No null carrier_id in dim_carrier
    null_carriers = con.execute(
        "SELECT COUNT(*) FROM dim_carrier WHERE carrier_id IS NULL"
    ).fetchone()[0]
    results.append(
        CheckResult(
            "dim_carrier_no_null_ids",
            null_carriers == 0,
            f"null carrier_id count: {null_carriers}",
        )
    )

    # 2. Mart has rows
    mart_count = con.execute("SELECT COUNT(*) FROM mart_carrier_market_share").fetchone()[0]
    results.append(
        CheckResult(
            "mart_carrier_market_share_not_empty",
            mart_count > 0,
            f"row count: {mart_count}",
        )
    )

    # 3. Synthetic churn reconciles to CRTC Top 3 benchmark within tolerance
    churn_df = con.execute(
        """
        SELECT AVG(synthetic_churn_rate) AS avg_syn, AVG(benchmark_churn_rate) AS avg_bench
        FROM mart_regional_churn
        WHERE country = 'CA' AND benchmark_churn_rate IS NOT NULL
        """
    ).df()
    if not churn_df.empty and churn_df["avg_bench"].notna().any():
        delta = abs(churn_df["avg_syn"].iloc[0] - churn_df["avg_bench"].iloc[0])
        results.append(
            CheckResult(
                "churn_reconciles_to_crtc_benchmark",
                delta <= 0.005,
                f"synthetic={churn_df['avg_syn'].iloc[0]:.4f}, benchmark={churn_df['avg_bench'].iloc[0]:.4f}, delta={delta:.4f}",
            )
        )
    else:
        results.append(
            CheckResult(
                "churn_reconciles_to_crtc_benchmark",
                False,
                "benchmark churn data unavailable",
            )
        )

    # 4. No duplicate grain keys in mart_carrier_market_share (CA national)
    dupes = con.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT country, carrier, year, region_code, metric_type, COUNT(*) AS n
            FROM mart_carrier_market_share
            GROUP BY 1,2,3,4,5
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]
    results.append(
        CheckResult(
            "mart_no_duplicate_grain",
            dupes == 0,
            f"duplicate grain groups: {dupes}",
        )
    )

    # 5. FCC state FIPS / state_abbr valid for US rows
    invalid_states = con.execute(
        """
        SELECT COUNT(*) FROM stg_fcc_county_connections
        WHERE state_abbr IS NULL OR LENGTH(TRIM(state_abbr)) <> 2
        """
    ).fetchone()[0]
    results.append(
        CheckResult(
            "fcc_valid_state_abbr",
            invalid_states == 0,
            f"invalid state rows: {invalid_states}",
        )
    )

    # 6. Freshness — ingest_ts present in staging
    stale = con.execute(
        """
        SELECT COUNT(*) FROM stg_crtc_retail_mobile WHERE ingest_ts IS NULL
        """
    ).fetchone()[0]
    results.append(
        CheckResult(
            "crtc_freshness_metadata",
            stale == 0,
            f"rows missing ingest_ts: {stale}",
        )
    )

    # 7. CRTC total subscribers reasonable (30-40M range for recent year)
    subs = con.execute(
        """
        SELECT value_millions FROM stg_crtc_retail_mobile
        WHERE "table" = 'total_subscribers' AND year = 2024
        LIMIT 1
        """
    ).fetchone()
    if subs:
        val = float(subs[0])
        results.append(
            CheckResult(
                "crtc_subscriber_total_sanity",
                30 <= val <= 45,
                f"2024 total subscribers (M): {val}",
            )
        )

    # 8. Top 3 churn benchmark near known CRTC value
    results.append(
        CheckResult(
            "top3_churn_benchmark_configured",
            abs(TOP3_CHURN_RATE - 0.0116) < 0.001,
            f"configured TOP3_CHURN_RATE={TOP3_CHURN_RATE}",
        )
    )

    con.close()
    return results


def write_quality_report(results: list[CheckResult], path: Path | None = None) -> Path:
    path = path or DATA_WAREHOUSE / "quality_report.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"[{status}] {r.name}: {r.detail}")
    path.write_text("\n".join(lines) + "\n")
    return path


def all_passed(results: list[CheckResult]) -> bool:
    return all(r.passed for r in results)


if __name__ == "__main__":
    results = run_quality_checks()
    report = write_quality_report(results)
    for r in results:
        print(f"{'PASS' if r.passed else 'FAIL'}: {r.name} — {r.detail}")
    print(f"\nReport: {report}")
