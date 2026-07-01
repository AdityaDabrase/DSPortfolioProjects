"""Load processed parquet into local DuckDB warehouse and build marts."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src.config import DATA_PROCESSED, DATA_WAREHOUSE, SEEDS, SQL_DIR


def _read_sql(name: str) -> str:
    path = SQL_DIR / name
    return path.read_text()


def build_local_warehouse(
    processed_dir: Path | None = None,
    warehouse_dir: Path | None = None,
) -> Path:
    processed_dir = processed_dir or DATA_PROCESSED
    warehouse_dir = warehouse_dir or DATA_WAREHOUSE
    warehouse_dir.mkdir(parents=True, exist_ok=True)
    db_path = warehouse_dir / "na_telecom.duckdb"

    con = duckdb.connect(str(db_path))

    # Staging loads from parquet
    crtc_path = processed_dir / "crtc" / "stg_crtc_retail_mobile.parquet"
    fcc_path = processed_dir / "fcc" / "stg_fcc_county_connections.parquet"
    subs_path = processed_dir / "subscriptions" / "stg_subscriptions_daily.parquet"
    carriers_path = SEEDS / "carrier_mapping.csv"

    con.execute(
        f"CREATE OR REPLACE TABLE stg_crtc_retail_mobile AS "
        f"SELECT * FROM read_parquet('{crtc_path.as_posix()}')"
    )
    con.execute(
        f"CREATE OR REPLACE TABLE stg_fcc_county_connections AS "
        f"SELECT * FROM read_parquet('{fcc_path.as_posix()}')"
    )
    con.execute(
        f"CREATE OR REPLACE TABLE stg_subscriptions_daily AS "
        f"SELECT * FROM read_parquet('{subs_path.as_posix()}')"
    )
    con.execute(
        f"CREATE OR REPLACE TABLE seed_carrier_mapping AS "
        f"SELECT * FROM read_csv('{carriers_path.as_posix()}', header=true, auto_detect=true)"
    )

    # Dimensions
    con.execute(_read_sql("dimensions/dim_carrier.sql"))
    con.execute(_read_sql("dimensions/dim_region.sql"))
    con.execute(_read_sql("dimensions/dim_date.sql"))

    # Facts
    con.execute(_read_sql("staging/stg_market_metrics.sql"))
    con.execute(_read_sql("staging/stg_subscriber_snapshot.sql"))

    # Marts
    con.execute(_read_sql("marts/mart_carrier_market_share.sql"))
    con.execute(_read_sql("marts/mart_regional_churn.sql"))
    con.execute(_read_sql("marts/mart_cross_border_summary.sql"))

    # Export marts to parquet for README / inspection
    for mart in ("mart_carrier_market_share", "mart_regional_churn", "mart_cross_border_summary"):
        out = warehouse_dir / f"{mart}.parquet"
        con.execute(f"COPY {mart} TO '{out.as_posix()}' (FORMAT PARQUET)")

    con.close()
    return db_path


def export_mart_samples(warehouse_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    warehouse_dir = warehouse_dir or DATA_WAREHOUSE
    samples = {}
    for mart in ("mart_carrier_market_share", "mart_regional_churn", "mart_cross_border_summary"):
        path = warehouse_dir / f"{mart}.parquet"
        if path.exists():
            samples[mart] = pd.read_parquet(path)
    return samples


if __name__ == "__main__":
    db = build_local_warehouse()
    print(f"  warehouse: {db}")
