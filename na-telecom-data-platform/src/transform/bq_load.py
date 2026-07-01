"""Load staging tables to BigQuery (optional cloud mode)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import BQ_DATASET, DATA_PROCESSED, GCP_PROJECT, USE_BIGQUERY


def _get_client():
    if not USE_BIGQUERY or not GCP_PROJECT:
        raise RuntimeError("Set USE_BIGQUERY=1 and GCP_PROJECT to enable BigQuery loads")
    from google.cloud import bigquery

    return bigquery.Client(project=GCP_PROJECT)


def load_parquet_to_bq(
    parquet_path: Path,
    table_id: str,
    write_disposition: str = "WRITE_TRUNCATE",
) -> str:
    client = _get_client()
    full_id = f"{GCP_PROJECT}.{BQ_DATASET}.{table_id}"
    job_config = client.load_table_from_uri if str(parquet_path).startswith("gs://") else None

    if job_config:
        from google.cloud import bigquery

        config = bigquery.LoadJobConfig(write_disposition=write_disposition, source_format=bigquery.SourceFormat.PARQUET)
        job = client.load_table_from_uri(str(parquet_path), full_id, job_config=config)
    else:
        df = pd.read_parquet(parquet_path)
        job = client.load_table_from_dataframe(df, full_id)

    job.result()
    return full_id


def run_bq_transforms() -> list[str]:
    """Execute mart SQL against BigQuery."""
    if not USE_BIGQUERY:
        return []

    client = _get_client()
    from src.transform.local_warehouse import _read_sql

    executed = []
    for sql_file in (
        "dimensions/dim_carrier.sql",
        "dimensions/dim_region.sql",
        "marts/mart_carrier_market_share.sql",
        "marts/mart_regional_churn.sql",
    ):
        sql = _read_sql(sql_file)
        client.query(sql).result()
        executed.append(sql_file)
    return executed


def load_all_staging(processed_dir: Path | None = None) -> list[str]:
    processed_dir = processed_dir or DATA_PROCESSED
    if not USE_BIGQUERY:
        return []

    loads = [
        (processed_dir / "crtc" / "stg_crtc_retail_mobile.parquet", "stg_crtc_retail_mobile"),
        (processed_dir / "fcc" / "stg_fcc_county_connections.parquet", "stg_fcc_county_connections"),
        (processed_dir / "subscriptions" / "stg_subscriptions_daily.parquet", "stg_subscriptions_daily"),
    ]
    return [load_parquet_to_bq(p, t) for p, t in loads if p.exists()]
