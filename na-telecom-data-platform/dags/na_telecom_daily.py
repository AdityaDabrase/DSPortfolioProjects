"""Airflow DAG: North America telecom market intelligence pipeline."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

default_args = {
    "owner": "na-telecom",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


def _ingest_crtc(**context):
    from src.ingest.crtc_ingest import ingest_crtc_mobile

    outputs = ingest_crtc_mobile(skip_download=context.get("skip_download", False))
    return {k: str(v) for k, v in outputs.items()}


def _ingest_fcc(**context):
    from src.ingest.fcc_ingest import ingest_fcc_county

    path = ingest_fcc_county(
        skip_download=context.get("skip_download", False),
        use_sample=context.get("use_sample", False),
    )
    return str(path)


def _generate_subscriptions(**context):
    from src.ingest.generate_subscriptions import generate_subscriptions

    path = generate_subscriptions()
    return str(path)


def _build_warehouse(**context):
    from src.transform.local_warehouse import build_local_warehouse

    db = build_local_warehouse()
    return str(db)


def _run_quality(**context):
    from src.quality.expectations import all_passed, run_quality_checks, write_quality_report

    results = run_quality_checks()
    report = write_quality_report(results)
    if not all_passed(results):
        failed = [r.name for r in results if not r.passed]
        raise RuntimeError(f"Quality checks failed: {failed}")
    return str(report)


def _load_bigquery(**context):
    from src.config import USE_BIGQUERY
    from src.transform.bq_load import load_all_staging, run_bq_transforms

    if not USE_BIGQUERY:
        return "skipped (USE_BIGQUERY=0)"
    tables = load_all_staging()
    run_bq_transforms()
    return tables


def _generate_summary_report(**context):
    from src.transform.local_warehouse import build_local_warehouse
    from src.report.generate_summary_report import generate_summary_report

    db = build_local_warehouse()
    path = generate_summary_report(db)
    return str(path)


def _update_freshness(**context):
    from src.ingest.utils import utc_now_iso

    ts = utc_now_iso()
    context["ti"].xcom_push(key="freshness_ts", value=ts)
    return ts


with DAG(
    dag_id="na_telecom_pipeline",
    default_args=default_args,
    description="Ingest CRTC/FCC telecom data and build NA market marts",
    schedule_interval="0 6 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["telecom", "crtc", "fcc", "na"],
) as dag:
    with TaskGroup("canada_market") as canada_market:
        ingest_crtc = PythonOperator(
            task_id="ingest_crtc",
            python_callable=_ingest_crtc,
            op_kwargs={"skip_download": False},
        )

    with TaskGroup("us_market") as us_market:
        ingest_fcc = PythonOperator(
            task_id="ingest_fcc",
            python_callable=_ingest_fcc,
            op_kwargs={"skip_download": False, "use_sample": False},
        )

    with TaskGroup("operational") as operational:
        generate_subs = PythonOperator(
            task_id="generate_subscriptions",
            python_callable=_generate_subscriptions,
        )

    with TaskGroup("load_staging") as load_staging:
        build_warehouse = PythonOperator(
            task_id="build_local_warehouse",
            python_callable=_build_warehouse,
        )
        load_bq = PythonOperator(
            task_id="load_bigquery",
            python_callable=_load_bigquery,
        )

    with TaskGroup("run_transforms") as run_transforms:
        transforms_done = PythonOperator(
            task_id="transforms_complete",
            python_callable=lambda **ctx: "transforms applied in build_local_warehouse",
        )

    with TaskGroup("run_quality_checks") as run_quality_checks:
        quality = PythonOperator(
            task_id="run_expectations",
            python_callable=_run_quality,
        )

    summary_report = PythonOperator(
        task_id="generate_summary_report",
        python_callable=_generate_summary_report,
    )

    freshness = PythonOperator(
        task_id="update_freshness_metadata",
        python_callable=_update_freshness,
    )

    [canada_market, us_market, operational] >> load_staging
    build_warehouse >> load_bq >> run_transforms >> run_quality_checks >> summary_report >> freshness
