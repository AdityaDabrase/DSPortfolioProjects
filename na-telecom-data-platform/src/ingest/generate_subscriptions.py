"""Generate synthetic subscriber operations calibrated to CRTC benchmarks."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ARPU_CAD,
    CA_PROVINCES,
    DATA_PROCESSED,
    SEEDS,
    TOP3_CHURN_RATE,
    TOTAL_CA_SUBSCRIBERS_M,
)
from src.ingest.utils import upload_to_gcs, utc_now_iso, write_parquet

RNG = np.random.default_rng(42)
PLAN_TIERS = ("basic", "standard", "premium", "unlimited")


def _load_carriers() -> pd.DataFrame:
    carriers = pd.read_csv(SEEDS / "carrier_mapping.csv")
    ca = carriers[carriers["country"] == "CA"].copy()
    us = carriers[carriers["country"] == "US"].copy()
    return ca, us


def generate_subscriptions(
    n_subscribers: int = 75_000,
    snapshot_date: date | None = None,
    processed_dir: Path | None = None,
) -> Path:
    processed_dir = processed_dir or DATA_PROCESSED / "subscriptions"
    snapshot_date = snapshot_date or date.today()
    ca_carriers, us_carriers = _load_carriers()

    ca_weights = ca_carriers["market_weight_ca"].values
    ca_weights = ca_weights / ca_weights.sum()
    us_weights = us_carriers["market_weight_us"].values
    us_weights = us_weights / us_weights.sum()

    n_ca = int(n_subscribers * 0.65)
    n_us = n_subscribers - n_ca

    records: list[dict] = []

    for i in range(n_ca):
        carrier = ca_carriers.iloc[RNG.choice(len(ca_carriers), p=ca_weights)]
        province = RNG.choice(list(CA_PROVINCES))
        days_active = int(RNG.integers(30, 365 * 4))
        activation = snapshot_date - timedelta(days=days_active)
        churn_rate = TOP3_CHURN_RATE if carrier["is_mno"] else TOP3_CHURN_RATE * 1.1
        churned = RNG.random() < churn_rate
        churn_date = snapshot_date - timedelta(days=int(RNG.integers(1, 30))) if churned else None
        arpu = ARPU_CAD * float(RNG.uniform(0.75, 1.35))

        records.append(
            {
                "subscriber_id": f"CA-{i:06d}",
                "carrier_id": carrier["carrier_id"],
                "carrier_name": carrier["display_name"],
                "parent_group": carrier["parent_group"],
                "country": "CA",
                "region_code": province,
                "plan_tier": RNG.choice(PLAN_TIERS, p=[0.1, 0.35, 0.35, 0.2]),
                "activation_date": activation.isoformat(),
                "churn_date": churn_date.isoformat() if churn_date else None,
                "is_churned": int(churned),
                "is_postpaid": int(RNG.random() < 0.88),
                "monthly_revenue_cad": round(arpu, 2),
                "snapshot_date": snapshot_date.isoformat(),
                "ingest_ts": utc_now_iso(),
            }
        )

    for i in range(n_us):
        carrier = us_carriers.iloc[RNG.choice(len(us_carriers), p=us_weights)]
        state = RNG.choice(["WA", "NY", "MI", "CA", "TX", "FL"])
        days_active = int(RNG.integers(30, 365 * 4))
        activation = snapshot_date - timedelta(days=days_active)
        churn_rate = TOP3_CHURN_RATE * float(RNG.uniform(0.95, 1.15))
        churned = RNG.random() < churn_rate
        churn_date = snapshot_date - timedelta(days=int(RNG.integers(1, 30))) if churned else None
        arpu_usd = (ARPU_CAD * 0.74) * float(RNG.uniform(0.8, 1.4))

        records.append(
            {
                "subscriber_id": f"US-{i:06d}",
                "carrier_id": carrier["carrier_id"],
                "carrier_name": carrier["display_name"],
                "parent_group": carrier["parent_group"],
                "country": "US",
                "region_code": state,
                "plan_tier": RNG.choice(PLAN_TIERS, p=[0.08, 0.32, 0.38, 0.22]),
                "activation_date": activation.isoformat(),
                "churn_date": churn_date.isoformat() if churn_date else None,
                "is_churned": int(churned),
                "is_postpaid": int(RNG.random() < 0.85),
                "monthly_revenue_cad": round(arpu_usd, 2),
                "snapshot_date": snapshot_date.isoformat(),
                "ingest_ts": utc_now_iso(),
            }
        )

    df = pd.DataFrame(records)

    # Scale CA count to approximate CRTC total subscriber benchmark (millions)
    scale_factor = (TOTAL_CA_SUBSCRIBERS_M * 1_000_000) / max(n_ca, 1)
    df["weight_to_population"] = np.where(df["country"] == "CA", scale_factor, 1.0)

    out = write_parquet(df, processed_dir / "stg_subscriptions_daily.parquet")

    from src.config import GCS_BUCKET, USE_GCS

    if USE_GCS and GCS_BUCKET:
        upload_to_gcs(out, f"gs://{GCS_BUCKET}/raw/subscriptions/stg_subscriptions_daily.parquet")

    return out


if __name__ == "__main__":
    path = generate_subscriptions()
    print(f"  subscriptions: {path}")
