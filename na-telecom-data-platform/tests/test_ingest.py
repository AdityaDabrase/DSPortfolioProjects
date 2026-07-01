"""Tests for NA telecom ingest layer."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingest.crtc_ingest import ingest_crtc_mobile
from src.ingest.fcc_ingest import ingest_fcc_county
from src.ingest.generate_subscriptions import generate_subscriptions


@pytest.fixture(scope="module")
def crtc_staging():
    outputs = ingest_crtc_mobile(skip_download=True)
    return pd.read_parquet(outputs["staging"])


def test_crtc_market_share_has_big_three(crtc_staging):
    ms = crtc_staging[crtc_staging["table"] == "market_share"]
    providers = set(ms["provider"].dropna())
    assert {"Bell", "TELUS", "Rogers"}.issubset(providers)


def test_crtc_provincial_share_not_empty(crtc_staging):
    prov = crtc_staging[crtc_staging["table"] == "provincial_share"]
    assert len(prov) > 100
    assert "BC" in set(prov["province"].dropna())


def test_crtc_churn_benchmark(crtc_staging):
    churn = crtc_staging[crtc_staging["table"] == "churn"]
    top3 = churn[churn["segment"] == "Top 3"]
    assert not top3.empty
    assert top3["value_pct"].max() < 2.0


def test_fcc_ingest_produces_rows():
    path = ingest_fcc_county(skip_download=True, use_sample=True)
    df = pd.read_parquet(path)
    assert len(df) >= 10
    assert set(df["state_abbr"].unique()) <= {"WA", "NY", "MI"}


def test_subscriptions_shape():
    path = generate_subscriptions(n_subscribers=1000)
    df = pd.read_parquet(path)
    assert len(df) == 1000
    assert df["is_churned"].mean() < 0.05
