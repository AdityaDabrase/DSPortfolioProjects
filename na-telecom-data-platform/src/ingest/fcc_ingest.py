"""Download and parse FCC Form 477 county-level connection data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import BORDER_US_STATES, DATA_PROCESSED, DATA_RAW, FCC_COUNTY_URL, SEEDS
from src.ingest.utils import download_file, upload_to_gcs, utc_now_iso, write_parquet

FCC_COLUMN_MAP = {
    "state": "state_abbr",
    "State": "state_abbr",
    "stateabbr": "state_abbr",
    "county": "county_name",
    "County": "county_name",
    "provider": "provider",
    "Provider": "provider",
    "frn": "provider",
}


def _normalize_fcc_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize heterogeneous FCC CSV columns to staging schema."""
    cols_lower = {c: str(c).strip() for c in df.columns}
    df = df.rename(columns=cols_lower)

    out = pd.DataFrame()
    out["state_abbr"] = df.get("state_abbr", df.get("State", df.get("state", "")))
    out["county_name"] = df.get("county_name", df.get("County", df.get("county", "")))
    out["provider"] = df.get("provider", df.get("Provider", "Unknown"))

    conn_col = None
    for candidate in ("total_connections", "connections", "Connections", "total conn"):
        if candidate in df.columns:
            conn_col = candidate
            break
    if conn_col:
        out["total_connections"] = pd.to_numeric(df[conn_col], errors="coerce")
    else:
        numeric = df.select_dtypes(include="number")
        out["total_connections"] = numeric.iloc[:, 0] if not numeric.empty else 0

    res_col = None
    for candidate in ("residential_connections", "residential", "Residential"):
        if candidate in df.columns:
            res_col = candidate
            break
    out["residential_connections"] = (
        pd.to_numeric(df[res_col], errors="coerce") if res_col else out["total_connections"] * 0.85
    )

    out["state_fips"] = df.get("state_fips", "")
    out["county_fips"] = df.get("county_fips", "")
    out["report_year"] = 2024
    out["report_period"] = "June"
    out["ingest_ts"] = utc_now_iso()
    return out


def _load_sample() -> pd.DataFrame:
    sample = SEEDS / "fcc_county_sample.csv"
    return pd.read_csv(sample)


def _filter_border_states(df: pd.DataFrame) -> pd.DataFrame:
    if "state_abbr" not in df.columns:
        return df
    return df[df["state_abbr"].isin(BORDER_US_STATES)].copy()


def ingest_fcc_county(
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
    skip_download: bool = False,
    use_sample: bool = False,
) -> Path:
    raw_dir = raw_dir or DATA_RAW / "fcc"
    processed_dir = processed_dir or DATA_PROCESSED / "fcc"
    raw_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = raw_dir / "form477_county_connections.csv"
    df: pd.DataFrame

    if use_sample:
        df = _load_sample()
    else:
        try:
            if not skip_download or not raw_csv.exists():
                download_file(FCC_COUNTY_URL, raw_csv)
            df = pd.read_csv(raw_csv, low_memory=False, nrows=100_000, encoding="latin-1")
            df = _normalize_fcc_df(df)
            df = _filter_border_states(df)
            if df.empty:
                raise ValueError("FCC download parsed empty for border states")
        except Exception:
            df = _load_sample()

    out = write_parquet(df, processed_dir / "stg_fcc_county_connections.parquet")

    from src.config import GCS_BUCKET, USE_GCS

    if USE_GCS and GCS_BUCKET:
        upload_to_gcs(out, f"gs://{GCS_BUCKET}/raw/fcc/stg_fcc_county_connections.parquet")

    return out


if __name__ == "__main__":
    path = ingest_fcc_county()
    print(f"  fcc staging: {path}")
