"""Download and parse CRTC Retail Mobile Sector open data."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.config import CRTC_MOBILE_URL, DATA_PROCESSED, DATA_RAW, SEEDS
from src.ingest.utils import (
    download_file,
    extract_zip,
    pct_to_float,
    read_crtc_csv,
    upload_to_gcs,
    utc_now_iso,
    write_parquet,
    year_from_column,
)


def _parse_provider_market_share(csv_path: Path) -> pd.DataFrame:
    """Parse MB-S1 provider revenue/subscriber share tables."""
    raw = read_crtc_csv(csv_path)
    rows: list[dict] = []
    section = None
    header_years: list[str] = []

    for _, row in raw.iterrows():
        cells = [str(c).strip() if pd.notna(c) else "" for c in row]
        joined = " ".join(cells)

        if "Retail mobile revenue market share" in joined:
            section = "revenue_share"
            header_years = []
            continue
        if "Retail mobile subscriber market share" in joined:
            section = "subscriber_share"
            header_years = []
            continue
        if cells[0] == "Service provider" and section:
            header_years = cells[1:]
            continue
        if section and cells[0] in {"Bell", "TELUS", "Rogers", "Other providers"}:
            provider = cells[0]
            sub_metric = "revenue_share" if section == "revenue_share" else "subscriber_share"
            for col, val in zip(header_years, cells[1:]):
                year = year_from_column(col)
                if year is None:
                    continue
                rows.append(
                    {
                        "source_file": csv_path.name,
                        "metric_type": sub_metric,
                        "provider": provider,
                        "year": year,
                        "value_pct": pct_to_float(val),
                        "ingest_ts": utc_now_iso(),
                    }
                )
    return pd.DataFrame(rows)


def _parse_churn_rates(csv_path: Path) -> pd.DataFrame:
    """Parse MB-F17 Top 3 vs other churn rates."""
    raw = read_crtc_csv(csv_path)
    rows: list[dict] = []
    for _, row in raw.iterrows():
        cells = [str(c).strip() if pd.notna(c) else "" for c in row]
        if re.fullmatch(r"\d{4}.*", cells[0]):
            year = year_from_column(cells[0])
            if year is None:
                continue
            for segment, val in [("Top 3", cells[1]), ("Other providers", cells[2])]:
                rate = pct_to_float(val)
                if rate is not None:
                    rows.append(
                        {
                            "source_file": csv_path.name,
                            "metric_type": "blended_monthly_churn",
                            "segment": segment,
                            "year": year,
                            "value_pct": rate,
                            "ingest_ts": utc_now_iso(),
                        }
                    )
    return pd.DataFrame(rows)


def _province_code(name: str) -> str:
    mapping_path = SEEDS / "province_mapping.csv"
    if mapping_path.exists():
        mapping = pd.read_csv(mapping_path)
        match = mapping[mapping["province_name"] == name]
        if not match.empty:
            return str(match.iloc[0]["province_code"])
    return name


def _parse_provincial_share(csv_path: Path) -> pd.DataFrame:
    """Parse MB-F5 provincial market share into long format."""
    raw = read_crtc_csv(csv_path)
    rows: list[dict] = []
    providers = ["Bell Group", "TELUS", "Rogers", "Others"]
    year_starts: list[tuple[int, int]] = []

    for _, row in raw.iterrows():
        cells = [str(c).strip() if pd.notna(c) else "" for c in row]

        if not year_starts and cells[0] == "" and any(year_from_column(c) for c in cells[1:5]):
            for idx, cell in enumerate(cells):
                year = year_from_column(cell)
                if year is not None:
                    year_starts.append((idx, year))
            continue

        if cells[0] in {"", "Province/territory"} or cells[0].startswith("Source"):
            continue

        province = cells[0]
        if not province or province.startswith("Note"):
            continue

        for start_idx, year in year_starts:
            for offset, prov in enumerate(providers):
                col_idx = start_idx + offset
                if col_idx >= len(cells):
                    continue
                share = pct_to_float(cells[col_idx])
                if share is not None:
                    rows.append(
                        {
                            "source_file": csv_path.name,
                            "metric_type": "provincial_subscriber_share",
                            "province": _province_code(province),
                            "province_name": province,
                            "provider_group": prov,
                            "year": year,
                            "value_pct": share,
                            "ingest_ts": utc_now_iso(),
                        }
                    )
    return pd.DataFrame(rows)


def _parse_total_subscribers(csv_path: Path) -> pd.DataFrame:
    """Parse MB-S5 total subscriber counts."""
    raw = read_crtc_csv(csv_path)
    rows: list[dict] = []
    header_years: list[str] = []

    for _, row in raw.iterrows():
        cells = [str(c).strip() if pd.notna(c) else "" for c in row]
        if cells[0] == "Type":
            header_years = cells[1:]
            continue
        if cells[0] == "Subscribers":
            for col, val in zip(header_years, cells[1:]):
                year = year_from_column(col)
                if year is None:
                    continue
                text = val.replace(",", "").strip()
                if text:
                    rows.append(
                        {
                            "source_file": csv_path.name,
                            "metric_type": "total_subscribers_millions",
                            "year": year,
                            "value_millions": float(text),
                            "ingest_ts": utc_now_iso(),
                        }
                    )
    return pd.DataFrame(rows)


def ingest_crtc_mobile(
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
    skip_download: bool = False,
) -> dict[str, Path]:
    raw_dir = raw_dir or DATA_RAW / "crtc"
    processed_dir = processed_dir or DATA_PROCESSED / "crtc"
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / "data-retail-mobile-sector.zip"
    extract_dir = raw_dir / "mobile_extracted"

    if not skip_download or not zip_path.exists():
        download_file(CRTC_MOBILE_URL, zip_path)
    extract_zip(zip_path, extract_dir)

    outputs: dict[str, Path] = {}
    parsers = {
        "market_share": (_parse_provider_market_share, "MB-S1.csv"),
        "churn": (_parse_churn_rates, "MB-F17.csv"),
        "provincial_share": (_parse_provincial_share, "MB-F5.csv"),
        "total_subscribers": (_parse_total_subscribers, "MB-S5.csv"),
    }

    frames: list[pd.DataFrame] = []
    for name, (parser, fname) in parsers.items():
        path = extract_dir / fname
        if not path.exists():
            continue
        df = parser(path)
        out = write_parquet(df, processed_dir / f"crtc_{name}.parquet")
        outputs[name] = out
        frames.append(df.assign(table=name))

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined_path = write_parquet(combined, processed_dir / "stg_crtc_retail_mobile.parquet")
        outputs["staging"] = combined_path

        from src.config import GCS_BUCKET, USE_GCS

        if USE_GCS and GCS_BUCKET:
            upload_to_gcs(combined_path, f"gs://{GCS_BUCKET}/raw/crtc/stg_crtc_retail_mobile.parquet")

    return outputs


if __name__ == "__main__":
    paths = ingest_crtc_mobile()
    for k, v in paths.items():
        print(f"  {k}: {v}")
