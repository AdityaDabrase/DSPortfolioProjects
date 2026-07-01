"""Project paths and environment configuration."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_WAREHOUSE = PROJECT_ROOT / "data" / "warehouse"
SEEDS = PROJECT_ROOT / "seeds"
SQL_DIR = PROJECT_ROOT / "sql"

CRTC_MOBILE_URL = (
    "https://applications.crtc.gc.ca/OpenData/CASP/"
    "COMMUNICATION%20MONITORING%20REPORTS/Retail%20Mobile%20Sector/English/"
    "data-retail-mobile-sector.zip"
)
CRTC_FIXED_URL = (
    "https://applications.crtc.gc.ca/OpenData/CASP/"
    "COMMUNICATION%20MONITORING%20REPORTS/Retail%20Fixed%20Internet%20Sector/English/"
    "data-retail-fixed-internet-sector.zip"
)
FCC_COUNTY_URL = (
    "https://www.fcc.gov/sites/default/files/"
    "form477_county_data_june2009_june2025.csv"
)

GCP_PROJECT = os.environ.get("GCP_PROJECT", "")
GCS_BUCKET = os.environ.get("GCS_BUCKET", "")
BQ_DATASET = os.environ.get("BQ_DATASET", "na_telecom")
USE_GCS = os.environ.get("USE_GCS", "0") == "1"
USE_BIGQUERY = os.environ.get("USE_BIGQUERY", "0") == "1"

BORDER_US_STATES = ("WA", "NY", "MI")
CA_PROVINCES = (
    "BC", "AB", "SK", "MB", "ON", "QC", "NB", "NS", "PE", "NL", "YT", "NT", "NU"
)

# CRTC benchmarks for synthetic subscriber calibration
TOP3_CHURN_RATE = 0.0116  # 1.16% monthly (2021 MP)
INDUSTRY_CHURN_RATE = 0.0122  # 1.22% monthly (2024 MP)
ARPU_CAD = 68.24
TOTAL_CA_SUBSCRIBERS_M = 37.7  # 2024 MP from CRTC MB-S5
