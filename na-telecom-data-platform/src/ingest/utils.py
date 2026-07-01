"""Shared ingest utilities."""

from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def download_file(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, dest)
    return dest


def extract_zip(zip_path: Path, dest_dir: Path) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    return sorted(dest_dir.glob("*.csv"))


def read_crtc_csv(path: Path) -> pd.DataFrame:
    """Read CRTC CSV with variable header rows."""
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, header=None, dtype=str, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, header=None, dtype=str, encoding="latin-1")


def pct_to_float(value: str) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().replace("%", "").replace(",", "")
    if not text or text.upper() in {"NA", "N/A", ""}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def year_from_column(col: str) -> int | None:
    text = str(col).strip()
    for token in text.replace("(MP)", "").split():
        if token.isdigit() and len(token) == 4:
            return int(token)
    return None


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def file_fingerprint(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def upload_to_gcs(local_path: Path, gcs_uri: str) -> None:
    """Upload file to GCS when USE_GCS is enabled."""
    from src.config import USE_GCS

    if not USE_GCS:
        return
    from google.cloud import storage

    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    bucket.blob(blob_name).upload_from_filename(str(local_path))
