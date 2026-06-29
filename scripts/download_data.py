#!/usr/bin/env python3
"""Download datasets for all portfolio projects.

Skips files that already exist. Uses kagglehub for SF Salaries (no API key required).
"""

from __future__ import annotations

import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DOWNLOADS: list[tuple[Path, str]] = [
    (
        ROOT / "linear_regression/data/Ecommerce Customers.csv",
        "https://raw.githubusercontent.com/araj2/customer-database/master/Ecommerce%20Customers.csv",
    ),
    (
        ROOT / "ecom_purchases/data/Ecommerce Purchases.csv",
        "https://raw.githubusercontent.com/reza16977/Ecommerce-Purchases-Exercise/main/Ecommerce%20Purchases",
    ),
    (
        ROOT / "logistic_regression/data/advertising.csv",
        "https://raw.githubusercontent.com/anishapareek/ad-clicks-prediction/main/advertising.csv",
    ),
    (
        ROOT / "k_means_clustering/data/Mall_Customers.csv",
        "https://raw.githubusercontent.com/kennedykwangari/Mall-Customer-Segmentation-Data/master/Mall_Customers.csv",
    ),
    (
        ROOT / "heart_disease_pca/data/hungarian.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data",
    ),
    (
        ROOT / "911_calls_analysis/data/911.csv",
        "https://raw.githubusercontent.com/Nelly2i/Data-Analysis-Projects/main/911.csv",
    ),
]


def fetch_url(dest: Path, url: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {dest.name} ...")
    urllib.request.urlretrieve(url, dest)


def fetch_salaries(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print("  downloading Salaries.csv via kagglehub ...")
    try:
        import kagglehub
    except ImportError:
        raise SystemExit(
            "kagglehub is required for Salaries.csv. Run: pip install kagglehub"
        ) from None
    cache = Path(kagglehub.dataset_download("kaggle/sf-salaries"))
    source = next(cache.rglob("Salaries.csv"))
    shutil.copy2(source, dest)


def main() -> int:
    missing = 0
    print("Fetching portfolio datasets into data/ folders...\n")

    for dest, url in DOWNLOADS:
        if dest.exists():
            print(f"  skip {dest.relative_to(ROOT)} (exists)")
            continue
        try:
            fetch_url(dest, url)
            print(f"  ok   {dest.relative_to(ROOT)}")
        except Exception as exc:
            print(f"  FAIL {dest.relative_to(ROOT)}: {exc}", file=sys.stderr)
            missing += 1

    sal_path = ROOT / "salary_analysis/data/Salaries.csv"
    if sal_path.exists():
        print(f"  skip {sal_path.relative_to(ROOT)} (exists)")
    else:
        try:
            fetch_salaries(sal_path)
            print(f"  ok   {sal_path.relative_to(ROOT)}")
        except Exception as exc:
            print(f"  FAIL {sal_path.relative_to(ROOT)}: {exc}", file=sys.stderr)
            missing += 1

    bundled = [
        ROOT / "insurance_prediction/data/insurance.csv",
        ROOT / "house_price_predictor/data/BostonHousing.xls",
    ]
    for path in bundled:
        status = "ok" if path.exists() else "MISSING"
        print(f"  {status:4} {path.relative_to(ROOT)} (bundled)")

    print()
    if missing:
        print(f"Finished with {missing} failure(s).")
        return 1
    print("All datasets ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
