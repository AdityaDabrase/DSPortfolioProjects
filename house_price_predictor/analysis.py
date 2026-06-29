"""Boston Housing data quality analysis.

Explores missing values and outliers in the Boston Housing dataset used
in the project report. Saves summary figures to assets/.

Note: The classic Boston Housing dataset has known ethical limitations
(proxies for socioeconomic factors). This project uses a modified version
for missing-data exercises only.

Usage:
    python analysis.py [--data data/BostonHousing.xls]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Data", header=0)
    return df.apply(pd.to_numeric, errors="coerce")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/BostonHousing.xls"))
    args = parser.parse_args()

    df = load_data(args.data)
    assets = Path("assets")
    assets.mkdir(exist_ok=True)

    print(f"Shape: {df.shape}")
    print("\nMissing values per column:")
    missing = df.isna().sum().sort_values(ascending=False)
    print(missing[missing > 0].to_string() if missing.any() else "None")

    fig, ax = plt.subplots(figsize=(10, 5))
    missing[missing > 0].sort_values(ascending=True).plot.barh(ax=ax, color="steelblue")
    ax.set_title("Missing values by column")
    ax.set_xlabel("Count")
    fig.tight_layout()
    fig.savefig(assets / "missing_values.png", dpi=150)
    plt.close(fig)

    if "PTRATIO" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, y="PTRATIO", ax=ax, color="coral")
        ax.set_title("PTRATIO distribution (outlier check)")
        fig.tight_layout()
        fig.savefig(assets / "ptratio_outliers.png", dpi=150)
        plt.close(fig)

    print(f"\nFigures saved to {assets}/")


if __name__ == "__main__":
    main()
