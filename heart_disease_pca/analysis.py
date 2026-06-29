"""Heart disease classification with PCA dimensionality reduction.

Loads the UCI Hungarian heart disease subset, cleans missing values,
reduces features with PCA, and compares classifiers on the reduced space.

Usage:
    python analysis.py [--data data/hungarian.data]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]
FEATURES = [c for c in COLUMNS if c != "num"]
RANDOM_STATE = 42


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path, sep=",", names=COLUMNS, na_values="?")
    y = (df["num"] > 0).astype(int)
    X = df[FEATURES]
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/hungarian.data"))
    args = parser.parse_args()

    X, y = load_data(args.data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    pca_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=2)),
        ]
    )
    X_train_pca = pca_pipe.fit_transform(X_train)
    X_test_pca = pca_pipe.transform(X_test)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        X_train_pca[:, 0],
        X_train_pca[:, 1],
        c=y_train,
        cmap="coolwarm",
        alpha=0.7,
        edgecolors="k",
    )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("Heart disease (train set) — PCA projection")
    fig.colorbar(scatter, ax=ax, label="Disease (0/1)")
    assets = Path("assets")
    assets.mkdir(exist_ok=True)
    fig.savefig(assets / "pca_projection.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=RANDOM_STATE
        ),
    }

    print(f"Loaded {len(X)} patients ({y.sum()} with disease)\n")
    for name, model in models.items():
        model.fit(X_train_pca, y_train)
        preds = model.predict(X_test_pca)
        acc = accuracy_score(y_test, preds)
        print(f"{name} (PCA features, hold-out): accuracy={acc:.3f}")
        print(classification_report(y_test, preds, digits=3))
        print()


if __name__ == "__main__":
    main()
