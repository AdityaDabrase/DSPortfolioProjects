"""Health insurance charge prediction.

Trains and compares five regression models (Linear, Ridge, Lasso, Random
Forest, Polynomial) on the classic insurance dataset using a shared
preprocessing pipeline and 5-fold cross-validation. Saves a metrics table
and figures to the output directory.

This script is the same analysis as Insurance.ipynb in standalone form:
the notebook walks through the EDA and explains each step, while this file
is the reproducible end-to-end run.

Usage:
    python InsurancePrediction.py [--data insurance.csv] [--output-dir results]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

# Use a non-interactive backend: the script saves figures to disk and must
# also work on machines/CI without a display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

NUMERIC_FEATURES = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]
TARGET = "charges"

# Fixing the random seed makes every run reproducible: the same CV folds,
# the same train/test split, and the same forest are built every time.
RANDOM_STATE = 42


def load_data(path: Path) -> pd.DataFrame:
    """Read the CSV and fail early with a clear message if columns are missing."""
    df = pd.read_csv(path)
    expected = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    return df


def make_preprocessor() -> ColumnTransformer:
    """Shared preprocessing applied inside every model's pipeline.

    - StandardScaler rescales the numeric features (age, bmi, children) to
      mean 0 / std 1. Ridge and Lasso penalize the *size* of coefficients,
      so features must be on a comparable scale or the penalty would
      unfairly punish features that happen to have large units.
    - OneHotEncoder turns each category into its own 0/1 column
      (e.g. smoker -> smoker_yes). One-hot encoding is used instead of
      label encoding because labels like northeast=0 < southeast=2 would
      impose an ordering that has no real meaning.
      drop="first" removes one column per feature (keeps smoker_yes, drops
      smoker_no) to avoid perfect multicollinearity — the "dummy variable
      trap" — which destabilizes linear models.

    Because the preprocessor lives *inside* each pipeline, it is re-fit on
    the training portion of every cross-validation fold. Statistics like
    the scaler's mean are therefore never computed on test data (no leakage).
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first"), CATEGORICAL_FEATURES),
        ]
    )


def build_models() -> dict[str, Pipeline]:
    """Create the five competing pipelines, each with its own preprocessor.

    The models, from simplest to most flexible:

    - Linear Regression: the baseline. Assumes charges are a weighted sum
      of the features. Fast and fully interpretable, but it cannot capture
      interactions (e.g. smoking amplifying the effect of BMI).
    - Ridge Regression: linear regression plus an L2 penalty that shrinks
      coefficients toward zero. Helps when features are correlated or the
      model overfits. The penalty strength `alpha` is a hyperparameter.
    - Lasso Regression: linear regression plus an L1 penalty. Unlike Ridge,
      Lasso can shrink coefficients to *exactly* zero, performing implicit
      feature selection.
    - Random Forest: an ensemble of decision trees, each trained on a
      bootstrap sample with random feature subsets, then averaged. Captures
      non-linearities and feature interactions automatically and is robust
      to outliers, at the cost of interpretability.
    - Polynomial Regression: a linear model fit on degree-2 features, i.e.
      squares (age^2) and pairwise interactions (bmi x smoker_yes). This
      hand-crafts exactly the non-linearity the EDA revealed while staying
      a linear (interpretable) model under the hood.

    Where a model has an important hyperparameter, it is wrapped in
    GridSearchCV: for each candidate value, an inner 5-fold CV is run on
    the training data only, and the best value is kept. Nesting the search
    inside the pipeline means hyperparameters are chosen without ever
    seeing the outer test fold.
    """
    models: dict[str, Pipeline] = {
        "Linear Regression": Pipeline(
            [("prep", make_preprocessor()), ("model", LinearRegression())]
        ),
        "Ridge Regression": Pipeline(
            [
                ("prep", make_preprocessor()),
                (
                    "model",
                    # Try alphas spanning four orders of magnitude; pick the
                    # one with the lowest inner-CV RMSE.
                    GridSearchCV(
                        Ridge(),
                        param_grid={"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
                        cv=5,
                        scoring="neg_root_mean_squared_error",
                    ),
                ),
            ]
        ),
        "Lasso Regression": Pipeline(
            [
                ("prep", make_preprocessor()),
                (
                    "model",
                    # Lasso's coordinate-descent solver can need many
                    # iterations to converge, hence max_iter=10_000.
                    GridSearchCV(
                        Lasso(max_iter=10_000),
                        param_grid={"alpha": [0.1, 1.0, 10.0, 100.0, 500.0]},
                        cv=5,
                        scoring="neg_root_mean_squared_error",
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("prep", make_preprocessor()),
                (
                    "model",
                    # max_depth and min_samples_leaf control tree complexity:
                    # shallower trees / bigger leaves = less overfitting.
                    # An unconstrained forest memorizes this small dataset
                    # (train R2 ~0.97), so regularizing matters here.
                    GridSearchCV(
                        RandomForestRegressor(random_state=RANDOM_STATE),
                        param_grid={
                            "n_estimators": [200],
                            "max_depth": [4, 6, 8, None],
                            "min_samples_leaf": [1, 5, 10],
                        },
                        cv=5,
                        scoring="neg_root_mean_squared_error",
                    ),
                ),
            ]
        ),
        "Polynomial Regression (deg=2)": Pipeline(
            [
                ("prep", make_preprocessor()),
                # include_bias=False because LinearRegression already fits
                # an intercept; a constant column would be redundant.
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("model", LinearRegression()),
            ]
        ),
    }
    return models


def evaluate_models(
    models: dict[str, Pipeline], X: pd.DataFrame, y: pd.Series
) -> pd.DataFrame:
    """Cross-validate every model on identical folds and return a metrics table.

    5-fold cross-validation: the data is split into 5 parts; each model is
    trained on 4 and scored on the held-out fifth, rotating until every part
    has served as the test set. The reported score is the mean of the 5
    runs, which is far less sensitive to a "lucky split" than a single
    train/test split. Using the same KFold object (same seed) for every
    model guarantees they are all judged on exactly the same folds.

    Metrics:
    - R2: fraction of the variance in charges the model explains (1.0 = perfect).
    - MAE: mean absolute error in dollars; easy to interpret, robust to outliers.
    - RMSE: root mean squared error in dollars; penalizes large misses harder.
    """
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    # scikit-learn always *maximizes* scores, so error metrics are exposed
    # as negated versions ("neg_..."); the sign is flipped back below.
    scoring = {
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
    }
    rows = []
    for name, pipeline in models.items():
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring)
        rows.append(
            {
                "Model": name,
                "R2 (CV mean)": scores["test_r2"].mean(),
                "R2 (CV std)": scores["test_r2"].std(),
                "MAE": -scores["test_mae"].mean(),
                "RMSE": -scores["test_rmse"].mean(),
            }
        )
        print(f"{name:32s} R2={rows[-1]['R2 (CV mean)']:.3f} "
              f"MAE={rows[-1]['MAE']:,.0f} RMSE={rows[-1]['RMSE']:,.0f}")
    return (
        pd.DataFrame(rows)
        .sort_values("RMSE")
        .reset_index(drop=True)
        .round({"R2 (CV mean)": 3, "R2 (CV std)": 3, "MAE": 0, "RMSE": 0})
    )


def plot_feature_importances(
    models: dict[str, Pipeline], X: pd.DataFrame, y: pd.Series, output_dir: Path
) -> None:
    """Fit the Random Forest on all data and plot which features matter most.

    A tree's importance for a feature is how much that feature's splits
    reduce prediction error, averaged over all trees in the forest. The
    values sum to 1, so they read as relative shares of predictive power.
    """
    pipeline = models["Random Forest"]
    pipeline.fit(X, y)
    search: GridSearchCV = pipeline.named_steps["model"]
    forest: RandomForestRegressor = search.best_estimator_
    # get_feature_names_out maps importances back to readable column names
    # (e.g. "cat__smoker_yes") after one-hot encoding.
    feature_names = pipeline.named_steps["prep"].get_feature_names_out()
    importances = (
        pd.Series(forest.feature_importances_, index=feature_names)
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    importances.plot.barh(ax=ax, color="steelblue", edgecolor="black")
    ax.set_title("Random Forest feature importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importances.png", dpi=150)
    plt.close(fig)
    print(f"Best RF params: {search.best_params_}")


def plot_predictions(
    models: dict[str, Pipeline], X: pd.DataFrame, y: pd.Series, output_dir: Path
) -> None:
    """Actual-vs-predicted scatter for the polynomial model on a hold-out set.

    Cross-validation gives averaged scores but no single set of predictions
    to draw, so this diagnostic uses a conventional 80/20 split: train on
    80%, predict the unseen 20%, and plot predictions against reality.
    Points on the dashed diagonal are perfect predictions; vertical distance
    from it is the error in dollars.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    pipeline = models["Polynomial Regression (deg=2)"]
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred, alpha=0.5, color="seagreen", edgecolor="black")
    lims = [0, max(y_test.max(), y_pred.max()) * 1.05]
    ax.plot(lims, lims, "r--", lw=2, label="Perfect prediction")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual charges ($)")
    ax.set_ylabel("Predicted charges ($)")
    ax.set_title("Polynomial Regression: actual vs predicted (hold-out set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)

    print(
        "Hold-out (Polynomial): "
        f"R2={r2_score(y_test, y_pred):.3f} "
        f"MAE={mean_absolute_error(y_test, y_pred):,.0f} "
        f"RMSE={np.sqrt(mean_squared_error(y_test, y_pred)):,.0f}"
    )


def plot_eda(df: pd.DataFrame, output_dir: Path) -> None:
    """Save the two headline EDA figures (see the notebook for the full EDA).

    - The distribution of charges is strongly right-skewed: most people cost
      a few thousand dollars, a long tail costs tens of thousands.
    - The smoker boxplot shows that tail is almost entirely smokers — the
      single biggest signal in the dataset.
    """
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df[TARGET], kde=True, color="steelblue", ax=ax)
    ax.set_title("Distribution of insurance charges")
    fig.tight_layout()
    fig.savefig(output_dir / "charges_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x="smoker", y=TARGET, hue="smoker", palette="Set2", ax=ax)
    ax.set_title("Charges by smoking status")
    fig.tight_layout()
    fig.savefig(output_dir / "charges_by_smoker.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("insurance.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    print(f"Loaded {len(df)} rows from {args.data}\n")
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    plot_eda(df, args.output_dir)

    # Compare all five models on identical CV folds, then persist the table.
    models = build_models()
    results = evaluate_models(models, X, y)

    results.to_csv(args.output_dir / "model_metrics.csv", index=False)
    print(f"\nMetrics table:\n{results.to_string(index=False)}")

    plot_feature_importances(models, X, y, args.output_dir)
    plot_predictions(models, X, y, args.output_dir)
    print(f"\nFigures and metrics saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
