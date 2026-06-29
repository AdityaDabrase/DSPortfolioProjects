# Boston Housing Price Analysis

How do missing values and outliers affect housing data quality before modeling?

Missing-data handling and outlier detection on a modified Boston Housing dataset. The full narrative is in the PDF report; `analysis.py` reproduces the data-quality visuals.

> The classic Boston Housing dataset was retired from scikit-learn over ethical concerns (socioeconomic proxies). This project uses a modified teaching dataset for imputation and outlier exercises.

## Skills

Python · Pandas · missing-data handling · outlier detection · imputation

## Dataset

Bundled as [`data/BostonHousing.xls`](data/BostonHousing.xls) — see [`data/README.md`](data/README.md).

## Key findings

- Missing values appear in predictors such as `INDUS`, `NOX`, and `DIS`.
- PTRATIO shows outliers suitable for typing-error vs. genuine-extreme classification.
- Imputation vs. row deletion trades completeness for potential bias.

## Run

From this folder:

```bash
pip install -r ../requirements.txt
python analysis.py
open assets/boston_housing_project.pdf
```

## Files

| File | Purpose |
| ---- | ------- |
| [`data/BostonHousing.xls`](data/BostonHousing.xls) | Dataset |
| [`analysis.py`](analysis.py) | Missing-value and outlier figures |
| [`assets/boston_housing_project.pdf`](assets/boston_housing_project.pdf) | Full write-up |
