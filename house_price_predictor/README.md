# Boston Housing Price Analysis

**Status:** Report only

Handling missing data and outlier detection in the Boston Housing dataset — imputation, omission techniques, and PTRATIO outlier classification. Notebook and script code will be added in a future update.

## Skills

Python · Pandas · missing-data handling · outlier detection · imputation · omission · data quality

## Dataset

Boston Housing data from the [StatLib archive](http://lib.stat.cmu.edu/datasets/boston) — bundled as [`data/BostonHousing.xls`](data/BostonHousing.xls) (167 cases, 11 attributes).

## Reports

- [Boston Housing Project PDF](assets/boston_housing_project.pdf) — full analysis write-up

## Key findings

- Missing values appear across predictors (excluding PTRATIO in the assignment scope).
- PTRATIO contains outliers classifiable as typing errors, data-entry errors, or genuine extremes.
- NaN substitution enables consistent missing-data workflows in Pandas.
- Omission vs. imputation (mean, median, ML-based) trade bias for completeness.

![Overview](assets/overview.jpg)

## Quickstart

```bash
pip install -r ../requirements.txt
# Review assets/boston_housing_project.pdf
# Code coming soon
```

## Project structure

| File | Purpose |
| ---- | ------- |
| [`data/BostonHousing.xls`](data/BostonHousing.xls) | Dataset |
| [`assets/boston_housing_project.pdf`](assets/boston_housing_project.pdf) | Analysis report |
| [`assets/overview.jpg`](assets/overview.jpg) | Preview image |
