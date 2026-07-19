# Health Insurance Charge Prediction

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange)

Predicting individual medical insurance charges from demographic and lifestyle factors, and comparing five regression models on the task.


![Overview](assets/overview.jpg)

## Skills

Python · Pandas · scikit-learn · pipelines · GridSearchCV · 5-fold cross-validation · Ridge · Lasso · Random Forest · polynomial regression · feature engineering

## Dataset

Classic [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance) (1,338 rows), bundled as [`data/insurance.csv`](data/insurance.csv).

| Column | Description |
| ------ | ----------- |
| `age` | Age of the primary beneficiary |
| `sex` | Gender (`female`, `male`) |
| `bmi` | Body mass index |
| `children` | Number of dependents |
| `smoker` | Smoking status (`yes`, `no`) |
| `region` | US region |
| `charges` | **Target** — medical costs billed |

## Quickstart

```bash
pip install -r requirements.txt
python analysis.py
# or
jupyter notebook notebook.ipynb
```

The script writes metrics and figures to `results/`.

## Key findings

- **Smoking is the strongest cost driver** — smokers pay roughly $23,000 more on average.
- **BMI and age come next**, amplified for smokers — non-linear models capture this interaction.
- Linear models perform similarly (R² ≈ 0.74); regularization adds little on this small feature set.
- Random Forest (R² = 0.855) and polynomial regression (R² = 0.835) cut error by roughly a third.

| Model | R² (CV mean) | MAE ($) | RMSE ($) |
| --- | --- | --- | --- |
| **Random Forest** | **0.855** | **2,536** | **4,547** |
| Polynomial Regression (deg=2) | 0.835 | 2,913 | 4,834 |
| Linear Regression | 0.740 | 4,203 | 6,077 |

![Feature importances](results/feature_importances.png)

## Project structure

| File | Purpose |
| ---- | ------- |
| [`notebook.ipynb`](notebook.ipynb) | Guided walkthrough with EDA and model explanations |
| [`analysis.py`](analysis.py) | Reproducible end-to-end script |
| [`data/insurance.csv`](data/insurance.csv) | Dataset |
| [`results/`](results/) | Metrics table and figures |
| [`requirements.txt`](requirements.txt) | Project dependencies |

The notebook and script run the **same analysis** in two formats.
