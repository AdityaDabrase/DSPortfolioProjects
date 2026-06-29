# Data Science Portfolio — Aditya Dabrase

![banner](assets/banner.png)

Independent data science projects, each in its own folder with its own data, notebook, and documentation. There is no shared application or pipeline — open any project below on its own.

## Setup (once)

Shared Python dependencies for all projects:

```bash
pip install -r requirements.txt
```

Dataset sources and download steps are documented in each project's `data/README.md`. To fetch every dataset in one step (optional):

```bash
python scripts/download_data.py
```

## Projects

### Exploratory Data Analysis

| Project | Link |
|---------|------|
| 911 Emergency Calls EDA | [911_calls_analysis](911_calls_analysis/) |
| E-Commerce Purchases Analysis | [ecom_purchases](ecom_purchases/) |
| San Francisco Salaries EDA | [salary_analysis](salary_analysis/) |
| Titanic Data Visualization | [titanic_visualization](titanic_visualization/) |

### Machine Learning

| Project | Link |
|---------|------|
| Health Insurance Charge Prediction | [insurance_prediction](insurance_prediction/) |
| E-Commerce Customer Spend (Linear Regression) | [linear_regression](linear_regression/) |
| Ad Click Prediction (Logistic Regression) | [logistic_regression](logistic_regression/) |
| Customer Segmentation (K-Means) | [k_means_clustering](k_means_clustering/) |
| Heart Disease Prediction (PCA) | [heart_disease_pca](heart_disease_pca/) |

### Reports & data quality

| Project | Link |
|---------|------|
| Boston Housing Price Analysis | [house_price_predictor](house_price_predictor/) |
| Flight Delay Classification | [flight_delay_predictor](flight_delay_predictor/) |

## Project layout

Each folder follows the same shape:

```
project_name/
├── README.md
├── data/
├── notebook.ipynb
├── analysis.py      # where applicable
└── assets/
```

[insurance_prediction](insurance_prediction/) is the most complete example: bundled data, script, notebook, and saved results.
