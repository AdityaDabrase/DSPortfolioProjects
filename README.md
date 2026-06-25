# Data Science Portfolio — Aditya Dabrase

![banner](assets/banner.png)

A collection of data science projects spanning exploratory analysis, regression, classification, clustering, and dimensionality reduction. Each project lives in its own folder with a consistent layout: `data/`, `notebook.ipynb`, optional `analysis.py`, and `assets/`.

## Skills demonstrated

Python · Pandas · NumPy · Matplotlib · Seaborn · scikit-learn · Jupyter · EDA · Regression · Classification · Clustering · PCA · Cross-validation · Pipelines · GridSearchCV

## Projects by category

### Exploratory Data Analysis

| Project | Skills | Status | Link |
|---------|--------|--------|------|
| 911 Emergency Calls EDA | time-series patterns, geographic viz, count plots | Needs data | [911_calls_analysis](911_calls_analysis/) |
| E-Commerce Purchases Analysis | Pandas filtering, groupby, Q&A style EDA | Needs data | [ecom_purchases](ecom_purchases/) |
| San Francisco Salaries EDA | salary statistics, job-title analysis | Needs data | [salary_analysis](salary_analysis/) |
| Titanic Data Visualization | Seaborn categorical plots, FacetGrid | Runnable | [titanic_visualization](titanic_visualization/) |

### Machine Learning

| Project | Models / methods | Status | Link |
|---------|------------------|--------|------|
| Health Insurance Charge Prediction | Linear, Ridge, Lasso, Random Forest, Polynomial | Runnable | [insurance_prediction](insurance_prediction/) |
| E-Commerce Customer Spend (Linear Regression) | OLS, train/test split, residual analysis | Needs data | [linear_regression](linear_regression/) |
| Ad Click Prediction (Logistic Regression) | binary classification, confusion matrix | Needs data | [logistic_regression](logistic_regression/) |
| Customer Segmentation (K-Means) | elbow method, StandardScaler, PCA viz | Needs data | [k_means_clustering](k_means_clustering/) |
| Heart Disease Prediction (PCA) | PCA, multiple classifiers | Needs data | [heart_disease_pca](heart_disease_pca/) |

### Reports & work in progress

| Project | Focus | Status | Link |
|---------|-------|--------|------|
| Flight Delay Classification | Naïve Bayes, CART, Logistic Regression | Report only | [flight_delay_predictor](flight_delay_predictor/) |
| Boston Housing Price Analysis | missing data, outlier detection, imputation | Report only | [house_price_predictor](house_price_predictor/) |

## Quickstart

```bash
git clone https://github.com/AdityaDabrase/DSPortfolioProjects.git
cd DSPortfolioProjects
pip install -r requirements.txt
```

**Runnable:** [insurance_prediction](insurance_prediction/) and [titanic_visualization](titanic_visualization/). Other projects include dataset download instructions in their `data/README.md` files.

## Flagship project

[insurance_prediction](insurance_prediction/) is the most complete project — bundled dataset, reproducible script, model comparison with cross-validation, and saved results. Use it as the template for the rest of the portfolio.
