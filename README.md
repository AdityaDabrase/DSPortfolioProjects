# Data Science Portfolio — Aditya Dabrase

![banner](assets/banner.png)

A collection of data science projects spanning exploratory analysis, regression, classification, clustering, and dimensionality reduction. Each project lives in its own folder with a consistent layout: `data/`, `notebook.ipynb`, optional `analysis.py`, and `assets/`.

## Skills demonstrated

Python · Pandas · NumPy · Matplotlib · Seaborn · scikit-learn · Jupyter · EDA · Regression · Classification · Clustering · PCA · Cross-validation · Pipelines · GridSearchCV

## Projects by category

### Exploratory Data Analysis

| Project | Skills | Link |
|---------|--------|------|
| 911 Emergency Calls EDA | time-series patterns, geographic viz, count plots | [911_calls_analysis](911_calls_analysis/) |
| E-Commerce Purchases Analysis | Pandas filtering, groupby, Q&A style EDA | [ecom_purchases](ecom_purchases/) |
| San Francisco Salaries EDA | salary statistics, job-title analysis | [salary_analysis](salary_analysis/) |
| Titanic Data Visualization | Seaborn categorical plots, FacetGrid | [titanic_visualization](titanic_visualization/) |

### Machine Learning

| Project | Models / methods | Link |
|---------|------------------|------|
| Health Insurance Charge Prediction | Linear, Ridge, Lasso, Random Forest, Polynomial | [insurance_prediction](insurance_prediction/) |
| E-Commerce Customer Spend (Linear Regression) | OLS, train/test split, residual analysis | [linear_regression](linear_regression/) |
| Ad Click Prediction (Logistic Regression) | binary classification, confusion matrix | [logistic_regression](logistic_regression/) |
| Customer Segmentation (K-Means) | elbow method, StandardScaler, PCA viz | [k_means_clustering](k_means_clustering/) |
| Heart Disease Prediction (PCA) | PCA, multiple classifiers | [heart_disease_pca](heart_disease_pca/) |

### Reports

| Project | Focus | Link |
|---------|-------|------|
| Flight Delay Classification | Naïve Bayes, CART, Logistic Regression | [flight_delay_predictor](flight_delay_predictor/) |
| Boston Housing Price Analysis | missing data, outlier detection, imputation | [house_price_predictor](house_price_predictor/) |

## Quickstart

```bash
git clone https://github.com/AdityaDabrase/DSPortfolioProjects.git
cd DSPortfolioProjects
pip install -r requirements.txt
```

Most projects include the dataset in `data/` or download instructions in `data/README.md`. Start with [insurance_prediction](insurance_prediction/) for the most complete end-to-end example.

## Flagship project

[insurance_prediction](insurance_prediction/) bundles the dataset, a reproducible script, model comparison with cross-validation, and saved results. Use it as the reference layout for other projects.
