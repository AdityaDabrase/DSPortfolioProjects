# Data Science Portfolio — Aditya Dabrase

![banner](assets/banner.png)

Independent data science and data engineering projects. Start with the **featured** work below, then explore EDA and classic ML folders as you like.

[Standalone A/B testing repo (same project, pinned on my profile)](https://github.com/AdityaDabrase/ab-testing-email-marketing) · [LinkedIn](https://www.linkedin.com/in/adityadabrase/)

---

## Featured

| Project | Why it matters |
| --- | --- |
| **[A/B Testing + Experiment Auditor](ab_testing_email_marketing/)** | Real 64k-customer randomized email test, decision memo with dollar impact, and a CLI that audits any A/B CSV for SRM / imbalance / power / peeking |
| **[NA Telecom Data Platform](na-telecom-data-platform/)** | Batch pipeline for CRTC + FCC regulatory telecom data (Airflow, BigQuery/DuckDB, data quality). [Guide](na-telecom-data-platform/docs/project_explained.md) |
| **[Health Insurance Charge Prediction](insurance_prediction/)** | Complete ML package: data, script, notebook, saved metrics/figures |

Also: **[Retail Reporting Automation](https://github.com/AdityaDabrase/retail-reporting-automation)** (separate repo) — CSV → charts, Excel, PowerPoint.

---

## Setup (once)

Shared notebook/ML dependencies:

```bash
pip install -r requirements.txt
```

The telecom platform has its **own** dependencies under `na-telecom-data-platform/` (Airflow, DuckDB, etc.).

Dataset sources and download steps live in each project's `data/README.md`. Optional bulk fetch:

```bash
python scripts/download_data.py
```

---

## All projects

### Experimentation & causal inference

| Project | Link |
| --- | --- |
| A/B Testing: Email Marketing + Experiment Auditor | [ab_testing_email_marketing](ab_testing_email_marketing/) |

### Data engineering

| Project | Link |
| --- | --- |
| North American Telecom Market Intelligence Pipeline | [na-telecom-data-platform](na-telecom-data-platform/) |

### Exploratory data analysis

| Project | Link |
| --- | --- |
| 911 Emergency Calls EDA | [911_calls_analysis](911_calls_analysis/) |
| E-Commerce Purchases Analysis | [ecom_purchases](ecom_purchases/) |
| San Francisco Salaries EDA | [salary_analysis](salary_analysis/) |
| Titanic Data Visualization | [titanic_visualization](titanic_visualization/) |

### Machine learning

| Project | Link |
| --- | --- |
| Health Insurance Charge Prediction | [insurance_prediction](insurance_prediction/) |
| E-Commerce Customer Spend (Linear Regression) | [linear_regression](linear_regression/) |
| Ad Click Prediction (Logistic Regression) | [logistic_regression](logistic_regression/) |
| Customer Segmentation (K-Means) | [k_means_clustering](k_means_clustering/) |
| Heart Disease Prediction (PCA) | [heart_disease_pca](heart_disease_pca/) |

### Reports & data quality

| Project | Link |
| --- | --- |
| Boston Housing Price Analysis | [house_price_predictor](house_price_predictor/) |
| Flight Delay — legacy PDF report (no training code in repo) | [flight_delay_predictor](flight_delay_predictor/) |

---

## Project layout

Folders vary by project type. A typical notebook project looks like:

```
project_name/
├── README.md
├── data/            # often gitignored; see data/README.md
├── notebook.ipynb   # or analysis.py
└── assets/          # figures where present
```

Larger systems (telecom, A/B testing) use their own `src/`, `docs/`, `tests/`, or `code/` layouts — see each project's README.

`insurance_prediction/` is the most complete classic-ML example: bundled data, script, notebook, and saved results.

## License

MIT — see [`LICENSE`](LICENSE). External datasets remain under their original terms (see each `data/README.md`).

## Contact

[LinkedIn](https://www.linkedin.com/in/adityadabrase/) · [dabrase.a@gmail.com](mailto:dabrase.a@gmail.com)
