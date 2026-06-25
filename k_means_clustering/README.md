# Customer Segmentation (K-Means)

Unsupervised customer segmentation using K-Means clustering on mall customer spending and income attributes, with elbow-method tuning and PCA visualization.

## Skills

Python · scikit-learn · K-Means · StandardScaler · elbow method · PCA · unsupervised learning · Matplotlib · Seaborn

## Dataset

[Mall Customer Segmentation](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) — bundled as [`data/Mall_Customers.csv`](data/Mall_Customers.csv) (see [data/README.md](data/README.md)).

Features include annual income, spending score, age, and gender.

## Quickstart

```bash
pip install -r ../requirements.txt
jupyter notebook notebook.ipynb
# or
python analysis.py
```

## Key findings

- The elbow method suggests an optimal cluster count balancing WCSS and interpretability.
- Distinct segments emerge by income vs. spending score (e.g. high income / low spend vs. high spenders).
- PCA projection makes cluster separation visually clear in 2D.
- Segments map to actionable marketing personas (budget-conscious, target buyers, etc.).

![K-Means preview](assets/kmeans_preview.jpg)

## Project structure

| File | Purpose |
| ---- | ------- |
| [`notebook.ipynb`](notebook.ipynb) | Clustering walkthrough with visualizations |
| [`analysis.py`](analysis.py) | Same analysis as a script |
| [`data/Mall_Customers.csv`](data/Mall_Customers.csv) | Dataset |
| [`assets/`](assets/) | Preview images |
