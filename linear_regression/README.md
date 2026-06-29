# E-Commerce Customer Spend (Linear Regression)

Which customer engagement signals best predict annual spend — app, website, or membership tenure?

## Skills

Python · Pandas · scikit-learn · linear regression · train/test split · residual analysis

## Dataset

See [`data/README.md`](data/README.md) for source and download.

## Key findings

- Length of membership is the strongest linear predictor of yearly spend.
- Time on App correlates more with spend than Time on Website.
- Residuals are approximately normal, supporting the linear model assumption.

![Pairplot](assets/pairplot.png)

## Run

From this folder:

```bash
pip install -r ../requirements.txt
jupyter notebook notebook.ipynb
# or
python analysis.py
```

## Files

| File | Purpose |
| ---- | ------- |
| [`notebook.ipynb`](notebook.ipynb) | EDA + linear regression |
| [`analysis.py`](analysis.py) | Same analysis as a script |
| [`data/`](data/) | Dataset |
| [`assets/`](assets/) | Visualizations |
