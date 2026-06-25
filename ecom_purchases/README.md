# E-Commerce Purchases Analysis

**Status:** Needs data

Pandas Q&A exercise on a synthetic e-commerce purchases dataset — exploring prices, job titles, purchase timing, and payment details.

## Skills

Python · Pandas · filtering · groupby · value counts · Q&A style EDA

## Dataset

**Ecommerce Purchases** CSV — place in [`data/Ecommerce Purchases.csv`](data/) (see [data/README.md](data/README.md)).

## Quickstart

```bash
pip install -r ../requirements.txt
# Place dataset in data/, then:
jupyter notebook notebook.ipynb
# or
python analysis.py
```

## Key findings

- Average and extreme purchase prices reveal the transaction spread.
- AM vs PM purchase patterns differ across the customer base.
- Job-title and email-provider distributions highlight demographic skew.
- Credit-card provider and expiry filters surface targeted customer segments.

## Project structure

| File | Purpose |
| ---- | ------- |
| [`notebook.ipynb`](notebook.ipynb) | Step-by-step Pandas exercises |
| [`analysis.py`](analysis.py) | Same analysis as a script |
| [`data/`](data/) | Place dataset here |
