# E-Commerce Purchases Analysis

Who buys what, when, and through which channels on a synthetic e-commerce platform?

## Skills

Python · Pandas · filtering · groupby · value counts

## Dataset

See [`data/README.md`](data/README.md) for source and download.

## Key findings

- Purchase prices span a wide range; AM vs. PM patterns differ.
- Job-title and email-provider distributions show demographic skew.
- Credit-card filters surface distinct customer micro-segments.

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
| [`notebook.ipynb`](notebook.ipynb) | Pandas exercises |
| [`analysis.py`](analysis.py) | Same analysis as a script |
| [`data/`](data/) | Dataset |
