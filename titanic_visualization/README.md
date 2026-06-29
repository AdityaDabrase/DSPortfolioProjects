# Titanic Data Visualization

How did passenger class, fare, and demographics relate to survival on the Titanic?

## Skills

Python · Seaborn · Matplotlib · FacetGrid · categorical visualization

## Dataset

Built-in via `sns.load_dataset('titanic')` — no local file required.

## Key findings

- Survival rates differ sharply by class and gender.
- Fare distributions are right-skewed with high outliers in first class.
- FacetGrids reveal class–gender interaction effects on age.

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
| [`notebook.ipynb`](notebook.ipynb) | Seaborn exercises |
| [`analysis.py`](analysis.py) | Same visualizations as a script |
| [`assets/`](assets/) | Saved figures |
