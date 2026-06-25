# Titanic Data Visualization

Seaborn visualization exercises on the classic Titanic passenger dataset — survival patterns, fare distributions, and categorical relationships.

## Skills

Python · Seaborn · Matplotlib · FacetGrid · count plots · distribution plots · categorical visualization

## Dataset

Built-in via `sns.load_dataset('titanic')` — no local file required.

## Quickstart

```bash
pip install -r ../requirements.txt
jupyter notebook notebook.ipynb
# or
python analysis.py
```

## Key findings

- Survival rates differ sharply by passenger class and gender.
- Fare distributions are right-skewed with high outliers in first class.
- Age distributions vary by embark port and survival status.
- Pairplots and FacetGrids reveal class-gender interaction effects.

## Project structure

| File | Purpose |
| ---- | ------- |
| [`notebook.ipynb`](notebook.ipynb) | Seaborn exercise walkthrough |
| [`analysis.py`](analysis.py) | Same visualizations as a script |
