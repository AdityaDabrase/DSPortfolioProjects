# Ad Click Prediction (Logistic Regression)

Binary classification to predict whether an internet user clicks an advertisement based on demographics, income, and browsing behavior.

## Skills

Python · Pandas · scikit-learn · logistic regression · classification report · Seaborn EDA · train/test split

## Dataset

Ad-click dataset (`advertising.csv`) — bundled in [`data/`](data/) or see [data/README.md](data/README.md) to re-download.

## Quickstart

```bash
pip install -r ../requirements.txt
jupyter notebook notebook.ipynb
# or
python analysis.py
```

## Key findings

- Age and daily time on site show distinct distributions between clickers and non-clickers.
- Area income and daily internet usage interact with ad-click behavior.
- Logistic regression achieves strong separation on the test set.
- Pairplots with hue on "Clicked on Ad" highlight the most predictive features.

## Project structure

| File | Purpose |
| ---- | ------- |
| [`notebook.ipynb`](notebook.ipynb) | EDA + logistic regression walkthrough |
| [`analysis.py`](analysis.py) | Same analysis as a script |
| [`data/advertising.csv`](data/advertising.csv) | Dataset |
| [`assets/`](assets/) | Visualizations |
