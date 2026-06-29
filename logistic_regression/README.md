# Ad Click Prediction (Logistic Regression)

Which user traits predict ad clicks so marketing can target high-intent audiences?

## Skills

Python · Pandas · scikit-learn · logistic regression · classification report · Seaborn EDA

## Dataset

See [`data/README.md`](data/README.md) for source and download.

## Key findings

- Age and daily time on site differ between clickers and non-clickers.
- Logistic regression achieves strong separation on the hold-out set.
- Pairplots with hue on "Clicked on Ad" highlight the most predictive features.

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
| [`notebook.ipynb`](notebook.ipynb) | EDA + logistic regression |
| [`analysis.py`](analysis.py) | Same analysis as a script |
| [`data/`](data/) | Dataset |
