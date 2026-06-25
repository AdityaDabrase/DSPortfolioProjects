# Heart Disease Prediction (PCA)

Dimensionality reduction and classification on the UCI Hungarian heart disease dataset — PCA for feature compression followed by multiple classifier comparison.

## Skills

Python · Pandas · scikit-learn · PCA · classification · model comparison · data preprocessing

## Dataset

[UCI Heart Disease (Hungary)](https://archive.ics.uci.edu/dataset/45/heart+disease) — bundled as [`data/hungarian.data`](data/hungarian.data) (see [data/README.md](data/README.md)).

## Quickstart

```bash
pip install -r ../requirements.txt
jupyter notebook notebook.ipynb
```

## Key findings

- PCA reduces feature dimensionality while preserving most variance.
- Multiple classifiers can be compared on the reduced feature space.
- Feature scaling and encoding are critical before PCA on mixed clinical attributes.
- Model performance varies by algorithm choice on the reduced components.

## Project structure

| File | Purpose |
| ---- | ------- |
| [`notebook.ipynb`](notebook.ipynb) | PCA + classification analysis |
| [`data/hungarian.data`](data/hungarian.data) | UCI Hungary heart disease subset |
| [`assets/`](assets/) | Visualizations |
