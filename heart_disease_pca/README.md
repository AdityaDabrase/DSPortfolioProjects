# Heart Disease Prediction (PCA)

**Status:** Needs data

Dimensionality reduction and classification on the UCI Hungarian heart disease dataset — PCA for feature compression followed by multiple classifier comparison.

## Skills

Python · Pandas · scikit-learn · PCA · classification · model comparison · data preprocessing

## Dataset

[UCI Heart Disease (Hungary)](https://archive.ics.uci.edu/dataset/45/heart+disease) — place `hungarian.data` in [`data/`](data/) (see [data/README.md](data/README.md)).

## Quickstart

```bash
pip install -r ../requirements.txt
# Place hungarian.data in data/, then:
jupyter notebook notebook.ipynb
```

## Key findings

- PCA reduces feature dimensionality while preserving most variance.
- Multiple classifiers can be compared on the reduced feature space.
- Feature scaling and encoding are critical before PCA on mixed clinical attributes.
- Model performance varies by algorithm choice on the reduced components.

## Planned project

[`cancer_death_rate_stub.ipynb`](cancer_death_rate_stub.ipynb) is a placeholder for a future cancer death-rate prediction project. Code will be added when available.

## Project structure

| File | Purpose |
| ---- | ------- |
| [`notebook.ipynb`](notebook.ipynb) | PCA + classification analysis |
| [`cancer_death_rate_stub.ipynb`](cancer_death_rate_stub.ipynb) | Planned project — code TBD |
| [`data/`](data/) | Place `hungarian.data` here |
