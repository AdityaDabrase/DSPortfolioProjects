# Heart Disease Prediction (PCA)

Can clinical heart-disease screening be simplified by compressing many measurements into fewer dimensions while keeping predictive accuracy?

## Skills

Python · Pandas · scikit-learn · PCA · imputation · classification

## Dataset

UCI Hungarian heart disease subset — see [`data/README.md`](data/README.md) for source and download.

## Key findings

- Median imputation retains 294 patients despite extensive missing values in several fields.
- Two PCA components preserve separable structure between disease and no-disease groups.
- Random Forest and logistic regression both reach ~84–85% hold-out accuracy on PCA features.

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
| [`notebook.ipynb`](notebook.ipynb) | EDA, PCA, and classification |
| [`analysis.py`](analysis.py) | Reproducible end-to-end script |
| [`data/`](data/) | Dataset |
| [`assets/`](assets/) | Figures |
