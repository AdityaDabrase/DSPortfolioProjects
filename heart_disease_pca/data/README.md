# Dataset

**File:** `hungarian.data`

Hungary subset of the [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease).

This repo uses the **processed** UCI file (comma-separated, `?` for missing values) saved as `hungarian.data` for compatibility with the notebook.

**Expected columns:** `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`, `num`

**Re-download (if needed):**

```bash
curl -L -o hungarian.data \
  "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data"
```
