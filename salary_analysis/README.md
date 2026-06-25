# San Francisco Salaries EDA

**Status:** Needs data

Exploring San Francisco city employee salary records — base pay, overtime, benefits, and job-title patterns from 2011–2014.

## Skills

Python · Pandas · aggregation · groupby · sorting · salary statistics

## Dataset

[SF Salaries](https://www.kaggle.com/datasets/kaggle/sf-salaries) — download `Salaries.csv` into [`data/`](data/) (see [data/README.md](data/README.md)).

## Quickstart

```bash
pip install -r ../requirements.txt
# Place Salaries.csv in data/, then:
jupyter notebook notebook.ipynb
# or
python analysis.py
```

## Key findings

- Base pay and overtime vary widely across departments and roles.
- A small set of job titles account for the majority of employees.
- Year-over-year average base pay shifts from 2011 to 2014.
- "Chief" titles and single-person roles highlight organizational structure.

## Project structure

| File | Purpose |
| ---- | ------- |
| [`notebook.ipynb`](notebook.ipynb) | Pandas Q&A exercises |
| [`analysis.py`](analysis.py) | Same analysis as a script |
| [`data/`](data/) | Place `Salaries.csv` here |
