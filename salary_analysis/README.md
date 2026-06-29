# San Francisco Salaries EDA

How is public-sector compensation distributed across roles and years in San Francisco (2011–2014)?

## Skills

Python · Pandas · groupby · aggregation · salary statistics

## Dataset

See [`data/README.md`](data/README.md) for source and download.

## Key findings

- Base pay and overtime vary widely across departments and roles.
- A small set of job titles account for many employees.
- Year-over-year average base pay shifts from 2011 to 2014; negative totals flag data-quality issues.

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
