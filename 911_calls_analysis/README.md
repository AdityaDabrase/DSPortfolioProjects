# 911 Emergency Calls EDA

When and where do EMS, fire, and traffic emergencies peak in Montgomery County — and how might dispatch resources be scheduled?

## Skills

Python · Pandas · Matplotlib · Seaborn · time-series features · heatmaps

## Dataset

Montgomery County 911 calls — see [`data/README.md`](data/README.md) for source and download.

## Key findings

- EMS dominates most time windows; fire and traffic show distinct hourly patterns.
- Heatmaps reveal weekday × hour hotspots by call type.
- A handful of zip codes and townships account for much of the volume.

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
| [`notebook.ipynb`](notebook.ipynb) | EDA walkthrough |
| [`analysis.py`](analysis.py) | Same analysis as a script |
| [`data/`](data/) | Dataset |
