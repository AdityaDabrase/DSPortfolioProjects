# 911 Emergency Calls EDA

Exploring Montgomery County 911 emergency call data to uncover temporal patterns, geographic hotspots, and call-type breakdowns (EMS, Traffic, Fire).

## Skills

Python · Pandas · NumPy · Matplotlib · Seaborn · EDA · time-series analysis · geographic visualization

## Dataset

[Montgomery County 911 Calls](https://www.kaggle.com/mchirico/montcoalert) — download `911.csv` into [`data/`](data/) (see [data/README.md](data/README.md)).

663,522 records with latitude, longitude, zip code, township, timestamp, and call description.

## Quickstart

```bash
pip install -r ../requirements.txt
# Place 911.csv in data/, then:
jupyter notebook notebook.ipynb
# or
python analysis.py
```

## Key findings

- Peak call volumes cluster at specific hours and days of the week.
- Certain zip codes and townships dominate total call volume.
- EMS calls outweigh Traffic and Fire across most time windows.
- Heatmaps and cluster maps reveal geographic concentration of emergencies.

## Project structure

| File | Purpose |
| ---- | ------- |
| [`notebook.ipynb`](notebook.ipynb) | Guided EDA walkthrough with visualizations |
| [`analysis.py`](analysis.py) | Same analysis as a script |
| [`data/`](data/) | Place `911.csv` here |
| [`assets/`](assets/) | Screenshots and GIFs |
