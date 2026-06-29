# Flight Delay Classification

What drives flight delays for D.C.-area flights to New York — and which models best predict them?

Analysis of commercial flight delays (January 2004), documented in PDF reports. Training data is not included in this repo.

## Skills

Naïve Bayes · CART · Logistic Regression · feature reduction · pivot-table EDA

## Reports

- [Flight Delay EDA](assets/flight_delay_eda.pdf) — preprocessing and exploration
- [Flight Delay Predictions](assets/flight_delay_predictions.pdf) — model building and testing

A delay is defined as arrival ≥ 15 minutes later than scheduled.

## Key findings

- Domain-knowledge feature reduction simplified the predictor set before modeling.
- Naïve Bayes, CART, and logistic regression serve different output types.
- Pivot-table summaries reveal delay patterns across airports and scheduled times.

## Run

```bash
open assets/flight_delay_eda.pdf
open assets/flight_delay_predictions.pdf
```

## Files

| File | Purpose |
| ---- | ------- |
| [`assets/flight_delay_eda.pdf`](assets/flight_delay_eda.pdf) | EDA report |
| [`assets/flight_delay_predictions.pdf`](assets/flight_delay_predictions.pdf) | Modeling report |
| [`data/README.md`](data/README.md) | Dataset notes |
