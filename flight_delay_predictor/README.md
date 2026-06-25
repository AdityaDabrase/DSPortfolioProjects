# Flight Delay Classification

Predicting flight delays for commercial flights from the Washington, D.C. area to New York in January 2004 — documented methodology with PDF reports.

## Skills

Python · Naïve Bayes · CART · Logistic Regression · data reduction · pivot-table EDA · classification

## Dataset

Training and test CSVs referenced in the reports are not included in the repo. See [data/README.md](data/README.md).

## Reports

- [Flight Delay EDA](assets/flight_delay_eda.pdf) — Phase A: data preprocessing, exploration, and conversion
- [Flight Delay Predictions](assets/flight_delay_predictions.pdf) — Phase B & C: model building and testing

## Key findings

- Domain-knowledge feature reduction simplifies the predictor set before modeling.
- Naïve Bayes, CART, and Logistic Regression serve different output types (categorical vs. numerical).
- Pivot-table summaries reveal delay patterns across airports and scheduled times.
- A delay is defined as arrival ≥ 15 minutes later than scheduled.

![Overview](assets/overview.jpg)

## Quickstart

```bash
# Review the PDF reports in assets/
open assets/flight_delay_eda.pdf
open assets/flight_delay_predictions.pdf
```

## Project structure

| File | Purpose |
| ---- | ------- |
| [`assets/flight_delay_eda.pdf`](assets/flight_delay_eda.pdf) | EDA and preprocessing report |
| [`assets/flight_delay_predictions.pdf`](assets/flight_delay_predictions.pdf) | Model building report |
| [`data/`](data/) | Dataset notes |
