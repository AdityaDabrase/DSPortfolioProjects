# Demo Walkthrough Script (optional, 3–5 minutes)

Use this outline if recording a video walkthrough of the project.

## 1. Context (30 sec)

- North American telecom data comes from public regulators: **CRTC** in Canada, **FCC** in the US
- This pipeline combines that market data with a synthetic subscriber layer and publishes analyst-ready marts

## 2. Architecture (60 sec)

- Show `assets/architecture.svg` or README diagram
- Walk through: Sources → Landing → Airflow → Warehouse → Marts → Quality checks → Summary report

## 3. Live demo (90 sec)

```bash
python scripts/run_pipeline.py
cat data/warehouse/quality_report.txt
```

- Point out quality checks passing
- Open `docs/SUMMARY_REPORT.md` and one chart from `assets/report/`

## 4. Business insight (60 sec)

- Provincial share: Rogers in BC vs AB (post-Shaw dynamics)
- Churn reconciliation: synthetic vs CRTC benchmark
- Cross-border: BC/WA example

## 5. Close (30 sec)

- Point to `docs/project_explained.md` for industry and engineering context
- Repo structure: `src/ingest`, `sql/marts`, `tests/`
