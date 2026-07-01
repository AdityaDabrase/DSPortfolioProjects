# Understanding North American Telecom Data: A Pipeline Walkthrough

How public regulator data from **CRTC** (Canada) and **FCC** (United States) connects to market intelligence, subscriber analytics, and data engineering patterns — using a batch pipeline as the working example.

For deeper results after running the pipeline, see [SUMMARY_REPORT.md](SUMMARY_REPORT.md).

---

## Why telecom is a rich data domain

Telecommunications is one of the most regulated, metrics-heavy industries in North America. Carriers don't just sell phone plans — they report **subscriber counts**, **revenue by segment**, **churn**, **provincial market share**, and **broadband deployment** to federal agencies.

That creates two opportunities for data work:

1. **Market intelligence** — How are Bell, Rogers, TELUS, Verizon, AT&T, and T-Mobile positioned by region and over time?
2. **Operational analytics** — Do internal subscriber metrics (activations, churn, ARPU) align with what regulators publish?

This project builds a pipeline that mirrors how strategy, BI, and data platform teams combine **regulatory data** with **operational subscriber data** into one warehouse.

![Architecture diagram](../assets/architecture.svg)

---

## The business context

### Canada: CRTC and the Big Three (+ Quebecor)

The **Canadian Radio-television and Telecommunications Commission (CRTC)** publishes Communications Monitoring Report (CMR) data as open CSVs. Key carrier groups:

| Parent group | Consumer brands | Role |
|--------------|-----------------|------|
| Bell Group | Bell, Virgin Plus, Lucky Mobile | National incumbent |
| Rogers Group | Rogers, Fido, Shaw Mobile | National; AB/BC share shifted post-Shaw acquisition |
| TELUS Group | TELUS, Koodo, Public Mobile | National wireless + wireline |
| Quebecor Group | Videotron | Regional challenger (Quebec) |

The CRTC 2025 market report notes that the four largest groups accounted for **85.6%** of total telecom service revenue in 2023, with the Top 3 mobile operators holding **86.9%** subscriber share.

**Business questions this data answers:**

- Who leads mobile share in Alberta vs British Columbia?
- How fast is the total subscriber base growing?
- What blended monthly churn do Top 3 carriers report vs smaller providers?

### United States: FCC Form 477

The **Federal Communications Commission (FCC)** collects Form 477 data on fixed Internet access connections at county level. Major providers include Verizon, AT&T, T-Mobile, and Comcast (Xfinity).

**Business questions:**

- How many fixed broadband connections exist in border states (WA, NY, MI) adjacent to Canadian provinces?
- How does US fixed deployment compare when viewed alongside Canadian mobile share?

See [business_context.md](business_context.md) for carrier hierarchy detail.

---

## What this pipeline does

The **NA Telecom Market Intelligence Pipeline** is a batch ETL/ELT system with three data layers:

### Layer 1 — Regulatory market data (real)

| Source | Tables ingested | Example metrics |
|--------|-----------------|-----------------|
| CRTC Retail Mobile ZIP | MB-S1, MB-F5, MB-F17, MB-S5 | National share, provincial share, churn, total subscribers |
| FCC Form 477 county CSV | County-level connections | Fixed broadband by provider and state |

Ingest scripts download sources, parse multi-row CRTC headers, normalize to long-format parquet, and land in a raw/processed zone (local disk or GCS).

### Layer 2 — Synthetic subscriber operations

Real customer-level churn data is not public. The pipeline generates **documented synthetic** subscriber records (~75K) mapped to real carrier names via `seeds/carrier_mapping.csv`, calibrated to CRTC benchmarks:

- Top 3 blended churn ~**1.16%**/month
- ARPU ~**$68 CAD**
- Total subscribers ~**37.7M** (2024)

This layer demonstrates how operational DE patterns (daily snapshots, churn flags, revenue fields) sit alongside regulatory marts.

### Layer 3 — Warehouse marts

SQL transforms build a star-style model in DuckDB (local) or BigQuery (cloud):

| Layer | Examples |
|-------|----------|
| Staging | `stg_crtc_retail_mobile`, `stg_fcc_county_connections`, `stg_subscriptions_daily` |
| Dimensions | `dim_carrier`, `dim_region`, `dim_date` |
| Facts | `fct_market_metrics`, `fct_subscriber_snapshot` |
| Marts | `mart_carrier_market_share`, `mart_regional_churn`, `mart_cross_border_summary` |

---

## Data engineering concepts in practice

### Batch orchestration (Airflow)

The DAG `na_telecom_pipeline` runs daily with **TaskGroups**:

- `canada_market` — CRTC ingest
- `us_market` — FCC ingest
- `operational` — synthetic subscriber generation
- `load_staging` → transforms → quality checks → summary report

This mirrors how teams split ownership by domain while keeping one coordinated schedule.

### Idempotent ingest

Downloads skip if files exist; parsers handle CRTC encoding (UTF-8 / Latin-1), percentage strings, and year columns like `2024 (MP)`. Regulatory data rarely arrives clean — robust parsing is part of the job.

### Data quality as validation, not an afterthought

Eight automated checks run after each build:

- No null carrier IDs in dimensions
- Marts non-empty
- Synthetic churn reconciles to CRTC Top 3 benchmark within tolerance
- No duplicate grain keys
- Subscriber totals in sane range (37.7M for 2024)
- Freshness metadata present

Quality results appear in `data/warehouse/quality_report.txt` and inside [SUMMARY_REPORT.md](SUMMARY_REPORT.md).

### Cloud-ready design

Local mode uses DuckDB + Parquet. Cloud mode (optional) loads the same staging files to **GCS** and **BigQuery** via environment flags — same logic, different landing zone.

---

## Analytics concepts illustrated

### Market share analysis

Provincial mobile share (CRTC MB-F5) shows **regional competition** matters as much as national totals. Example from the latest pipeline run (2024):

| Province | Rogers | TELUS | Bell Group |
|----------|--------|-------|------------|
| BC | 40.9% | 41.6% | 16.8% |
| AB | 27.9% | 49.1% | 22.8% |
| ON | 45.1% | 21.4% | 31.2% |

Western Canada tells a different story than Ontario — relevant for strategy, pricing, and network investment discussions.

### Subscriber growth

MB-S5 tracks total Canadian mobile subscribers rising to **37.7 million** in 2024 — a macro indicator of market maturity and penetration.

### Churn benchmarking

CRTC publishes blended monthly churn for Top 3 vs other providers (MB-F17). The pipeline compares synthetic group-level churn against that benchmark — a pattern used when internal ops data must be sanity-checked against external references.

### Cross-border view

Pairing CA provinces with adjacent US states (BC/WA, ON/NY, ON/MI) connects mobile share on one side with fixed broadband deployment on the other — useful for North America market narratives.

All six charts and a written conclusion live in [SUMMARY_REPORT.md](SUMMARY_REPORT.md).

---

## How the pieces tie together (business → engineering)

```text
Leadership question:  "How is Rogers doing in the West after Shaw?"
        ↓
Data source:          CRTC MB-F5 provincial subscriber share
        ↓
Engineering:          Ingest → stage → mart_carrier_market_share
        ↓
Analytics output:     Rogers 40.9% in BC, 27.9% in AB (2024)
        ↓
Validation:           Quality checks + summary report
```

The pipeline is not the analysis itself — it is the **repeatable system** that makes the analysis available on schedule, with documented lineage and tests.

---

## Explore the repository

| Doc / path | Contents |
|------------|----------|
| [SUMMARY_REPORT.md](SUMMARY_REPORT.md) | Conclusions, charts, quality results |
| [architecture.md](architecture.md) | Data flow and deployment modes |
| [data_sources.md](data_sources.md) | CRTC/FCC links and licenses |
| [business_context.md](business_context.md) | Carrier structure and market structure |
| `scripts/run_pipeline.py` | End-to-end local run |
| `sql/marts/` | Mart definitions |
| `assets/report/` | Generated visualizations |

### Run it locally

```bash
pip install -r requirements-dev.txt
python scripts/download_all.py
python scripts/run_pipeline.py
```

---

## Limitations (read before interpreting results)

- **Subscriber data is synthetic** — only CRTC/FCC market metrics are from official open data.
- **FCC ingest** may use a border-state sample when the full multi-GB county file is unavailable.
- **CRTC 2020** metrics include reporting breaks (NA values) due to methodology changes.

---

## Further reading

- [CRTC Communications Monitoring Reports — Open Data](https://crtc.gc.ca/eng/publications/reports/policymonitoring/cmrd.htm)
- [CRTC Canadian Telecommunications Market Report 2025](https://crtc.gc.ca/eng/publications/reports/PolicyMonitoring/2025/ctmr.htm)
- [FCC Form 477 County Data](https://www.fcc.gov/form-477-county-data-internet-access-services)

---

*Independent learning project. Not affiliated with any carrier or regulator.*
