# Data Sources

## Canada — CRTC Communications Monitoring Reports

| Dataset | URL | Update cadence | License |
|---------|-----|----------------|---------|
| Retail Mobile Sector (CSV ZIP) | [CRTC Open Data](https://crtc.gc.ca/eng/publications/reports/policymonitoring/cmrd.htm) | Quarterly / annual | Open Government Licence – Canada |
| Retail Fixed Internet Sector | Same portal | Quarterly / annual | Open Government Licence – Canada |

**Key tables ingested:**

- `MB-S1` — Revenue and subscriber market share by service provider (Bell, TELUS, Rogers)
- `MB-S5` — Total mobile subscribers in Canada (millions)
- `MB-F5` — Provincial subscriber market share
- `MB-F17` — Top 3 vs other blended monthly churn rates

## United States — FCC Form 477

| Dataset | URL | Update cadence | License |
|---------|-----|----------------|---------|
| County-Level Connection Data | [FCC Form 477 County Data](https://www.fcc.gov/form-477-county-data-internet-access-services) | Semi-annual | US Government public domain |

Includes residential and total fixed Internet access connections by county (2009–2025).

If the live FCC download is unavailable, the pipeline falls back to `seeds/fcc_county_sample.csv` (border states: WA, NY, MI) for local development.

## Synthetic operational layer

Generated in `src/ingest/generate_subscriptions.py`:

- ~50K–100K subscriber records
- Mapped to real carrier names via `seeds/carrier_mapping.csv`
- Calibrated to CRTC benchmarks: Top 3 churn ~1.16%/month, ARPU ~$68 CAD, 37.7M total subscribers (2024)

**Clearly labeled as synthetic** in all outputs and documentation.
