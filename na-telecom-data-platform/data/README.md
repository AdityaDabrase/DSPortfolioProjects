# Data directory

Raw downloads are gitignored (large). Regenerate with:

```bash
python scripts/download_all.py
```

| Path | Contents |
|------|----------|
| `raw/crtc/` | CRTC Retail Mobile ZIP + extracted CSVs |
| `raw/fcc/` | FCC county connection CSV (or sample fallback) |
| `raw/subscriptions/` | Reserved for exported synthetic snapshots |
| `processed/` | Parsed parquet staging files |
| `warehouse/` | DuckDB database + mart parquet exports |

Processed parquet and quality reports are kept for portfolio demo purposes.
