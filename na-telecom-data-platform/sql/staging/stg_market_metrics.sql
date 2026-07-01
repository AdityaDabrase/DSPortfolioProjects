-- fct_market_metrics: unified long-format market metrics from CRTC + FCC
CREATE OR REPLACE TABLE fct_market_metrics AS
SELECT
    'CA' AS country,
    COALESCE(provider, provider_group, segment) AS carrier_or_segment,
    COALESCE(province, 'National') AS region_code,
    CAST(year AS INTEGER) AS period_year,
    metric_type,
    COALESCE(value_pct, value_millions) AS metric_value,
    CASE WHEN value_pct IS NOT NULL THEN 'pct' WHEN value_millions IS NOT NULL THEN 'millions' END AS metric_unit,
    ingest_ts
FROM stg_crtc_retail_mobile
WHERE year IS NOT NULL
UNION ALL
SELECT
    'US' AS country,
    provider AS carrier_or_segment,
    state_abbr AS region_code,
    CAST(report_year AS INTEGER) AS period_year,
    'fixed_broadband_connections' AS metric_type,
    CAST(total_connections AS DOUBLE) AS metric_value,
    'connections' AS metric_unit,
    ingest_ts
FROM stg_fcc_county_connections;
