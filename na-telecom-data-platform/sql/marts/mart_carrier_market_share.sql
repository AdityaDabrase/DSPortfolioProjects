-- mart_carrier_market_share: Canada mobile share + US fixed connections by region
CREATE OR REPLACE TABLE mart_carrier_market_share AS
WITH ca_share AS (
    SELECT
        'CA' AS country,
        provider AS carrier,
        CAST(year AS INTEGER) AS year,
        'National' AS region_code,
        value_pct AS market_share_pct,
        metric_type
    FROM stg_crtc_retail_mobile
    WHERE "table" = 'market_share'
      AND metric_type = 'subscriber_share'
      AND provider IN ('Bell', 'TELUS', 'Rogers')
),
ca_provincial AS (
    SELECT
        'CA' AS country,
        provider_group AS carrier,
        CAST(year AS INTEGER) AS year,
        province AS region_code,
        value_pct AS market_share_pct,
        'provincial_subscriber_share' AS metric_type
    FROM stg_crtc_retail_mobile
    WHERE "table" = 'provincial_share'
      AND provider_group IN ('Bell Group', 'TELUS', 'Rogers')
),
us_fixed AS (
    SELECT
        'US' AS country,
        provider AS carrier,
        CAST(report_year AS INTEGER) AS year,
        state_abbr AS region_code,
        SUM(total_connections) AS total_connections,
        'fixed_broadband_connections' AS metric_type
    FROM stg_fcc_county_connections
    GROUP BY 1, 2, 3, 4, 6
)
SELECT country, carrier, year, region_code, market_share_pct, NULL::DOUBLE AS total_connections, metric_type
FROM ca_share
UNION ALL
SELECT country, carrier, year, region_code, market_share_pct, NULL::DOUBLE, metric_type
FROM ca_provincial
UNION ALL
SELECT country, carrier, year, region_code, NULL::DOUBLE AS market_share_pct, total_connections, metric_type
FROM us_fixed;
