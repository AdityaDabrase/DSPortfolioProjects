-- mart_cross_border_summary: adjacent border region pairs
CREATE OR REPLACE TABLE mart_cross_border_summary AS
WITH pairs AS (
    SELECT * FROM (VALUES
        ('BC', 'WA'),
        ('ON', 'NY'),
        ('ON', 'MI')
    ) AS t(ca_region, us_region)
),
ca_metrics AS (
    SELECT
        region_code,
        AVG(market_share_pct) AS avg_ca_share
    FROM mart_carrier_market_share
    WHERE country = 'CA'
      AND metric_type = 'provincial_subscriber_share'
      AND year = (SELECT MAX(year) FROM mart_carrier_market_share WHERE country = 'CA' AND market_share_pct IS NOT NULL)
      AND carrier IN ('Rogers Group', 'Rogers')
    GROUP BY 1
),
us_metrics AS (
    SELECT
        region_code,
        SUM(total_connections) AS us_connections
    FROM mart_carrier_market_share
    WHERE country = 'US'
      AND metric_type = 'fixed_broadband_connections'
      AND year = (SELECT MAX(year) FROM mart_carrier_market_share WHERE country = 'US')
    GROUP BY 1
)
SELECT
    p.ca_region,
    p.us_region,
    c.avg_ca_share AS rogers_group_ca_provincial_share_pct,
    u.us_connections AS us_border_state_fixed_connections
FROM pairs p
LEFT JOIN ca_metrics c ON p.ca_region = c.region_code
LEFT JOIN us_metrics u ON p.us_region = u.region_code;
