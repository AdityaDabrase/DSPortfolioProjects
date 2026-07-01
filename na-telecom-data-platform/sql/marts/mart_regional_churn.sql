-- mart_regional_churn: synthetic ops churn vs CRTC Top 3 benchmark
CREATE OR REPLACE TABLE mart_regional_churn AS
WITH synthetic_churn AS (
    SELECT
        country,
        region_code,
        parent_group,
        COUNT(*) AS subscriber_count,
        AVG(CAST(is_churned AS DOUBLE)) AS synthetic_churn_rate,
        AVG(monthly_revenue_cad) AS avg_revenue
    FROM fct_subscriber_snapshot
    GROUP BY 1, 2, 3
),
crtc_benchmark AS (
    SELECT
        'CA' AS country,
        segment,
        CAST(year AS INTEGER) AS year,
        value_pct / 100.0 AS benchmark_churn_rate
    FROM stg_crtc_retail_mobile
    WHERE "table" = 'churn'
      AND segment = 'Top 3'
      AND year = (SELECT MAX(CAST(year AS INTEGER)) FROM stg_crtc_retail_mobile WHERE "table" = 'churn')
)
SELECT
    s.country,
    s.region_code,
    s.parent_group,
    s.subscriber_count,
    s.synthetic_churn_rate,
    b.benchmark_churn_rate,
    ABS(s.synthetic_churn_rate - b.benchmark_churn_rate) AS churn_delta_vs_benchmark,
    s.avg_revenue
FROM synthetic_churn s
LEFT JOIN crtc_benchmark b
    ON s.country = b.country
    AND s.parent_group IN ('Bell Group', 'TELUS Group', 'Rogers Group')
WHERE s.country = 'CA'
   OR s.country = 'US';
