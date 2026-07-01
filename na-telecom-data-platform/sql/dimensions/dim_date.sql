-- dim_date: date spine from subscription snapshots and CRTC years
CREATE OR REPLACE TABLE dim_date AS
SELECT DISTINCT
    CAST(snapshot_date AS DATE) AS date_key,
    EXTRACT(YEAR FROM CAST(snapshot_date AS DATE)) AS year,
    EXTRACT(MONTH FROM CAST(snapshot_date AS DATE)) AS month
FROM stg_subscriptions_daily
WHERE snapshot_date IS NOT NULL;
