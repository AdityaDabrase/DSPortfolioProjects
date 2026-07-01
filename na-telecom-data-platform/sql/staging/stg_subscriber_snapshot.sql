-- fct_subscriber_snapshot: synthetic subscriber grain
CREATE OR REPLACE TABLE fct_subscriber_snapshot AS
SELECT
    subscriber_id,
    carrier_id,
    carrier_name,
    parent_group,
    country,
    region_code,
    plan_tier,
    CAST(activation_date AS DATE) AS activation_date,
    CAST(churn_date AS DATE) AS churn_date,
    CAST(is_churned AS BOOLEAN) AS is_churned,
    CAST(is_postpaid AS BOOLEAN) AS is_postpaid,
    monthly_revenue_cad,
    CAST(snapshot_date AS DATE) AS snapshot_date,
    weight_to_population,
    ingest_ts
FROM stg_subscriptions_daily;
