-- dim_carrier: canonical carrier dimension from seed + staging
CREATE OR REPLACE TABLE dim_carrier AS
SELECT
    carrier_id,
    display_name,
    parent_group,
    country,
    CAST(is_mno AS BOOLEAN) AS is_mno,
    CAST(is_flanker_brand AS BOOLEAN) AS is_flanker_brand,
    market_weight_ca,
    market_weight_us
FROM seed_carrier_mapping;
