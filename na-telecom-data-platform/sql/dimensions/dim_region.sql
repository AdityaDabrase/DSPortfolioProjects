-- dim_region: provinces, states, and counties from staging sources
CREATE OR REPLACE TABLE dim_region AS
SELECT DISTINCT
    region_code,
    country,
    region_code AS region_name,
    CASE WHEN country = 'CA' THEN 'province' WHEN country = 'US' THEN 'state' END AS region_type
FROM stg_subscriptions_daily
UNION
SELECT DISTINCT
    province AS region_code,
    'CA' AS country,
    province AS region_name,
    'province' AS region_type
FROM stg_crtc_retail_mobile
WHERE "table" = 'provincial_share' AND province IS NOT NULL AND province <> ''
UNION
SELECT DISTINCT
    state_abbr AS region_code,
    'US' AS country,
    county_name AS region_name,
    'county' AS region_type
FROM stg_fcc_county_connections
WHERE state_abbr IS NOT NULL;
