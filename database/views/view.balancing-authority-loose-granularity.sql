CREATE VIEW
    balancing_authority_loose_granularity
AS
WITH cte AS (
    SELECT
        CAST(latitude AS NUMERIC(5, 1)) AS latitude,
        CAST(longitude AS NUMERIC(5, 1)) AS longitude,
        provider,
        balancing_authority
    FROM balancing_authority
),
row_number_cte AS (
    SELECT latitude, longitude, provider, balancing_authority, COUNT(*) AS row_count,
           ROW_NUMBER() OVER (PARTITION BY latitude, longitude, provider ORDER BY COUNT(*) DESC) AS rn
    FROM cte
    GROUP BY latitude, longitude, provider, balancing_authority
)
SELECT latitude, longitude, provider, balancing_authority
    FROM row_number_cte
    WHERE rn = 1;
