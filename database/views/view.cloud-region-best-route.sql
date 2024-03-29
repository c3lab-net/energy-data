CREATE VIEW cloud_region_best_route
AS
    WITH Filtered AS (
        WITH MinDistance AS (
            SELECT
                src_cloud,
                src_region,
                dst_cloud,
                dst_region,
                source,
                MIN(distance_km) AS min_distance
            FROM
                cloud_region_route_distribution
            GROUP BY
                src_cloud,
                src_region,
                dst_cloud,
                dst_region,
                source
        )
        SELECT
            t1.src_cloud,
            t1.src_region,
            t1.dst_cloud,
            t1.dst_region,
            t1.hop_count,
            t1.distance_km,
            t1.routers_latlon,
            t1.fiber_wkt_paths,
            t1.fiber_types,
            t1.source,
            ROW_NUMBER() OVER (PARTITION BY
                                    t1.src_cloud,
                                    t1.src_region,
                                    t1.dst_cloud,
                                    t1.dst_region,
                                    t1.source
                                ORDER BY t1.weight DESC) AS rn
        FROM
            cloud_region_route_distribution t1
        INNER JOIN
            MinDistance md
        ON
            t1.src_cloud = md.src_cloud
            AND t1.src_region = md.src_region
            AND t1.dst_cloud = md.dst_cloud
            AND t1.dst_region = md.dst_region
            AND t1.source = md.source
        WHERE
            t1.distance_km <= 1.2 * md.min_distance)
    SELECT
        f.src_cloud,
        f.src_region,
        f.dst_cloud,
        f.dst_region,
        f.hop_count,
        f.distance_km,
        f.routers_latlon,
        f.fiber_wkt_paths,
        f.fiber_types,
        f.source
    FROM
        Filtered f
    WHERE f.rn = 1