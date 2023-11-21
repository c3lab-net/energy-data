CREATE INDEX index_cloud_region_route_distribution_distance ON cloud_region_route_distribution
(
    src_cloud,
    src_region,
    dst_cloud,
    dst_region,
    distance_km
);
