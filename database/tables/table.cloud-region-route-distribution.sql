CREATE TABLE cloud_region_route_distribution(
    src_cloud VARCHAR(32) NOT NULL,
    dst_cloud VARCHAR(32) NOT NULL,
    src_region VARCHAR(32) NOT NULL,
    dst_region VARCHAR(32) NOT NULL,
    weight Integer NOT NULL,
    hop_count Integer NOT NULL,
    distance_km FLOAT NOT NULL,
    routers_latlon TEXT NOT NULL,
    fiber_wkt_paths TEXT NOT NULL,
    fiber_types TEXT NOT NULL
);