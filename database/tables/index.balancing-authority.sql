CREATE UNIQUE INDEX index_balancing_authority
    ON balancing_authority (latitude, longitude, provider, balancing_authority);
