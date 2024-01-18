#!/usr/bin/env python3

import ast
from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
from flask import current_app
from werkzeug.exceptions import NotFound
from psycopg2 import sql

from api.models.common import Coordinate
from api.util import get_psql_connection, load_yaml_data, psql_execute_list, simple_cache

@dataclass(unsafe_hash=True)
class CloudRegion:
    provider: str
    code: str
    name: str
    iso: str
    gps: Coordinate

    def __str__(self) -> str:
        return f'{self.provider}:{self.code}'


@dataclass
class PublicCloud:
    provider: str
    regions: list[CloudRegion]


class InterRegionRouteSource(str, Enum):
    ITDK = "itdk"
    IGDB_NO_POPS = "igdb.no-pops"
    IGDB_WITH_POPS = "igdb.with-pops"
    ITDK_AND_IGDB_NO_POPS = "itdk+igdb.no-pops"
    ITDK_AND_IGDB_WITH_POPS = "itdk+igdb.with-pops"


class CloudLocationManager:
    all_public_clouds: dict[str, PublicCloud] = {}

    def __init__(self) -> None:
        config_path = os.path.join(Path(__file__).parent.absolute(), 'cloud_location.yaml')
        yaml_data = load_yaml_data(config_path)
        public_cloud_list_name = 'public_clouds'
        assert yaml_data is not None and public_cloud_list_name in yaml_data, \
            f'Failed to load {public_cloud_list_name}'
        l_raw_public_clouds = yaml_data[public_cloud_list_name]
        for raw_public_cloud in l_raw_public_clouds:
            cloud_provider = raw_public_cloud['provider']
            l_raw_cloud_regions = raw_public_cloud['regions']
            l_cloud_regions: list[CloudRegion] = []
            for raw_cloud_region in l_raw_cloud_regions:
                region_code = raw_cloud_region['code']
                region_name = raw_cloud_region['name']
                region_iso = raw_cloud_region['iso'] if 'iso' in raw_cloud_region else None
                region_gps = tuple([float(coordinate) for coordinate in raw_cloud_region['gps']])
                assert len(region_gps) == 2 and abs(region_gps[0]) <= 90 and abs(region_gps[1]) <= 180, \
                    f"Invalid GPS coordinate {region_gps} for {cloud_provider}:{region_code}"
                new_cloud_region = CloudRegion(cloud_provider, region_code, region_name, region_iso, region_gps)
                l_cloud_regions.append(new_cloud_region)
            new_public_cloud = PublicCloud(cloud_provider, l_cloud_regions)
            self.all_public_clouds[cloud_provider] = new_public_cloud

    def get_all_clouds_by_provider(self) -> dict[str, PublicCloud]:
        return self.all_public_clouds

    def get_all_cloud_providers(self) -> list[str]:
        return sorted(self.all_public_clouds.keys())

    def get_all_cloud_regions(self, cloud_providers: list[str] = []) -> list[CloudRegion]:
        all_cloud_regions: list[CloudRegion] = []
        for cloud_provider in cloud_providers:
            if cloud_provider not in self.all_public_clouds:
                raise ValueError(f'Unknown cloud provider "{cloud_provider}"')
            all_cloud_regions += self.all_public_clouds[cloud_provider].regions
        return all_cloud_regions

    def get_cloud_region_codes(self, cloud_provider: str) -> list[str]:
        current_app.logger.debug('get_cloud_region_codes(%s)' % cloud_provider)
        if cloud_provider not in self.all_public_clouds:
            return []
        return [region.code for region in self.all_public_clouds[cloud_provider].regions]

    def get_gps_coordinate(self, cloud_region: CloudRegion = None, cloud_provider: str = None,
                           region_code: str = None) -> Coordinate:
        if not cloud_provider and not region_code and cloud_region:
            cloud_provider = cloud_region.provider
            region_code = cloud_region.code
        if cloud_provider not in self.all_public_clouds:
            raise NotFound('Unknown cloud provider "%s".' % cloud_provider)
        for region in self.all_public_clouds[cloud_provider].regions:
            if region.code == region_code:
                return region.gps
        raise NotFound('Unknown region "%s" for provider "%s".' % (region_code, cloud_provider))

    def get_cloud_region( self, cloud_provider: str, region_code: str) -> CloudRegion:
        if cloud_provider not in self.all_public_clouds:
            raise NotFound('Unknown cloud provider "%s".' % cloud_provider)
        for region in self.all_public_clouds[cloud_provider].regions:
            if region.code == region_code:
                return region
        raise NotFound('Unknown region "%s" for provider "%s".' % (region_code, cloud_provider))

@simple_cache.memoize(timeout=0)
def get_route_between_cloud_regions(src_cloud_region: str, dst_cloud_region: str,
                                    route_source: InterRegionRouteSource) -> \
        tuple[list[Coordinate], str, list[str]]:
    """Get the route between two cloud regions.

    Args:
        src_cloud_region: The source cloud region in the format of "provider:code".
        dst_cloud_region: The destination cloud region in the format of "provider:code".

    Returns:
        The router hop coordinates in (lat, lon)-format, the fiber paths in multi-linestring format, and the fiber types of each linestring.
    """
    if src_cloud_region == dst_cloud_region:
        return [], '', []

    current_app.logger.debug('get_route_between_cloud_regions(%s, %s)' % (src_cloud_region, dst_cloud_region))

    # records in database are in lower format
    (src_cloud, src_region) = src_cloud_region.lower().split(':', 1)
    (dst_cloud, dst_region) = dst_cloud_region.lower().split(':', 1)
    with get_psql_connection() as conn:
        cursor = conn.cursor()
        records: str = psql_execute_list(
            cursor,
            sql.SQL("""SELECT routers_latlon, fiber_wkt_paths, fiber_types FROM cloud_region_best_route
                WHERE src_cloud = %s AND src_region = %s
                    AND dst_cloud = %s AND dst_region = %s
                    AND source = {source}
                    LIMIT 1;""").format(source=sql.Literal(route_source)),
            [src_cloud, src_region, dst_cloud, dst_region])

    if len(records) < 1:
        error_message = f'No route found between {src_cloud_region} and {dst_cloud_region}'
        current_app.logger.error(error_message)
        raise ValueError(error_message)

    (routers_latlon_str, fiber_wkt_paths, fiber_types_str) = records[0]

    routers_latlon: list[Coordinate] = [ast.literal_eval(t) for t in routers_latlon_str.split('|')]
    fiber_types: list[str] = fiber_types_str.split('|') if fiber_types_str else []

    return routers_latlon, fiber_wkt_paths, fiber_types
