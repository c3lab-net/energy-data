#!/usr/bin/env python3

from collections import defaultdict
import os
from pathlib import Path
import traceback
from typing import Any, Optional
from flask import current_app
from psycopg2 import sql
from werkzeug.exceptions import InternalServerError
from api.models.common import Coordinate, IsoFormat
from api.models.common import ISO_PREFIX_WATTTIME
from api.models.common import ISO_PREFIX_C3LAB
from api.models.common import ISO_PREFIX_EMAP

from api.util import CustomHTTPException, get_psql_connection, load_yaml_data, log_runtime, psql_execute_list, psql_execute_scalar, psql_execute_values, simple_cache, iso_cache
from api.external.watttime.ba_from_loc import get_watttime_ba_from_loc
from api.external.electricitymap.ba_from_loc import get_emap_ba_from_loc

YAML_CONFIG = 'balancing_authority.yaml'


def get_mapping_watttime_ba_to_region(config_path: os.path, map_name: str):
    """Load the region-to-WattTime-BA mapping from config and inverse it to provide direct lookup table"""
    # Load region-to-WattTime-BA mapping from yaml config
    yaml_data = load_yaml_data(config_path)
    assert yaml_data is not None and map_name in yaml_data, f'Failed to load {map_name}'
    reverse_mapping = yaml_data[map_name]
    # Inverse the one-to-many mapping to get direct lookup table (WattTime BA -> region)
    lookup_table = {}
    for region, l_watttime_ba in reverse_mapping.items():
        for watttime_ba in l_watttime_ba:
            assert watttime_ba not in lookup_table, \
                f"Duplicate ba in region-to-WattTime-BA mapping table: {watttime_ba}"
            lookup_table[watttime_ba] = region
    return lookup_table


WATTTIME_BA_MAPPING_FILE = os.path.join(Path(__file__).parent.absolute(), YAML_CONFIG)

MAPPING_WATTTIME_BA_TO_C3LAB_REGION = get_mapping_watttime_ba_to_region(
    WATTTIME_BA_MAPPING_FILE, 'map_c3lab_region_to_watttime_ba')
MAPPING_WATTTIME_BA_TO_AZURE_REGION = get_mapping_watttime_ba_to_region(
    WATTTIME_BA_MAPPING_FILE, 'map_azure_region_to_watttime_ba')

def convert_watttime_ba_abbrev_to_c3lab_region(watttime_abbrev) -> str:
    if watttime_abbrev in MAPPING_WATTTIME_BA_TO_C3LAB_REGION:
        return ISO_PREFIX_C3LAB + MAPPING_WATTTIME_BA_TO_C3LAB_REGION[watttime_abbrev]
    else:
        current_app.logger.warning('Unknown watttime abbrev "%s"' % watttime_abbrev)
        return ISO_PREFIX_C3LAB + 'unknown:' + watttime_abbrev


@simple_cache.memoize(timeout=0)
def lookup_watttime_balancing_authority(latitude: float, longitude: float) -> dict[str, Any]:
    """
        Lookup the balancing authority from WattTime API, and returns the WattTime lookup result."""
    current_app.logger.debug(f'lookup_watttime_balancing_authority({latitude}, {longitude})')
    watttime_response = get_watttime_ba_from_loc(latitude, longitude)
    watttime_json = watttime_response.json()

    if not watttime_response.ok:
        error = watttime_json['error'] if 'error' in watttime_json else 'Unknown'
        raise CustomHTTPException('WattTime error: %s' % error, watttime_response.status_code)

    try:
        watttime_abbrev = watttime_json['abbrev']
        watttime_name = watttime_json['name']
    except Exception as e:
        current_app.logger.error('Response: %s' % watttime_json)
        current_app.logger.error(f"Failed to parse watttime response: {e}")
        raise InternalServerError('Failed to parse WattTime API response')

    return {
        'watttime_abbrev': watttime_abbrev,
        'watttime_name': watttime_name,
    }


def lookup_emap_balancing_authority_from_api(latitude: float, longitude: float) -> str:
    """Lookup the balancing authority from EMAP API, and returns the zone id (country code)."""
    current_app.logger.debug(f'lookup_emap_balancing_authority({latitude}, {longitude})')
    emap_response = get_emap_ba_from_loc(latitude, longitude)
    emap_json = emap_response.json()

    if not emap_response.ok or emap_json.get('status', '') not in ['ok', 'no-data']:
        error = emap_json['error'] if 'error' in emap_json else 'Unknown'
        current_app.logger.error('Electricity map response: %s' % emap_response.text)
        raise CustomHTTPException('Electricity map lookup error: %s' % error, emap_response.status_code)

    try:
        return emap_json['countryCode']
    except Exception as e:
        current_app.logger.error('Response: %s' % emap_json)
        current_app.logger.error(f"Failed to parse Electricity map response: {e}")
        raise InternalServerError('Failed to parse Electricity map API response')

@log_runtime
def lookup_balancing_authorities_from_database(coordinates: list[Coordinate],
                                               iso_format: IsoFormat,
                                               exact_match = True) -> list[tuple[float, float, str]]:
    """Lookup the balancing authority from database, and returns the matched coordinate and balancing authority."""
    if len(coordinates) == 0:
        return []
    try:
        if exact_match:
            balancing_authority_table_name = 'balancing_authority'
        else:
            balancing_authority_table_name = 'balancing_authority_loose_granularity'
            d_loose_coordinate_to_exact_coordinates: dict[Coordinate, list[Coordinate]] = defaultdict(list)
            for latitude, longitude in coordinates:
                loose_coordinate = (round(latitude, 1), round(longitude, 1))
                d_loose_coordinate_to_exact_coordinates[loose_coordinate].append((latitude, longitude))
            coordinates = list(d_loose_coordinate_to_exact_coordinates.keys())
        with get_psql_connection() as conn:
            cursor = conn.cursor()
            rows = psql_execute_values(
                cursor,
                sql.SQL("""WITH input (latitude, longitude) AS (VALUES %s)
                            SELECT input.latitude, input.longitude, ba.balancing_authority
                                FROM input LEFT JOIN {table} ba ON (
                                    input.latitude = ba.latitude AND
                                    input.longitude = ba.longitude AND
                                    provider = {provider}
                                );""").format(
                                    table = sql.Identifier(balancing_authority_table_name),
                                    provider = sql.Literal(iso_format)
                                ),
                        coordinates,
                        page_size=len(coordinates))
            rows = [(float(latitude), float(longitude), ba) for latitude, longitude, ba in rows]
            if not exact_match:
                # Map back to original coordinates
                rows = [(latitude, longitude, balancing_authority)
                        for loose_lat, loose_lon, balancing_authority in rows
                        for latitude, longitude in d_loose_coordinate_to_exact_coordinates[(loose_lat, loose_lon)]]
            return rows
    except Exception as e:
        current_app.logger.error(f"Failed to lookup balancing authorities from database: {e}")
        current_app.logger.error(traceback.format_exc())
        raise e

def save_emap_balancing_authority_to_database(latitude: float, longitude: float, ba: str) -> None:
    current_app.logger.info(f'save_emap_balancing_authority_to_database({latitude}, {longitude}, {ba})')
    try:
        with get_psql_connection() as conn:
            cursor = conn.cursor()
            psql_execute_scalar(
                cursor,
                sql.SQL("""INSERT INTO balancing_authority (latitude, longitude, provider, balancing_authority)
                            VALUES (%s, %s, {provider}, %s)
                            ON CONFLICT DO NOTHING;""").format(
                                provider = sql.Literal(ba)
                                ),
                        [latitude, longitude, ba])
    except Exception as e:
        current_app.logger.warning(f"Failed to save balancing authority to database: {e}")
        # Fail silently, as this is not critical

def lookup_emap_balancing_authority(latitude: float, longitude: float) -> str:
    balancing_authority = lookup_emap_balancing_authority_from_api(latitude, longitude)
    # save_emap_balancing_authority_to_database(latitude, longitude, balancing_authority)
    return balancing_authority


def get_iso_from_gps(latitude: float, longitude: float, iso_format: IsoFormat) -> str:
    """Get the ISO region name for the given latitude and longitude, using the specified ISO format."""
    match iso_format:
        case IsoFormat.C3Lab:
            iso = convert_watttime_ba_abbrev_to_c3lab_region(
                lookup_watttime_balancing_authority(latitude, longitude)['watttime_abbrev'])
        case IsoFormat.WattTime:
            iso = ISO_PREFIX_WATTTIME + lookup_watttime_balancing_authority(latitude, longitude)['watttime_abbrev']
        case IsoFormat.EMap:
            iso = ISO_PREFIX_EMAP + lookup_emap_balancing_authority(latitude, longitude)
        case _:
            raise NotImplementedError(f'Unknown ISO format {iso_format}')
    iso_cache.set((latitude, longitude, iso_format), iso)
    return iso

@log_runtime
def get_cached_isos_from_gps(coordinates: list[Coordinate], iso_format: IsoFormat) -> list[Optional[str]]:
    length = len(coordinates)
    cached_isos = [None] * length

    d_coordinates_indices = defaultdict(list)
    # Lookup in-memory cache first
    for i in range(length):
        coordinate = coordinates[i]
        cached_iso = iso_cache.get((coordinate[0], coordinate[1], iso_format))
        if cached_iso is not None:
            cached_isos[i] = cached_iso
        else:
            # If not found, lookup in database
            d_coordinates_indices[coordinate].append(i)

    # Lookup in database
    database_query_coordinates = list(d_coordinates_indices.keys())
    match iso_format:
        case IsoFormat.EMap:
            rows_loose_match = lookup_balancing_authorities_from_database(database_query_coordinates, iso_format, False)
            rows_exact_match = lookup_balancing_authorities_from_database(database_query_coordinates, iso_format)
            for latitude, longitude, balancing_authority in rows_loose_match + rows_exact_match:
                if not balancing_authority:
                    continue
                for i in d_coordinates_indices[(latitude, longitude)]:
                    cached_isos[i] = ISO_PREFIX_EMAP + balancing_authority
                    iso_cache.set((latitude, longitude, iso_format), cached_isos[i])
        case _:
            # No caching for other ISO formats
            pass

    return cached_isos

# def get_all_balancing_authorities():
#     """Return a list of all balancing authorities for which we have collect data."""
#     with get_psql_connection() as conn:
#         cursor = conn.cursor()
#         results: list[tuple[str]] = psql_execute_list(cursor, "SELECT DISTINCT region FROM EnergyMixture ORDER BY region;")
#         return [row[0] for row in results]  # one column per row
