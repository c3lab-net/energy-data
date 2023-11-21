#!/usr/bin/env python3

from datetime import datetime
from flask import current_app
import psycopg2
from psycopg2 import sql
from api.helpers.carbon_intensity_shared import validate_region_exists, validate_time_range

from api.models.common import ISO_PREFIX_EMAP
from api.util import carbon_data_cache, get_psql_connection, psql_execute_list

TABLE_NAME = 'emapcarbonintensity'
REGION_COLUMN = 'zoneid'


def get_emap_region_from_iso(iso: str) -> str:
    """Transform ISO region to electricity map region."""
    if iso.startswith(ISO_PREFIX_EMAP):
        return iso.removeprefix(ISO_PREFIX_EMAP)
    raise NotImplementedError(f'Unknown EMAP region for iso {iso}')

def _get_carbon_intensity_timeseries(conn: psycopg2.extensions.connection,
                                 region: str, start: datetime, end: datetime) -> list[dict]:
    """Get the carbon intensity time series data from the database."""
    cursor = conn.cursor()
    # in case start/end lie in between two timestamps, find the timestamp <= start and >= end.
    records: list[tuple[datetime, float]] = psql_execute_list(
        cursor,
        sql.SQL("""SELECT datetime, CarbonIntensity
            FROM {table}
            WHERE zoneid = %(region)s
                AND datetime >= (SELECT COALESCE(
                    (SELECT MAX(datetime) FROM EnergyMixture
                        WHERE datetime <= %(start)s AND zoneid = %(region)s),
                    (SELECT MIN(datetime) FROM EnergyMixture
                        WHERE zoneid = %(region)s)))
                AND datetime <= (SELECT COALESCE(
                    (SELECT MIN(datetime) FROM EnergyMixture
                        WHERE datetime >= %(end)s AND zoneid = %(region)s),
                    (SELECT MAX(datetime) FROM EnergyMixture
                        WHERE zoneid = %(region)s)))
            ORDER BY datetime;""").format(table=sql.Identifier(TABLE_NAME)),
        dict(region=region, start=start, end=end))
    l_carbon_intensity = []
    for tuple in records:
        (timestamp, carbon_intensity) = tuple
        l_carbon_intensity.append({
            'timestamp': timestamp,
            'carbon_intensity': carbon_intensity,
        })
    return l_carbon_intensity

def get_carbon_intensity_list(iso: str, start: datetime, end: datetime,
        use_prediction: bool = False) -> list[dict]:
    """Retrieve the carbon intensity time series data in the given time window.

        Args:
            iso: the ISO region name.
            start: the start time.
            end: the end time.
            use_prediction: whether to use prediction or actual data.

        Returns:
            A list of time series data.
    """
    region = get_emap_region_from_iso(iso)
    if use_prediction:
        return fetch_prediction(region, start, end)
    else:
        return fetch_emissions(region, start, end)

@carbon_data_cache.memoize()
def fetch_prediction(region: str, start: datetime, end: datetime) -> list[dict]:
    current_app.logger.debug(f'fetch_prediction({region}, {start}, {end})')
    raise ValueError('Electricit map carbon data source does not support prediction')

@carbon_data_cache.memoize()
def fetch_emissions(region: str, start: datetime, end: datetime) -> list[dict]:
    current_app.logger.debug(f'fetch_emissions({region}, {start}, {end})')
    with get_psql_connection() as conn:
        validate_region_exists(conn, region, TABLE_NAME, REGION_COLUMN)
        validate_time_range(conn, region, start, end, TABLE_NAME, REGION_COLUMN)
        return _get_carbon_intensity_timeseries(conn, region, start, end)
