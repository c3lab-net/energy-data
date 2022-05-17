#!/usr/bin/env python3

import os
import numpy as np
import psycopg2
from pathlib import Path
from datetime import datetime
from flask_restful import Resource
from webargs import fields
from webargs.flaskparser import use_kwargs

from api.util import logger, loadYamlData, get_psql_connection, psql_execute_list, psql_execute_scalar
from api.resources.balancing_authority import convert_watttime_ba_abbrev_to_region, lookup_watttime_balancing_authority

class ElectricityDataLookupException(Exception):
    pass

def get_map_carbon_intensity_by_fuel_source(config_path: os.path) -> dict[str, float]:
    '''Load the carbon intensity per fuel source map from config.'''
    # Load ISO-to-WattTime-BA mapping from yaml config
    yaml_data = loadYamlData(config_path)
    carbon_intensity_map_name = 'carbon_intensity_by_fuel_source'
    assert yaml_data is not None and carbon_intensity_map_name in yaml_data, \
        f'Failed to load {carbon_intensity_map_name} from config.'
    return yaml_data[carbon_intensity_map_name]

MAP_CARBON_INTENSITY_BY_FUEL_SOURCE = get_map_carbon_intensity_by_fuel_source(os.path.join(Path(__file__).parent.absolute(), 'carbon_intensity.yaml'))
DEFAULT_CARBON_INTENSITY_FOR_UNKNOWN_SOURCE = 700

def validate_region_exists(conn: psycopg2.extensions.connection, region: str) -> None:
    cursor = conn.cursor()
    region_exists = psql_execute_scalar(cursor,
        "SELECT EXISTS(SELECT 1 FROM EnergyMixture WHERE region = %s)",
        [region])
    if not region_exists:
        raise ElectricityDataLookupException(f"Region {region} doesn't exist in database.")

def get_matching_timestamp(conn: psycopg2.extensions.connection, region: str, timestamp: datetime) -> datetime:
    """Get the matching stamp in electricity generation records for the given time."""
    cursor = conn.cursor()
    timestamp_before: datetime|None = psql_execute_scalar(cursor,
        "SELECT MAX(DateTime) FROM EnergyMixture WHERE Region = %s AND DateTime <= %s;"
        , [region, timestamp])
    timestamp_after: datetime|None = psql_execute_scalar(cursor,
        "SELECT MIN(DateTime) FROM EnergyMixture WHERE Region = %s AND DateTime >= %s;"
        , [region, timestamp])
    if timestamp_before is None:
        raise ElectricityDataLookupException("Time range is too old. No data available.")
    if timestamp_after is None:
        raise ElectricityDataLookupException("Time range is too new. Data not yet available.")
    assert timestamp_before <= timestamp_after, "before must be less than after"
    # Always choose the beginning of the period
    return timestamp_before

def get_power_by_timemstamp_and_fuel_source(conn: psycopg2.extensions.connection, region: str, start: datetime, end: datetime) -> dict[datetime, dict[str, float]]:
    cursor = conn.cursor()
    records: list[tuple[str, float]] = psql_execute_list(cursor,
        """SELECT datetime, category, power_mw FROM EnergyMixture
            WHERE region = %s AND %s <= datetime AND datetime <= %s
            ORDER BY datetime, category;""",
        [region, start, end])
    d_power_bytimestamp_and_fuel_source: dict[datetime, dict[str, float]] = {}
    for (timestamp, category, power_mw) in records:
        if timestamp not in d_power_bytimestamp_and_fuel_source:
            d_power_bytimestamp_and_fuel_source[timestamp] = {}
        d_power_bytimestamp_and_fuel_source[timestamp][category] = power_mw
    return d_power_bytimestamp_and_fuel_source

def calculate_average_carbon_intensity(power_by_timestamp_and_fuel_source: dict[datetime, dict[str, float]]) -> dict[datetime, float]:
    l_carbon_intensity_by_timestamp = []
    for timestamp, power_by_fuel_source in power_by_timestamp_and_fuel_source.items():
        l_carbon_intensity = []
        l_weight = []   # aka power in MW
        for fuel_source, power_in_mw in power_by_fuel_source.items():
            if fuel_source not in MAP_CARBON_INTENSITY_BY_FUEL_SOURCE:
                carbon_intensity = DEFAULT_CARBON_INTENSITY_FOR_UNKNOWN_SOURCE
            else:
                carbon_intensity = MAP_CARBON_INTENSITY_BY_FUEL_SOURCE[fuel_source]
            l_carbon_intensity.append(carbon_intensity)
            l_weight.append(power_in_mw)
        average_carbon_intensity = np.average(l_carbon_intensity, weights=l_weight)
        l_carbon_intensity_by_timestamp.append({
            'timestamp': timestamp,
            'carbon_intensity': average_carbon_intensity,
        })
    return l_carbon_intensity_by_timestamp

def get_carbon_intensity_list(region: str, start: datetime, end: datetime) -> float:
    conn = get_psql_connection()
    validate_region_exists(conn, region)
    matching_start = get_matching_timestamp(conn, region, start)
    matching_end = get_matching_timestamp(conn, region, end)
    power_by_fuel_source = get_power_by_timemstamp_and_fuel_source(conn, region, matching_start, matching_end)
    return calculate_average_carbon_intensity(power_by_fuel_source)

carbon_intensity_args = {
    'latitude': fields.Float(required=True, validate=lambda x: abs(x) <= 90.),
    'longitude': fields.Float(required=True, validate=lambda x: abs(x) <= 180.),
    'start': fields.DateTime(format="iso", required=True),
    'end': fields.DateTime(format="iso", required=True),
}

class CarbonIntensity(Resource):
    @use_kwargs(carbon_intensity_args, location='query')
    def get(self, latitude: float, longitude: float, start: datetime, end: datetime):
        orig_request = { 'request': {
            'latitude': latitude,
            'longitude': longitude,
            'start': start,
            'end': end,
        } }
        logger.info("get(%f, %f, %s, %s)" % (latitude, longitude, start, end))

        watttime_lookup_result, error_status_code = lookup_watttime_balancing_authority(latitude, longitude)
        if error_status_code:
            return orig_request | watttime_lookup_result, error_status_code

        region = convert_watttime_ba_abbrev_to_region(watttime_lookup_result['watttime_abbrev'])
        try:
            l_carbon_intensity = get_carbon_intensity_list(region, start, end)
        except ElectricityDataLookupException as e:
            return orig_request | {
                'error': str(e)
            }, 500

        return orig_request | watttime_lookup_result | {
            'region': region,
            'carbon_intensities': l_carbon_intensity,
        }
