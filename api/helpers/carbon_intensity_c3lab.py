#!/usr/bin/env python3

import os
from flask import current_app
import numpy as np
import psycopg2
from pathlib import Path
from datetime import datetime

from api.helpers.balancing_authority import MAPPING_WATTTIME_BA_TO_C3LAB_REGION
from api.helpers.carbon_intensity_shared import validate_region_exists, validate_time_range
from api.models.common import ISO_PREFIX_C3LAB, ISO_PREFIX_WATTTIME
from api.util import load_yaml_data, get_psql_connection, psql_execute_list, carbon_data_cache

TABLE_NAME = 'energymixture'
REGION_COLUMN = 'region'

M_ISO_TO_C3LAB_REGION = MAPPING_WATTTIME_BA_TO_C3LAB_REGION

def get_c3lab_region_from_iso(iso: str) -> str:
    # Transform ISO region to c3lab region
    if iso.startswith(ISO_PREFIX_WATTTIME):
        iso = iso.removeprefix(ISO_PREFIX_WATTTIME)
        if iso not in M_ISO_TO_C3LAB_REGION:
            raise ValueError(f'Unknown c3lab region for iso {iso}')
        return M_ISO_TO_C3LAB_REGION[iso]
    elif iso.startswith(ISO_PREFIX_C3LAB):
        return iso.removeprefix(ISO_PREFIX_C3LAB)
    else:
        raise NotImplementedError(f'Unknown c3lab region for iso {iso}')

def get_map_carbon_intensity_by_fuel_source(config_path: os.path) -> dict[str, float]:
    """Load the carbon intensity per fuel source map from config."""
    # Load ISO-to-WattTime-BA mapping from yaml config
    yaml_data = load_yaml_data(config_path)
    carbon_intensity_map_name = 'carbon_intensity_by_fuel_source'
    assert yaml_data is not None and carbon_intensity_map_name in yaml_data, \
        f'Failed to load {carbon_intensity_map_name} from config.'
    return yaml_data[carbon_intensity_map_name]


MAP_CARBON_INTENSITY_BY_FUEL_SOURCE = get_map_carbon_intensity_by_fuel_source(
    os.path.join(Path(__file__).parent.absolute(), 'carbon_intensity.yaml'))
DEFAULT_CARBON_INTENSITY_FOR_UNKNOWN_SOURCE = 700


def _calculate_scaled_carbon_intensity(
            renewable_carbon_intensity: float,
            nonrenewable_carbon_intensity: float,
            current_renewable_ratio: float,
            desired_renewable_ratio: float = None) -> float:
    # Cannot scale if only one type of sources is used
    if renewable_carbon_intensity is None:
        return nonrenewable_carbon_intensity
    if nonrenewable_carbon_intensity is None:
        return renewable_carbon_intensity

    def _get_weighted_carbon_intensity(renewable_ratio: float):
        if renewable_ratio is None:
            return None
        return np.average([renewable_carbon_intensity, nonrenewable_carbon_intensity],
                          weights=[renewable_ratio, 1 - renewable_ratio])
    if desired_renewable_ratio:
        return _get_weighted_carbon_intensity(desired_renewable_ratio)
    else:
        return _get_weighted_carbon_intensity(current_renewable_ratio)

def _get_average_carbon_intensity(conn: psycopg2.extensions.connection,
                                 region: str, start: datetime, end: datetime,
                                 desired_renewable_ratio: float) -> list[dict]:
    cursor = conn.cursor()
    # in case start/end lie in between two timestamps, find the timestamp <= start and >= end.
    records: list[tuple[datetime, float]] = psql_execute_list(
        cursor,
        """SELECT datetime,
                Renewable_CarbonIntensity,
                NonRenewable_CarbonIntensity,
                Renewable_Ratio
            FROM CarbonIntensityByRenewable
            WHERE region = %(region)s
                AND datetime >= (SELECT COALESCE(
                    (SELECT MAX(datetime) FROM EnergyMixture
                        WHERE datetime <= %(start)s AND region = %(region)s),
                    (SELECT MIN(datetime) FROM EnergyMixture
                        WHERE region = %(region)s)))
                AND datetime <= (SELECT COALESCE(
                    (SELECT MIN(datetime) FROM EnergyMixture
                        WHERE datetime >= %(end)s AND region = %(region)s),
                    (SELECT MAX(datetime) FROM EnergyMixture
                        WHERE region = %(region)s)))
            ORDER BY datetime;""",
        dict(region=region, start=start, end=end))
    l_carbon_intensity = []
    for tuple in records:
        (timestamp, renewable_carbon_intensity, nonrenewable_carbon_intensity, renewable_ratio) = tuple
        carbon_intensity = _calculate_scaled_carbon_intensity(
            renewable_carbon_intensity,
            nonrenewable_carbon_intensity,
            renewable_ratio,
            desired_renewable_ratio)
        l_carbon_intensity.append({
            'timestamp': timestamp,
            'carbon_intensity': carbon_intensity,
            'renewable_ratio': desired_renewable_ratio if desired_renewable_ratio else renewable_ratio
        })
    return l_carbon_intensity


def _get_power_by_timestamp_and_fuel_source(conn: psycopg2.extensions.connection,
                                           region: str, start: datetime, end: datetime) -> \
        dict[datetime, dict[str, float]]:
    cursor = conn.cursor()
    records: list[tuple[str, datetime, datetime]] = psql_execute_list(
        cursor,
        """SELECT datetime, category, power_mw FROM EnergyMixture
            WHERE region = %s AND %s <= datetime AND datetime <= %s
            ORDER BY datetime, category;""",
        [region, start, end])
    d_power_by_timestamp_and_fuel_source: dict[datetime, dict[str, float]] = {}
    for (timestamp, category, power_mw) in records:
        if timestamp not in d_power_by_timestamp_and_fuel_source:
            d_power_by_timestamp_and_fuel_source[timestamp] = {}
        d_power_by_timestamp_and_fuel_source[timestamp][category] = power_mw
    return d_power_by_timestamp_and_fuel_source


def _calculate_average_carbon_intensity(
        power_by_timestamp_and_fuel_source: dict[datetime, dict[str, float]]) -> list[dict]:
    l_carbon_intensity_by_timestamp = []
    for timestamp, power_by_fuel_source in power_by_timestamp_and_fuel_source.items():
        l_carbon_intensity = []
        l_weight = []  # aka power in MW
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

@carbon_data_cache.memoize()
def fetch_emissions(region: str, start: datetime, end: datetime,
                                 desired_renewable_ratio: float) -> list[dict]:
    # TODO: make this not throw exception for memoize to work
    current_app.logger.debug(f'fetch_emissions({region}, {start}, {end}, {desired_renewable_ratio})')
    with get_psql_connection() as conn:
        validate_region_exists(conn, region, TABLE_NAME, REGION_COLUMN)
        validate_time_range(conn, region, start, end, TABLE_NAME, REGION_COLUMN)
        return _get_average_carbon_intensity(conn, region, start, end, desired_renewable_ratio)
        # power_by_fuel_source = get_power_by_timestamp_and_fuel_source(conn, region, start, end)
        # return _calculate_average_carbon_intensity(power_by_fuel_source)

@carbon_data_cache.memoize()
def fetch_prediction(region: str, start: datetime, end: datetime,
        desired_renewable_ratio: float) -> list[dict]:
    current_app.logger.debug(f'fetch_prediction({region}, {start}, {end}, {desired_renewable_ratio})')
    raise ValueError('c3lab carbon data source does not support prediction')

def get_carbon_intensity_list(iso: str, start: datetime, end: datetime,
        use_prediction: bool = False,
        desired_renewable_ratio: float = None) -> list[dict]:
    """Retrieve the carbon intensity time series data in the given time window.

        Args:
            region: the ISO region name.
            start: the start time.
            end: the end time.
            use_prediction: whether to use prediction or actual data.
            desired_renewable_ratio: the percentage of renewables to simulate.

        Returns:
            A list of time series data.
    """
    
    region = get_c3lab_region_from_iso(iso)
    if use_prediction:
        return fetch_prediction(region, start, end, desired_renewable_ratio)
    else:
        return fetch_emissions(region, start, end, desired_renewable_ratio)


def get_power_by_fuel_type(iso: str, start: datetime, end: datetime) -> list[dict]:
    """Retrieves the raw power (in MW) broken down by timestamp and fuel type."""
    region = get_c3lab_region_from_iso(iso)
    with get_psql_connection() as conn:
        validate_region_exists(conn, region, TABLE_NAME, REGION_COLUMN)
        validate_time_range(conn, region, start, end, TABLE_NAME, REGION_COLUMN)
        d_timestamp_fuel_power = _get_power_by_timestamp_and_fuel_source(conn, region, start, end)
    result = []
    for timestamp in d_timestamp_fuel_power:
        l_fuel_mix = []
        for fuel in d_timestamp_fuel_power[timestamp]:
            power_mw = d_timestamp_fuel_power[timestamp][fuel]
            l_fuel_mix.append({
                'type': fuel,
                'power_mw': power_mw
            })
        result.append({
            'timestamp': timestamp,
            'values': l_fuel_mix
        })
    return result
