#!/usr/bin/env python3
from collections import defaultdict
from datetime import timedelta, timezone
import json
from math import ceil
import math
from multiprocessing import Pool
import os
from pathlib import Path
import traceback
from typing import Any

import marshmallow_dataclass
from flask import current_app
from flask_restful import Resource
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from webargs.flaskparser import use_args

from api.helpers.balancing_authority import get_cached_isos_from_gps, get_iso_from_gps
from api.helpers.carbon_intensity import get_carbon_intensity_list, calculate_total_carbon_emissions_linear, calculate_total_carbon_emissions_naive
from api.models.cloud_location import CloudLocationManager, CloudRegion, get_route_between_cloud_regions
from api.models.common import CarbonDataSource, Coordinate, ISOName, IsoFormat, get_iso_format_for_carbon_source, identify_iso_format
from api.models.network_device import NetworkDevice, NetworkDeviceType, create_network_devices
from api.models.optimization_engine import OptimizationEngine, OptimizationFactor
from api.models.wan_bandwidth import load_wan_bandwidth_model
from api.models.workload import DEFAULT_DC_PUE, DEFAULT_NETWORK_PUE, DEFAULT_NETWORK_REDUNDANCY, DEFAULT_STORAGE_POWER, CarbonAccountingMode, CloudLocation, InterRegionRouteSource, NetworkHopCarbonEstimationHeuristic, Workload
from api.models.dataclass_extensions import *
from api.util import Rate, RateUnit, Size, SizeUnit, load_yaml_data, round_down, round_up, log_runtime

g_cloud_manager = CloudLocationManager()
OPTIMIZATION_FACTORS_AND_WEIGHTS = [
    (OptimizationFactor.EnergyUsage, 1000),
    (OptimizationFactor.CarbonEmission, 1),
    (OptimizationFactor.WanNetworkUsage, 0.001),
]
g_optimizer = OptimizationEngine([t[0] for t in OPTIMIZATION_FACTORS_AND_WEIGHTS],
                                 [t[1] for t in OPTIMIZATION_FACTORS_AND_WEIGHTS])
g_wan_bandwidth = load_wan_bandwidth_model()


def get_candidate_regions(candidate_providers: list[str], candidate_locations: list[CloudLocation],
                          original_location: str) \
        -> dict[str, CloudRegion]:
    try:
        if candidate_providers:
            candidate_regions = g_cloud_manager.get_all_cloud_regions(candidate_providers)
            d_candidate_regions = { str(region): region for region in candidate_regions }
            if original_location:
                assert original_location in d_candidate_regions, "Original location not defined in candidate regions"
            return d_candidate_regions

        d_candidate_regions = {}
        if original_location:
            candidate_locations += [CloudLocation(original_location)]
        for location in candidate_locations:
            if location.id in d_candidate_regions:
                continue
            (provider, region_name) = location.id.split(':', 1)
            if location.latitude and location.longitude:
                gps = (location.latitude, location.longitude)
                cloud_region = CloudRegion(provider, region_name, location.id, None, gps)
            else:
                cloud_region = g_cloud_manager.get_cloud_region(provider, region_name)
            d_candidate_regions[str(cloud_region)] = cloud_region
        return d_candidate_regions
    except Exception as ex:
        raise ValueError(f'Failed to get candidate regions: {ex}') from ex

@log_runtime
def lookup_all_isos(regions: list[CloudRegion|NetworkDevice],
                    carbon_data_source: CarbonDataSource) -> list[tuple]:
    iso_format = get_iso_format_for_carbon_source(carbon_data_source)
    coordinates_to_indices: dict[Coordinate, list[int]] = defaultdict(list)
    for i in range(len(regions)):
        region = regions[i]
        # Re-query if ISO format is different.
        if region.iso and iso_format != identify_iso_format(region.iso):
            region.iso = None
        if not region.iso:
            coordinates_to_indices[region.gps].append(i)

    # Query for cached ISOs first
    coordinates = list(coordinates_to_indices.keys())
    cached_isos = get_cached_isos_from_gps(coordinates, carbon_data_source)
    for coordinate, cached_iso in zip(coordinates, cached_isos):
        if not cached_iso:
            continue
        for i in coordinates_to_indices[coordinate]:
            regions[i].iso = cached_iso
    cached_region_count = len([region for region in regions if region.iso])
    current_app.logger.info(f'ISO cached/total: {cached_region_count}/{len(regions)}')

    # Fill in the rest with new queries
    results = []
    for region in regions:
        if region.iso:
            results.append((str(region), region.iso, region.gps, None, None))
        else:
            try:
                (latitude, longitude) = region.gps
                iso = get_iso_from_gps(latitude, longitude, iso_format)
                results.append((str(region), iso, region.gps, None, None))
            except Exception as ex:
                results.append((str(region), None, region.gps, str(ex), traceback.format_exc()))
    return results

def init_preload_carbon_data(_workload: Workload,
                                    _carbon_data_source: CarbonDataSource,
                                    _use_prediction: bool,
                                    _desired_renewable_ratio: float = None):
    global workload, carbon_data_source, use_prediction, desired_renewable_ratio
    workload = _workload
    carbon_data_source = _carbon_data_source
    use_prediction = _use_prediction
    desired_renewable_ratio = _desired_renewable_ratio

def get_emap_full_range_carbon_isos(config_path: os.path) -> list[str]:
    """Load the list of emap ISOs with full yearly coverage."""
    # Load ISO-to-WattTime-BA mapping from yaml config
    yaml_data = load_yaml_data(config_path)
    carbon_intensity_map_name = 'emap_full_range_carbon_isos'
    assert yaml_data is not None and carbon_intensity_map_name in yaml_data, \
        f'Failed to load {carbon_intensity_map_name} from config.'
    isos = yaml_data[carbon_intensity_map_name]
    return [f'{IsoFormat.EMap}:{iso}' for iso in isos]

EMAP_FULL_RANGE_CARBON_ISOS = get_emap_full_range_carbon_isos(
    os.path.join(Path(__file__).parent.absolute(), 'emap_full_range_carbon_isos.yaml'))

def preload_carbon_data(workload: Workload,
                        iso: str,
                        carbon_data_source: CarbonDataSource,
                        use_prediction: bool,
                        desired_renewable_ratio: float = None):
    carbon_data_store = dict()
    running_intervals = workload.get_running_intervals_in_24h()
    for (start, end) in running_intervals:
        max_delay = workload.schedule.max_delay
        l_carbon_intensity = get_carbon_intensity_list(iso, start, end + max_delay,
                                                       carbon_data_source, use_prediction,
                                                       desired_renewable_ratio)
        assert len(l_carbon_intensity) > 0, f'No carbon data for {iso} in time range [{start}, {end}]'
        carbon_data_store[(iso, start, end)] = \
            convert_carbon_intensity_to_pd_series(iso, l_carbon_intensity, start, end + max_delay)
    return carbon_data_store

def task_preload_carbon_data(iso: str) -> tuple:
    global workload, carbon_data_source, use_prediction, desired_renewable_ratio
    try:
        carbon_data = preload_carbon_data(workload, iso, carbon_data_source, use_prediction, desired_renewable_ratio)
        return iso, carbon_data, None, None
    except Exception as ex:
        return iso, None, str(ex), traceback.format_exc()

def init_parallel_process_candidate(_workload: Workload,
                                    _carbon_data_source: CarbonDataSource,
                                    _use_prediction: bool,
                                    _carbon_data_store: dict,
                                    _d_candidate_routes: dict[str, list[NetworkDevice]],
                                    _origin_iso: ISOName):
    global workload, carbon_data_source, use_prediction, carbon_data_store, d_candidate_routes, origin_iso
    workload = _workload
    carbon_data_source = _carbon_data_source
    use_prediction = _use_prediction
    carbon_data_store = _carbon_data_store
    d_candidate_routes = _d_candidate_routes
    origin_iso = _origin_iso

def get_preloaded_carbon_data(iso: str, start: datetime, end: datetime,
                              only_emap_full_range_isos: bool = False) -> pd.Series:
    global carbon_data_store
    key = (iso, start, end)
    if key in carbon_data_store and \
            (not (only_emap_full_range_isos and iso not in EMAP_FULL_RANGE_CARBON_ISOS)):
        return carbon_data_store[key]
    else:
        raise ValueError(f'No carbon data found for iso {iso} in time range ({start}, {end})')

def has_carbon_data(iso: str, start: datetime, end: datetime,
                    only_emap_full_range_isos: bool = False) -> bool:
    global carbon_data_store
    if only_emap_full_range_isos and iso not in EMAP_FULL_RANGE_CARBON_ISOS:
        return False
    key = (iso, start, end)
    return key in carbon_data_store

def get_transfer_rate(route: list[NetworkDevice], start: datetime, end: datetime, max_delay: timedelta) -> Rate:
    # TODO: update this to consider route
    # return g_wan_bandwidth.available_bandwidth_at(timestamp=start.time())
    return Rate(1024, RateUnit.Mbps)

def get_transfer_time(data_size_gb: float, transfer_rate: Rate) -> timedelta:
    data_size = Size(data_size_gb, SizeUnit.GB)
    transfer_time: timedelta = (data_size / transfer_rate)
    # Round to whole seconds for a later algorithm.
    return timedelta(seconds=ceil(transfer_time.total_seconds()))

def convert_carbon_intensity_to_pd_series(iso: ISOName, l_carbon_intensity: list[dict],
                                          start: datetime, end: datetime) -> pd.Series:
    df = pd.DataFrame(l_carbon_intensity)
    # Force conversion using UTC is needed to handle multiple timezones
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    ds = df['carbon_intensity']

    # Only consider hourly data, using average carbon intensity.
    granularity = timedelta(hours=1)
    # Remove data outside of the time range, to avoid returning more data than requested
    ds = ds.loc[round_down(start, granularity):round_up(end, granularity)]
    # Resample to hourly granularity
    ds = ds.resample(granularity).mean()

    assert ds.index.min() <= start and end <= ds.index.max(), \
        f'Carbon data not available for iso {iso} for the entire time range [{start}, {end}]'

    # Insert end-of-time index with zero value to avoid out-of-bound read corner case handling
    if len(ds.index) < 2:
        ds_freq = pd.DateOffset(hours=1)
    else:
        ds_freq = to_offset(np.diff(ds.index).min())
        # pd.infer_freq() only works with perfectly regular frequency
        # ds_freq = to_offset(pd.infer_freq(ds.index))
    end_time_of_series = ds.index.max() + ds_freq
    ds[end_time_of_series.to_pydatetime()] = 0.

    return ds

def calculate_carbon_emission_rates(carbon_intensity: pd.Series, power_in_watts: float):
    """Calculate carbon emission rate in gCO2/s."""
    # Conversion: gCO2/kWh * W * 1/(1000*3600) kh/s = gCO2/s
    carbon_emission_rate = carbon_intensity * power_in_watts / (1000 * 3600)
    carbon_emission_rate.sort_index(inplace=True)
    return carbon_emission_rate

def get_carbon_emission_rates(iso: ISOName, start: datetime, end: datetime, power_in_watts: float,
                              only_emap_full_range_isos: bool = False) -> pd.Series:
    carbon_intensity = get_preloaded_carbon_data(iso, start, end, only_emap_full_range_isos)
    return calculate_carbon_emission_rates(carbon_intensity, power_in_watts)

def get_constant_carbon_emission_rates(carbon_intensity_value: float, reference: pd.Series,
                                       power_in_watts: float) -> pd.Series:
    carbon_intensity = pd.Series(carbon_intensity_value, index=reference.index)
    return calculate_carbon_emission_rates(carbon_intensity, power_in_watts)

def get_network_carbon_emission_rates_with_estimation(route: list[NetworkDevice],
                                                      start: datetime, end: datetime,
                                                      transfer_rate: Rate,
                                                      estimation_heuristic: NetworkHopCarbonEstimationHeuristic,
                                                      carbon_estimation_minimum_known_carbon_power_ratio: float,
                                                      carbon_estimation_route_average_ratio_threshold: float,
                                                      carbon_estimation_distance_km_threshold: float,
                                                      only_emap_full_range_isos) -> \
                                                        tuple[pd.Series, float]:
    """Calcualte total network carbon emission rates, including estimated carbon cost for hops without carbon data.

        Returns:
            ds_network: carbon emission rates for the entire network
            power_ratio_with_carbon_data: ratio of network power with carbon data to total network power
    """
    def _get_nearst_neighbor_index_with_carbon_data(route: list[NetworkDevice],
                                                    target_index: int,
                                                    no_carbon_isos: set[ISOName],
                                                    distance_km_threshold: float) -> int:
        """Find the nearest neighbor with carbon data, within a distance threshold."""
        def _hop_distance_km(i, j) -> float:
            distance_i = route[i].distance_from_route_start_km
            distance_j = route[j].distance_from_route_start_km
            return abs(distance_i - distance_j)

        candidate_index_left = next((i for i in range(target_index - 1, -1, -1) \
                                      if route[i].iso not in no_carbon_isos), None)
        candidate_index_right = next((i for i in range(target_index + 1, len(route)) \
                                       if route[i].iso not in no_carbon_isos), None)
        candidate_indices = filter(lambda i: i is not None, [candidate_index_left, candidate_index_right])
        return next((t[0] for t in sorted([(i, _hop_distance_km(target_index, i)) for i in candidate_indices],
                                          key=lambda t: t[1]) if t[1] <= distance_km_threshold), -1)

    def _get_per_hop_power_in_watts(hop: NetworkDevice) -> float:
        return hop.get_energy_intensity_w_per_gbps() * transfer_rate.gbps() * \
            DEFAULT_NETWORK_PUE * DEFAULT_NETWORK_REDUNDANCY

    # Keep track of ISOs without carbon data, in case we need to estimate for these hops later.
    no_carbon_isos: set[ISOName] = set()

    # Group devices by their ISO to minimize timeseries operations.
    total_network_power_per_iso: dict[ISOName, float] = defaultdict(float)
    for hop in route:
        iso = hop.iso
        per_hop_power_in_watts = _get_per_hop_power_in_watts(hop)
        total_network_power_per_iso[iso] += per_hop_power_in_watts
        # Keep track of ISOs without carbon data, in case we need to estimate for these hops later.
        if not has_carbon_data(iso, start, end, only_emap_full_range_isos):
            no_carbon_isos.add(iso)

    # Sum up carbon time series for all hops with carbon data.
    ds_network = pd.Series(dtype=float)
    for iso, network_power_per_iso in total_network_power_per_iso.items():
        if iso in no_carbon_isos:
            continue
        ds_hop = get_carbon_emission_rates(iso, start, end, network_power_per_iso, only_emap_full_range_isos)
        ds_network = ds_network.add(ds_hop, fill_value=0)

    # All hops have carbon data, no need to estimate.
    if len(no_carbon_isos) == 0:
        return ds_network, 1.

    # Estimate carbon emission rate for hops without carbon data.
    sum_network_power_without_carbon_data = sum(total_network_power_per_iso[iso] for iso in no_carbon_isos)
    total_network_power = sum(total_network_power_per_iso.values())
    power_ratio_with_carbon_data = 1 - sum_network_power_without_carbon_data / total_network_power

    error_message = f'Carbon data is only available for {power_ratio_with_carbon_data * 100:.2f}% ' \
                    f'of total network power.\n' \
                    f'ISOs with missing carbon data ({len(no_carbon_isos)}): {", ".join(no_carbon_isos)}.\n'

    if power_ratio_with_carbon_data < carbon_estimation_minimum_known_carbon_power_ratio:
        raise ValueError(error_message + 'Available known power ratio is below the threshold ' + \
            f'({carbon_estimation_minimum_known_carbon_power_ratio * 100:.2f}%).')

    match estimation_heuristic:
        case NetworkHopCarbonEstimationHeuristic.RouteAverage:
            # If carbon data is not available for a small portion of the hops (in total power), we'll use re-scale
            #   carbon emission rate based on this ratio of power with carbon data and total power.
            if power_ratio_with_carbon_data >= carbon_estimation_route_average_ratio_threshold:
                ds_network /= power_ratio_with_carbon_data
            else:
                raise ValueError(error_message + \
                    f'Only {(power_ratio_with_carbon_data * 100):.2f}% of the network power has carbon data, '
                    f'but we need at least {(carbon_estimation_route_average_ratio_threshold * 100):.2f}%. ')
        case NetworkHopCarbonEstimationHeuristic.NearestNeighbor:
            # Use the nearest neighbor with carbon data to estimate carbon emission rate.
                # For re-mapping ISOs to nearest neighbors, if it doesn't have carbon data.
            total_network_power_per_remapped_iso: dict[ISOName, float] = defaultdict(float)
            remap_failed_hop_count = 0
            for i in range(len(route)):
                if route[i].iso not in no_carbon_isos:
                    continue
                nearest_neighbor_index = _get_nearst_neighbor_index_with_carbon_data(
                    route, i, no_carbon_isos, carbon_estimation_distance_km_threshold)
                if nearest_neighbor_index >= 0:
                    remapped_iso = route[nearest_neighbor_index].iso
                    total_network_power_per_remapped_iso[remapped_iso] += _get_per_hop_power_in_watts(route[i])
                else:
                    remap_failed_hop_count += 1
            if remap_failed_hop_count > 0:
                raise ValueError(error_message + \
                                 f'No nearest neighbor with carbon data found for {remap_failed_hop_count} hops '
                                 f'(distance threshold: {carbon_estimation_distance_km_threshold}).')
            for iso, network_power_per_iso in total_network_power_per_remapped_iso.items():
                ds_hop = get_carbon_emission_rates(iso, start, end, network_power_per_iso, only_emap_full_range_isos)
                ds_network = ds_network.add(ds_hop, fill_value=0)
        case NetworkHopCarbonEstimationHeuristic.WorldAverage:
            # Source: https://www.iea.org/reports/global-energy-co2-status-report-2019/emissions
            WORLD_AVERAGE_CARBON_INTENSITY = 475.0 # gCO2/kWh
            ds_estimate = get_constant_carbon_emission_rates(WORLD_AVERAGE_CARBON_INTENSITY, ds_network,
                                                             sum_network_power_without_carbon_data)
            ds_network = ds_network.add(ds_estimate, fill_value=0)
        case NetworkHopCarbonEstimationHeuristic.NoEstimation:
            raise ValueError(error_message + f'No estimation is specified.')
        case _:
            raise NotImplementedError('Unknown NetworkHopCarbonEstimationHeuristic')
    return ds_network, power_ratio_with_carbon_data

def get_transfer_carbon_emission_rates(route: list[NetworkDevice], start: datetime, end: datetime,
                                       src_iso: ISOName, dst_iso: ISOName,
                                       transfer_rate: Rate, host_transfer_power_in_watts: float,
                                       estimation_heuristic: NetworkHopCarbonEstimationHeuristic,
                                       carbon_estimation_minimum_known_carbon_power_ratio: float,
                                       carbon_estimation_route_average_ratio_threshold: float,
                                       carbon_estimation_distance_km_threshold: float,
                                       only_emap_full_range_isos: bool) -> \
                                        tuple[pd.Series,pd.Series,pd.Series, float]:
    if len(route) == 0: # Same region, no transfer needed.
        return [pd.Series(dtype=float),pd.Series(dtype=float),pd.Series(dtype=float), None]

    # Transfer power includes both end hosts and network devices

    # Part 1: End host power consumption, at the locations of the first and last hop.
    ds_endpoints = pd.Series(dtype=float)
    for endpoint_iso in [src_iso, dst_iso]:
        ds_endpoint = get_carbon_emission_rates(endpoint_iso, start, end, host_transfer_power_in_watts, only_emap_full_range_isos)
        ds_endpoints = ds_endpoints.add(ds_endpoint, fill_value=0)

    # Part 2: Network power consumption from all devices along the route.
    ds_network, power_ratio_with_carbon_data = \
        get_network_carbon_emission_rates_with_estimation(route, start, end, transfer_rate, estimation_heuristic,
                                                          carbon_estimation_minimum_known_carbon_power_ratio,
                                                          carbon_estimation_route_average_ratio_threshold,
                                                          carbon_estimation_distance_km_threshold,
                                                          only_emap_full_range_isos)

    return (ds_network.add(ds_endpoints, fill_value=0), ds_network, ds_endpoints, power_ratio_with_carbon_data)

def dump_emission_rates(ds: pd.Series) -> dict:
    if not ds.empty:
        # Remove last value if it is zero, which is a placeholder for end-of-time
        if ds.iloc[-1] == 0:
            ds = ds.iloc[:-1]
    return json.loads(ds.to_json(orient='index', date_format='iso'))

def calculate_workload_scores(workload: Workload, region: CloudRegion,
                              origin_iso: ISOName) -> tuple[dict[OptimizationFactor, float], dict[str, Any]]:
    current_app.logger.debug('Calculating scores for region %s ...' % region)

    global d_candidate_routes
    d_scores = {}
    d_misc = {}
    for factor in OptimizationFactor:
        match factor:
            case OptimizationFactor.EnergyUsage:
                # score = per-core power (kW) * cpu usage (h)
                score = workload.get_energy_usage_24h()
                # TODO: add data transfer energy cost
            case OptimizationFactor.CarbonEmissionFromCompute: continue
            case OptimizationFactor.CarbonEmissionFromMigration: continue
            case OptimizationFactor.CarbonEmission:
                # score = energy usage (kWh) * grid carbon intensity (kgCO2/kWh)
                running_intervals = workload.get_running_intervals_in_24h()
                max_delay = workload.schedule.max_delay
                route = d_candidate_routes[str(region)]
                if route is None:
                    raise ValueError(f'No route found for {region}')
                score = 0
                d_misc['timings'] = []
                d_misc['emission_rates'] = {}
                d_misc['emission_integral'] = {}
                # 24 hour / 5 min = 288 slots
                for (start, end) in running_intervals:
                    transfer_rate = get_transfer_rate(route, start, end, max_delay)
                    transfer_input_time = get_transfer_time(workload.dataset.input_size_gb, transfer_rate) if route else timedelta()
                    transfer_output_time = get_transfer_time(workload.dataset.output_size_gb, transfer_rate) if route else timedelta()

                    compute_carbon_emission_rates = get_carbon_emission_rates(
                        region.iso, start, end, workload.get_power_in_watts() * DEFAULT_DC_PUE)
                    all_transfer_carbon_emission_rates = get_transfer_carbon_emission_rates(
                        route, start, end,
                        origin_iso, region.iso,
                        transfer_rate,
                        DEFAULT_STORAGE_POWER * DEFAULT_DC_PUE,
                        workload.network_hop_carbon_estimation_heuristic,
                        workload.network_hop_carbon_estimation_minimum_known_carbon_power_ratio,
                        workload.network_hop_carbon_estimation_route_average_ratio_threshold,
                        workload.network_hop_carbon_estimation_distance_km_threshold,
                        workload.only_emap_full_range_isos_for_network_hops)
                    (transfer_carbon_emission_rates, \
                        transfer_network_carbon_emission_rates, \
                        transfer_endpoint_carbon_emission_rates,
                        power_ratio_with_carbon_data) = all_transfer_carbon_emission_rates

                    if workload.use_new_optimization:
                        calculate_total_carbon_emissions = calculate_total_carbon_emissions_linear
                    else:
                        calculate_total_carbon_emissions = calculate_total_carbon_emissions_naive

                    runtime = end - start
                    if workload.optimize_carbon:
                        (compute_carbon_emissions, transfer_carbon_emission), timings = \
                            calculate_total_carbon_emissions(start,
                                                            runtime,
                                                            max_delay,
                                                            transfer_input_time,
                                                            transfer_output_time,
                                                            compute_carbon_emission_rates,
                                                            transfer_carbon_emission_rates)
                    else:
                        (compute_carbon_emissions, transfer_carbon_emission), timings = \
                            ((math.nan, math.nan), {
                                'min_start': start,
                                'max_end': end + max_delay,
                                'compute_duration': runtime,
                                'input_transfer_duration': transfer_input_time,
                                'output_transfer_duration': transfer_output_time,
                                'total_transfer_time': transfer_input_time + transfer_output_time,
                            })
                    d_scores[OptimizationFactor.CarbonEmissionFromCompute] = compute_carbon_emissions
                    d_scores[OptimizationFactor.CarbonEmissionFromMigration] = transfer_carbon_emission
                    d_misc['timings'].append(timings)
                    d_misc['route'] = [str(dev) for dev in route]
                    is_router = lambda d: d.device_type == NetworkDeviceType.ROUTER
                    d_misc['route.hop_count'] = sum(1 for dev in route if is_router(dev))
                    if power_ratio_with_carbon_data:
                        d_misc['transfer.network.power_ratio_with_carbon_data'] = power_ratio_with_carbon_data
                    d_misc['emission_rates']['compute'] = dump_emission_rates(compute_carbon_emission_rates)
                    d_misc['emission_rates']['transfer'] = dump_emission_rates(transfer_carbon_emission_rates)
                    d_misc['emission_rates']['transfer.network'] = dump_emission_rates(transfer_network_carbon_emission_rates)
                    d_misc['emission_rates']['transfer.endpoint'] = dump_emission_rates(transfer_endpoint_carbon_emission_rates)
                    # total_transfer_time = transfer_input_time + transfer_output_time
                    # d_misc['emission_integral']['compute'] = dump_emission_rates(
                    #     compute_carbon_emission_rates * runtime.total_seconds())
                    # d_misc['emission_integral']['transfer'] = dump_emission_rates(
                    #     transfer_carbon_emission_rates * total_transfer_time.total_seconds())
                    # d_misc['emission_integral']['transfer.network'] = dump_emission_rates(
                    #     transfer_network_carbon_emission_rates * total_transfer_time.total_seconds())
                    # d_misc['emission_integral']['transfer.endpoint'] = dump_emission_rates(
                    #     transfer_endpoint_carbon_emission_rates * total_transfer_time.total_seconds())
                    score += (compute_carbon_emissions + transfer_carbon_emission)
            case OptimizationFactor.WanNetworkUsage:
                # score = input + output data size (GB)
                # TODO: add WAN demand as weight
                score = workload.dataset.input_size_gb + workload.dataset.output_size_gb
            case _:  # Other factors ignored
                current_app.logger.debug(f'Ignoring factor {factor} ...')
                score = 0
                continue
        d_scores[factor] = score
    return d_scores, d_misc

def task_process_candidate(region: CloudRegion) -> tuple:
    global workload, carbon_data_source, use_prediction, origin_iso
    region_name = str(region)
    iso = region.iso
    try:
        scores, d_misc = calculate_workload_scores(workload, region, origin_iso)
        return region_name, iso, scores, d_misc, None, None
    except Exception as ex:
        return region_name, iso, None, None, str(ex), traceback.format_exc()

@log_runtime
def get_routes_by_region(original_location: str,
                         d_candidate_regions: dict[str, CloudRegion],
                         carbon_accounting_mode: CarbonAccountingMode,
                         inter_region_route_source: InterRegionRouteSource) -> dict[str, list[NetworkDevice]]:
    d_region_route: dict[str, list[NetworkDevice]] = {}
    for candidate_region in d_candidate_regions:
        match carbon_accounting_mode:
            case CarbonAccountingMode.ComputeAndNetwork:
                try:
                    route_info = get_route_between_cloud_regions(original_location, candidate_region,
                                                                 inter_region_route_source)
                    d_region_route[candidate_region] = create_network_devices(*route_info)
                except Exception as ex:
                    current_app.logger.error(f'Failed to get route between {original_location} and {candidate_region}: {ex}')
                    current_app.logger.error(traceback.format_exc())
                    d_region_route[candidate_region] = None
            case CarbonAccountingMode.ComputeOnly:
                d_region_route[candidate_region] = []
            case _:
                raise NotImplementedError('Unknown carbon_accounting_mode')
    return d_region_route

def assign_iso_to_route_hops(d_candidate_routes: dict[str, list[NetworkDevice]],
                             d_transit_hop_iso: dict[Coordinate, ISOName]) -> None:
    """Assign ISO to transit hops in the route."""
    for candidate in d_candidate_routes:
        if d_candidate_routes[candidate] is None:
            continue
        for i in range(len(d_candidate_routes[candidate])):
            hop = d_candidate_routes[candidate][i]
            if hop.iso:
                continue
            if hop.gps in d_transit_hop_iso:
                d_candidate_routes[candidate][i].iso = d_transit_hop_iso[hop.gps]
            else:
                d_candidate_routes[candidate][i].iso = None

class CarbonAwareScheduler(Resource):
    @log_runtime
    @use_args(marshmallow_dataclass.class_schema(Workload)())
    def get(self, args: Workload):
        workload = args
        orig_request = {'request': workload}
        current_app.logger.info("CarbonAwareScheduler.get(%s)" % workload)

        if workload.use_prediction:
            min_start_time = round_up(datetime.now(timezone.utc), timedelta(minutes=5))
            if workload.schedule.start_time < min_start_time:
                workload.schedule.start_time = min_start_time

        d_candidate_regions = get_candidate_regions(args.candidate_providers,
                                                  args.candidate_locations,
                                                  args.original_location)
        d_candidate_routes = get_routes_by_region(args.original_location, d_candidate_regions,
                                                  args.carbon_accounting_mode, args.inter_region_route_source)
        candidate_regions = list(d_candidate_regions.values())
        transfer_hops = list(set(hop for route in d_candidate_routes.values() if route for hop in route))

        d_region_isos = dict()
        d_region_scores = dict()
        d_region_warnings = dict()
        d_misc_details = dict()
        d_transit_hop_iso = dict()

        current_app.logger.info(f'Looking up ISO for {len(candidate_regions)} regions and '
                                f'{len(transfer_hops)} unique transit hops ...')
        all_regions = candidate_regions + transfer_hops
        result_iso = lookup_all_isos(all_regions, args.carbon_data_source)
        for i in range(len(all_regions)):
            (region_name, iso, gps, ex, stack_trace) = result_iso[i]
            if iso:
                if i < len(candidate_regions):
                    d_region_isos[region_name] = iso
                    candidate_regions[i].iso = iso
                else:
                    d_transit_hop_iso[gps] = iso
            else:
                d_region_warnings[region_name] = ex
                current_app.logger.error(f'ISO lookup failed for {region_name}: {ex}')
                current_app.logger.error(stack_trace)
        assign_iso_to_route_hops(d_candidate_routes, d_transit_hop_iso)

        all_unique_isos = set(d_region_isos.values()) | set(d_transit_hop_iso.values())
        current_app.logger.info(f'Loading carbon data for {len(all_unique_isos)} unique regions ...')
        carbon_data = dict()
        d_iso_errors = dict()
        @log_runtime
        def _preload_all_carbon_data(all_unique_isos):
            with Pool(1 if __debug__ else 4,
                    initializer=init_preload_carbon_data,
                    initargs=(workload, args.carbon_data_source, args.use_prediction,
                                args.desired_renewable_ratio)
                    ) as pool:
                return pool.map(task_preload_carbon_data, all_unique_isos)
        result = _preload_all_carbon_data(all_unique_isos)
        for (iso, partial_carbon_data, ex, stack_trace) in result:
            if partial_carbon_data:
                carbon_data |= partial_carbon_data
            else:
                d_iso_errors[iso] = ex
                current_app.logger.error(f'Carbon data lookup failed for {iso}: {ex}')
                current_app.logger.error(stack_trace)

        current_app.logger.info(f'Calculating scores for {len(candidate_regions)} regions ...')
        @log_runtime
        def _process_all_candidates(candidate_regions):
            with Pool(1 if __debug__ else 8,
                    initializer=init_parallel_process_candidate,
                    initargs=(workload, args.carbon_data_source, args.use_prediction, carbon_data, d_candidate_routes,
                              d_region_isos[workload.original_location])
                    ) as pool:
                return pool.map(task_process_candidate, candidate_regions)
        result = _process_all_candidates(candidate_regions)
        for (region_name, iso, scores, d_misc, ex, stack_trace) in result:
            d_region_isos[region_name] = iso
            if not (ex or stack_trace):
                scores[OptimizationFactor.EnergyUsage + '-unit'] = 'kWh'
                scores[OptimizationFactor.CarbonEmission + '-unit'] = 'gCO2'
                d_region_scores[region_name] = scores
                d_misc_details[region_name] = d_misc
            else:
                if d_iso_errors.get(iso, None):
                    d_region_warnings[region_name] = d_iso_errors[iso]
                else:
                    d_region_warnings[region_name] = str(ex)
                    current_app.logger.error(f'Exception when calculating score for region {region_name}: {ex}')
                    current_app.logger.error(stack_trace)

        current_app.logger.info(f'Comparing {len(d_region_scores)} regions ...')
        optimal_regions, d_weighted_scores = g_optimizer.compare_candidates(d_region_scores, True)
        response = orig_request | {
            'original-region': str(args.original_location),
            'isos': d_region_isos,
            'warnings': d_region_warnings,
            'details': d_misc_details,
        }

        if not optimal_regions:
            return response | {
                'error': 'No viable candidate',
            }, 400
        else:
            return response | {
                'optimal-regions': optimal_regions,
                'weighted-scores': d_weighted_scores,
                'raw-scores': d_region_scores,
            }
