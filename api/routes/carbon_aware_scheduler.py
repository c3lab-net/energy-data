#!/usr/bin/env python3
from datetime import timedelta, timezone
import json
from multiprocessing import Pool
import traceback
from typing import Any

import marshmallow_dataclass
from flask import current_app
from flask_restful import Resource
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from webargs.flaskparser import use_args
from api.helpers.balancing_authority import get_iso_from_gps

from api.helpers.carbon_intensity import calculate_total_carbon_emissions, get_carbon_intensity_list
from api.models.cloud_location import CloudLocationManager, CloudRegion, get_route_between_cloud_regions
from api.models.common import CarbonDataSource, Coordinate, ISOName, RouteInISO, get_iso_format_for_carbon_source, identify_iso_format
from api.models.optimization_engine import OptimizationEngine, OptimizationFactor
from api.models.wan_bandwidth import load_wan_bandwidth_model
from api.models.workload import DEFAULT_DC_PUE, DEFAULT_NETWORK_PUE, DEFAULT_STORAGE_POWER, CarbonAccountingMode, CloudLocation, Workload
from api.models.dataclass_extensions import *
from api.util import Rate, RateUnit, Size, SizeUnit, round_up

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

def init_lookup_iso(_carbon_data_source: CarbonDataSource):
    global carbon_data_source
    carbon_data_source = _carbon_data_source

def task_lookup_iso(region: CloudRegion) -> tuple:
    global carbon_data_source
    iso_format = get_iso_format_for_carbon_source(carbon_data_source)
    if region.iso and iso_format == identify_iso_format(region.iso):
        return str(region), region.iso, region.gps, None, None
    try:
        (latitude, longitude) = region.gps
        iso = get_iso_from_gps(latitude, longitude, iso_format)
        return str(region), iso, region.gps, None, None
    except Exception as ex:
        return str(region), None, region.gps, str(ex), traceback.format_exc()

def init_preload_carbon_data(_workload: Workload,
                                    _carbon_data_source: CarbonDataSource,
                                    _use_prediction: bool,
                                    _desired_renewable_ratio: float = None):
    global workload, carbon_data_source, use_prediction, desired_renewable_ratio
    workload = _workload
    carbon_data_source = _carbon_data_source
    use_prediction = _use_prediction
    desired_renewable_ratio = _desired_renewable_ratio

def preload_carbon_data(workload: Workload,
                        iso: str,
                        carbon_data_source: CarbonDataSource,
                        use_prediction: bool,
                        desired_renewable_ratio: float = None):
    carbon_data_store = dict()
    running_intervals = workload.get_running_intervals_in_24h()
    for (start, end) in running_intervals:
        max_delay = workload.schedule.max_delay
        carbon_data = get_carbon_intensity_list(iso, start, end + max_delay,
                                                        carbon_data_source, use_prediction,
                                                        desired_renewable_ratio)
        assert len(carbon_data) > 0 and \
            min([d['timestamp'] for d in carbon_data]) <= start and \
            max([d['timestamp'] for d in carbon_data]) >= end + max_delay, \
                f'Not enough carbon data for {iso} to cover time range [{start}, {end}]'
        carbon_data_store[(iso, start, end)] = carbon_data
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
                                    _d_candidate_routes: dict[str, list[CloudRegion]]):
    global workload, carbon_data_source, use_prediction, carbon_data_store, d_candidate_routes
    workload = _workload
    carbon_data_source = _carbon_data_source
    use_prediction = _use_prediction
    carbon_data_store = _carbon_data_store
    d_candidate_routes = _d_candidate_routes

def get_preloaded_carbon_data(iso: str, start: datetime, end: datetime) -> list[dict]:
    global carbon_data_store
    key = (iso, start, end)
    if key in carbon_data_store:
        return carbon_data_store[key]
    else:
        raise ValueError(f'No carbon data found for iso {iso} in time range ({start}, {end})')

def get_transfer_rate(route: list[CloudRegion], start: datetime, end: datetime, max_delay: timedelta) -> Rate:
    # TODO: update this to consider route
    # return g_wan_bandwidth.available_bandwidth_at(timestamp=start.time())
    return Rate(125, RateUnit.Mbps)

def get_transfer_time(data_size_gb: float, transfer_rate: Rate) -> timedelta:
    data_size = Size(data_size_gb, SizeUnit.GB)
    return data_size / transfer_rate

def get_per_hop_transfer_power_in_watts(route: CloudRegion, transfer_rate: Rate) -> float:
    # NOTE: only consider routers for now.
    CORE_ROUTER_POWER_WATT = 640
    CORE_ROUTER_CAPACITY_GBPS = 64
    return transfer_rate / Rate(CORE_ROUTER_CAPACITY_GBPS, RateUnit.Gbps) * CORE_ROUTER_POWER_WATT

def get_carbon_emission_rates_as_pd_series(iso: ISOName, start: datetime, end: datetime, power_in_watts: float) -> pd.Series:
    l_carbon_intensity = get_preloaded_carbon_data(iso, start, end)
    df = pd.DataFrame(l_carbon_intensity)
    df.set_index('timestamp', inplace=True)

    # Only consider hourly data
    df = df.loc[df.index.minute == 0]
    ds = df['carbon_intensity'].sort_index()
    # Conversion: gCO2/kWh * W * 1/(1000*3600) kh/s = gCO2/s
    ds = ds * power_in_watts / (1000 * 3600)
    ds.sort_index(inplace=True)

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

def get_compute_carbon_emission_rates(iso: ISOName, start: datetime, end: datetime, host_power_in_watts: float) -> pd.Series:
    return get_carbon_emission_rates_as_pd_series(iso, start, end, host_power_in_watts)

def get_transfer_carbon_emission_rates(route: list[CloudRegion], start: datetime, end: datetime,
                                       host_transfer_power_in_watts: float, per_hop_power_in_watts: float) -> \
                                        tuple[pd.Series,pd.Series,pd.Series]:
    if len(route) == 0: # Same region, no transfer needed.
        return [pd.Series(dtype=float),pd.Series(dtype=float),pd.Series(dtype=float)]
    # Transfer power includes both end hosts and network devices
    ds_network = pd.Series(dtype=float)
    ds_endpoints = pd.Series(dtype=float)
    for i in range(len(route)):
        hop = route[i].iso
        # Part 1: Network power consumption
        ds_hop = get_carbon_emission_rates_as_pd_series(hop, start, end, per_hop_power_in_watts)
        ds_network = ds_network.add(ds_hop, fill_value=0)
        # Part 2: End host power consumption, or first and last hop.
        if i == 0 or i == len(route) - 1:
            ds_endpoint = get_carbon_emission_rates_as_pd_series(hop, start, end, host_transfer_power_in_watts)
            ds_endpoints = ds_endpoints.add(ds_endpoint, fill_value=0)
    return (ds_network.add(ds_endpoints, fill_value=0), ds_network, ds_endpoints)

def dump_emission_rates(ds: pd.Series) -> dict:
    return json.loads(ds.to_json(orient='index', date_format='iso'))

def calculate_workload_scores(workload: Workload, region: CloudRegion) -> tuple[dict[OptimizationFactor, float], dict[str, Any]]:
    current_app.logger.info('Calculating scores for region %s ...' % region)

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

                    compute_carbon_emission_rates = get_compute_carbon_emission_rates(
                        region.iso, start, end, workload.get_power_in_watts() * DEFAULT_DC_PUE)
                    host_transfer_power = DEFAULT_STORAGE_POWER
                    all_transfer_carbon_emission_rates = get_transfer_carbon_emission_rates(
                        route, start, end,
                        host_transfer_power * DEFAULT_DC_PUE,
                        get_per_hop_transfer_power_in_watts(route, transfer_rate) * DEFAULT_NETWORK_PUE)
                    (transfer_carbon_emission_rates, \
                        transfer_network_carbon_emission_rates, \
                        transfer_endpoint_carbon_emission_rates) = all_transfer_carbon_emission_rates

                    runtime = end - start
                    (compute_carbon_emissions, transfer_carbon_emission), timings = \
                        calculate_total_carbon_emissions(start,
                                                         runtime,
                                                         max_delay,
                                                         transfer_input_time,
                                                         transfer_output_time,
                                                         compute_carbon_emission_rates,
                                                         transfer_carbon_emission_rates) \
                                                            if workload.optimize_carbon else ((0, 0), {})
                    d_scores[OptimizationFactor.CarbonEmissionFromCompute] = compute_carbon_emissions
                    d_scores[OptimizationFactor.CarbonEmissionFromMigration] = transfer_carbon_emission
                    d_misc['timings'].append(timings)
                    d_misc['route'] = [f'{hop.name} {hop.gps}' for hop in route]
                    d_misc['route.hop_count'] = len(route)
                    d_misc['emission_rates']['compute'] = dump_emission_rates(compute_carbon_emission_rates)
                    d_misc['emission_rates']['transfer'] = dump_emission_rates(transfer_carbon_emission_rates)
                    d_misc['emission_rates']['transfer.network'] = dump_emission_rates(transfer_network_carbon_emission_rates)
                    d_misc['emission_rates']['transfer.endpoint'] = dump_emission_rates(transfer_endpoint_carbon_emission_rates)
                    total_transfer_time = transfer_input_time + transfer_output_time
                    d_misc['emission_integral']['compute'] = dump_emission_rates(
                        compute_carbon_emission_rates * runtime.total_seconds())
                    d_misc['emission_integral']['transfer'] = dump_emission_rates(
                        transfer_carbon_emission_rates * total_transfer_time.total_seconds())
                    d_misc['emission_integral']['transfer.network'] = dump_emission_rates(
                        transfer_network_carbon_emission_rates * total_transfer_time.total_seconds())
                    d_misc['emission_integral']['transfer.endpoint'] = dump_emission_rates(
                        transfer_endpoint_carbon_emission_rates * total_transfer_time.total_seconds())
                    score += (compute_carbon_emissions + transfer_carbon_emission)
            case OptimizationFactor.WanNetworkUsage:
                # score = input + output data size (GB)
                # TODO: add WAN demand as weight
                score = workload.dataset.input_size_gb + workload.dataset.output_size_gb
            case _:  # Other factors ignored
                current_app.logger.info(f'Ignoring factor {factor} ...')
                score = 0
                continue
        d_scores[factor] = score
    return d_scores, d_misc

def task_process_candidate(region: CloudRegion) -> tuple:
    global workload, carbon_data_source, use_prediction
    region_name = str(region)
    iso = region.iso
    try:
        scores, d_misc = calculate_workload_scores(workload, region)
        return region_name, iso, scores, d_misc, None, None
    except Exception as ex:
        return region_name, iso, None, None, str(ex), traceback.format_exc()

def get_routes_by_region(original_location: str,
                         d_candidate_regions: dict[str, CloudRegion],
                         carbon_accounting_mode: CarbonAccountingMode) -> dict[str, list[CloudRegion]]:
    d_region_route: dict[str, list[CloudRegion]] = {}
    for candidate_region in d_candidate_regions:
        match carbon_accounting_mode:
            case CarbonAccountingMode.ComputeAndNetwork:
                d_region_route[candidate_region] = get_route_between_cloud_regions(original_location, candidate_region)
            case CarbonAccountingMode.ComputeOnly:
                d_region_route[candidate_region] = []
            case _:
                raise NotImplementedError('Unknown carbon_accounting_mode')
    return d_region_route

def assign_iso_to_route_hops(d_candidate_routes: dict[str, list[CloudRegion]],
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
                                                  args.carbon_accounting_mode)
        candidate_regions = list(d_candidate_regions.values())
        transfer_hop_regions = list(set(hop for route in d_candidate_routes.values() if route for hop in route))

        d_region_isos = dict()
        d_region_scores = dict()
        d_region_warnings = dict()
        d_misc_details = dict()
        d_transit_hop_iso = dict()

        all_regions = candidate_regions + transfer_hop_regions
        with Pool(1 if __debug__ else 4,
                  initializer=init_lookup_iso,
                  initargs=(args.carbon_data_source,)) as pool:
            result_iso = pool.map(task_lookup_iso, all_regions)
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
        carbon_data = dict()
        d_iso_errors = dict()
        with Pool(1 if __debug__ else 4,
                  initializer=init_preload_carbon_data,
                  initargs=(workload, args.carbon_data_source, args.use_prediction,
                            args.desired_renewable_ratio)
                  ) as pool:
            result = pool.map(task_preload_carbon_data, all_unique_isos)
        for (iso, partial_carbon_data, ex, stack_trace) in result:
            if partial_carbon_data:
                carbon_data |= partial_carbon_data
            else:
                d_iso_errors[iso] = ex
                current_app.logger.error(f'Carbon data lookup failed for {iso}: {ex}')
                current_app.logger.error(stack_trace)

        with Pool(1 if __debug__ else 8,
                  initializer=init_parallel_process_candidate,
                  initargs=(workload, args.carbon_data_source, args.use_prediction, carbon_data, d_candidate_routes)
                  ) as pool:
            result = pool.map(task_process_candidate, candidate_regions)
        for (region_name, iso, scores, d_misc, ex, stack_trace) in result:
            d_region_isos[region_name] = iso
            if not ex:
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

        optimal_regions, d_weighted_scores = g_optimizer.compare_candidates(d_region_scores, True)
        if not optimal_regions:
            return orig_request | {
                'error': 'No viable candidate',
                'isos': d_region_isos,
                'details': d_region_warnings
            }, 400

        return orig_request | {
            'original-region': str(args.original_location),
            'optimal-regions': optimal_regions,
            'isos': d_region_isos,
            'weighted-scores': d_weighted_scores,
            'raw-scores': d_region_scores,
            'warnings': d_region_warnings,
            'details': d_misc_details,
        }
