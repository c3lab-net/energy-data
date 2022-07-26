#!/usr/bin/env python3

from datetime import datetime, timezone
from flask_restful import Resource
from webargs import fields
from webargs.flaskparser import use_args, use_kwargs
import marshmallow_dataclass
from flask import current_app

from api.helpers.carbon_intensity import convert_carbon_intensity_list_to_dict, calculate_total_carbon_emissions, get_carbon_intensity_list
from api.resources.balancing_authority import convert_watttime_ba_abbrev_to_region, lookup_watttime_balancing_authority
from api.models.cloud_location import CloudLocationManager, CloudRegion
from api.models.workload import DEFAULT_CPU_POWER_PER_CORE, Workload
from api.models.optimization_engine import OptimizationEngine, OptimizationFactor

g_cloud_manager = CloudLocationManager()
OPTIMIZATION_FACTORS_AND_WEIGHTS = [
    (OptimizationFactor.CarbonIntensity, 1),
    (OptimizationFactor.WanNetworkUsage, 1),
]
g_optimizer = OptimizationEngine([t[0] for t in OPTIMIZATION_FACTORS_AND_WEIGHTS],
                                [t[1] for t in OPTIMIZATION_FACTORS_AND_WEIGHTS])

def get_alternative_regions(cloud_region: CloudRegion, include_self = False) -> list[CloudRegion]:
    # NOTE: returns all possible regions for now, but can add filter/preference later.
    return g_cloud_manager.get_all_cloud_regions()

def calculate_workload_scores(workload: Workload, cloud_region: CloudRegion, iso_region: str) -> dict[OptimizationFactor, float]:
    d_scores = {}
    for factor in OptimizationFactor:
        match factor:
            case OptimizationFactor.EnergyUsage:
                # score = per-core power (kW) * cpu usage (h)
                score = workload.get_energy_usage_24h()
            case OptimizationFactor.CarbonIntensity:
                # score = energy usage (kWh) * grid carbon intensity (kgCO2/kWh)
                running_intervals = workload.get_running_intervals_in_24h()
                score = 0
                for (start, end) in running_intervals:
                    l_carbon_intensity = get_carbon_intensity_list(iso_region, start, end)
                    carbon_intensity_by_timestamp = convert_carbon_intensity_list_to_dict(l_carbon_intensity)
                    score += calculate_total_carbon_emissions(start, end, DEFAULT_CPU_POWER_PER_CORE, carbon_intensity_by_timestamp)
            case OptimizationFactor.WanNetworkUsage:
                # score = input + output data size (GB)
                # TODO: add WAN demand as weight
                score = workload.dataset.input_size_gb + workload.dataset.output_size_gb
            case _: # Other factors ignored
                score = 0
        d_scores[factor] = score
    return d_scores

carbon_aware_scheduler_args = {
    'start': fields.DateTime(format="iso", required=True),
    'end': fields.DateTime(format="iso", required=True),
}

class CarbonAwareScheduler(Resource):
    @use_kwargs(carbon_aware_scheduler_args, location='query')
    @use_args(marshmallow_dataclass.class_schema(Workload)())
    def get(self, args: Workload, start: datetime, end: datetime):
        workload = args
        orig_request = { 'request': {
            'start': start,
            'end': end,
            'workload': workload
        } }
        current_app.logger.info("CarbonAwareScheduler.get(%s, %s, %s)" % (start, end, workload))
        if workload.schedule.start_time is None:
            workload.schedule.start_time = datetime.now(timezone.utc)

        # TODO: disamguite cloud region and ISO region
        candidate_cloud_regions = get_alternative_regions(args.preferred_cloud_location, True)
        candidate_iso_regions = []
        for candidate_cloud_region in candidate_cloud_regions:
            (latitude, longitude) = g_cloud_manager.get_gps_coordinate(candidate_cloud_region)
            watttime_lookup_result = lookup_watttime_balancing_authority(latitude, longitude)
            iso_region = convert_watttime_ba_abbrev_to_region(watttime_lookup_result['watttime_abbrev'])
            candidate_iso_regions.append(iso_region)
        l_region_scores = []
        l_region_names = []
        for i in range(len(candidate_cloud_regions)):
            l_region_scores.append(calculate_workload_scores(workload,
                candidate_cloud_regions[i],
                candidate_iso_regions[i]))
            l_region_names.append(str(candidate_cloud_regions[i]))
        index_best_region, l_weighted_score = g_optimizer.compare_candidates(l_region_scores, True)
        selected_region = l_region_names[index_best_region]

        return orig_request | watttime_lookup_result | {
            'requested-region': str(args.preferred_cloud_location),
            'selected-region': selected_region,
            'scores': { l_region_names[i]: l_weighted_score for i in range(candidate_cloud_regions) }
        }
