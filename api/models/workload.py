#!/usr/bin/env python3

from datetime import timedelta, timezone
from enum import Enum
from typing import Optional

import numpy as np
from marshmallow import validates_schema, ValidationError
from marshmallow_dataclass import dataclass
from api.models.common import CarbonDataSource

# from api.models.common import Coordinate
from api.models.cloud_location import CloudLocationManager
from api.models.dataclass_extensions import *

g_cloud_manager = CloudLocationManager()


class ScheduleType(Enum):
    UNIFORM_RANDOM = "uniform-random"
    POISSON = "poisson"
    ONETIME = "onetime"


class CarbonAccountingMode(str, Enum):
    ComputeOnly = "compute-only"
    ComputeAndNetwork = "compute-and-network"


@dataclass
class WorkloadSchedule:
    type: ScheduleType = field_enum(ScheduleType)
    start_time: Optional[datetime] = optional_field_with_validation(validate_is_timezone_aware)
    interval: Optional[timedelta] = field(metadata=metadata_timedelta_nonzero, default=None)
    max_delay: Optional[timedelta] = field(metadata=metadata_timedelta, default=timedelta())

    @validates_schema
    def validate_schema(self, data, **kwargs):
        errors = dict()
        if 'type' not in data:
            return
        if data['type'] is ScheduleType.ONETIME and 'interval' in data and data['interval']:
            errors['interval'] = 'interval must be empty for one-time workload'
        if data['type'] is not ScheduleType.ONETIME and ('interval' not in data or not data['interval']):
            errors['interval'] = 'interval must be specified for recurring workload'
        if errors:
            raise ValidationError(errors)
        if data.get('start_time', None) is None:
            data['start_time'] = datetime.now(timezone.utc)


@dataclass
class Dataset:
    input_size_gb: float = field_with_validation(validate_number_is_nonnegative)
    output_size_gb: float = field_with_validation(validate_number_is_nonnegative)


@dataclass
class CloudLocation:
    id: str = field_with_validation(validate.Regexp(r'([^:]*):([^:]*)'))
    latitude: Optional[float] = optional_field_with_validation(validate.Range(-90, 90))
    longitude: Optional[float] = optional_field_with_validation(validate.Range(-180, 180))


# Average of Xeon Platinum 8275CL, which has 48 HTs and a TDP of 240W
DEFAULT_CPU_TDP = 240
DEFAULT_CPU_POWER_PER_CORE = DEFAULT_CPU_TDP / 48  # in watt
# Storage system consumes roughly 20% of total DC energy, based on Borroso book.
DEFAULT_STORAGE_POWER = DEFAULT_CPU_TDP * 0.2
DEFAULT_DC_PUE = 1
DEFAULT_NETWORK_PUE = 2

ALL_CLOUD_PROVIDERS = g_cloud_manager.get_all_cloud_providers()

def _validate_providers(candidate_providers: list[str]):
    errors = dict()
    existing_providers = set()
    for i in range(len(candidate_providers)):
        provider = candidate_providers[i]
        if provider in existing_providers:
            errors[i] = f'Duplicate cloud provider "{provider}"'
        existing_providers.add(provider)
        if provider not in ALL_CLOUD_PROVIDERS:
            errors[i] = f'Unknown cloud provider "{provider}"'
    return errors

def _validate_locations(candidate_locations: list[CloudLocation]):
    errors = dict()
    existing_locations = set()
    for i in range(len(candidate_locations)):
        cloud_location = candidate_locations[i]
        if cloud_location.id in existing_locations:
            errors[i] = { 'id': f'Duplicate location "{cloud_location.id}"' }
            continue
        existing_locations.add(cloud_location.id)
        [provider, region] = cloud_location.id.split(':', 1)
        if provider in ALL_CLOUD_PROVIDERS and region in g_cloud_manager.get_cloud_region_codes(provider):
            # known location, no coordinates are needed
            error_message_name_conflict = 'Location is in pre-defined list. Must leave coordinates empty or choose a different name'
            if cloud_location.latitude:
                errors[i]['latitude'] = error_message_name_conflict
            if cloud_location.longitude:
                errors[i]['longitude'] = error_message_name_conflict
            continue
        # Unknown location, coordinates are needed
        if not cloud_location.latitude or not cloud_location.longitude:
            errors[i] = {}
            if not cloud_location.latitude:
                errors[i]['latitude'] = 'Must provide latitude for unknown location'
            if not cloud_location.longitude:
                errors[i]['longitude'] = 'Must provide longitude for unknown location'
    return errors

def _validate_location_is_defined(location: str, candidate_locations: list[CloudLocation]):
    splitted = location.split(':', 1)
    if len(splitted) != 2:
        return 'Location must be in the format of "provider:region"'
    [provider, region] = splitted
    if provider in ALL_CLOUD_PROVIDERS and region in g_cloud_manager.get_cloud_region_codes(provider):
        return None
    elif location in [location.id for location in candidate_locations]:
        return None
    else:
        return 'Location not defined'


@dataclass
class Workload:
    runtime: timedelta = field(metadata=metadata_timedelta_nonzero)
    schedule: WorkloadSchedule
    dataset: Dataset

    original_location: Optional[str] = field()
    candidate_providers: Optional[list[str]] = field(default_factory=list)
    candidate_locations: Optional[list[CloudLocation]] = field(default_factory=list)

    # TODO: add custom routes support
    # custom_routes: Optional[dict[str, RouteInCoordinate]] = field(default_factory=dict)

    carbon_data_source: CarbonDataSource = field_enum(CarbonDataSource, CarbonDataSource.C3Lab)
    use_prediction: bool = field(default=False)
    desired_renewable_ratio: Optional[float] = \
        optional_field_with_validation(lambda ratio: 0. <= ratio <= 1.)
    optimize_carbon: bool = field(default=True)

    watts_per_core: float = field(default=DEFAULT_CPU_POWER_PER_CORE)
    core_count: float = field(default=1.)

    carbon_accounting_mode: CarbonAccountingMode = field_enum(CarbonAccountingMode, CarbonAccountingMode.ComputeOnly)

    @validates_schema
    def validate_schema(self, data, **kwargs):
        errors = dict()
        if bool(data.get('candidate_providers', None)) == bool(data.get('candidate_locations', None)):
            errors['candidate_providers'] = errors['candidate_locations'] = \
                'Must provide one of candidate_providers and candidate_locations'
            raise ValidationError(errors)
        if data.get('candidate_providers', None):
            sub_errors = _validate_providers(data['candidate_providers'])
            if sub_errors:
                errors['candidate_providers'] = sub_errors
        else:
            sub_errors = _validate_locations(data['candidate_locations'])
            if sub_errors:
                errors['candidate_locations'] = sub_errors
        if data.get('carbon_accounting_mode', None) is CarbonAccountingMode.ComputeAndNetwork and \
                not data.get('original_location', None):
            errors['original_location'] = \
                f'Must provide original_location for carbon_accounting_mode {CarbonAccountingMode.ComputeAndNetwork}'
        if data.get('original_location', None):
            sub_errors = _validate_location_is_defined(data['original_location'], data['candidate_locations'])
            if sub_errors:
                errors['original_location'] = sub_errors
        if errors:
            raise ValidationError(errors)

    def get_cputime_in_24h(self) -> timedelta:
        run_count = 0
        match self.schedule.type:
            case ScheduleType.ONETIME:
                run_count = 1
            case ScheduleType.POISSON:
                run_count = timedelta(days=1) // self.schedule.interval
            case ScheduleType.UNIFORM_RANDOM:
                run_count = timedelta(days=1) // self.schedule.interval
            case _:
                raise NotImplementedError()
        return self.runtime * run_count

    def get_running_intervals_in_24h(self) -> list[tuple[datetime, datetime]]:
        intervals = []

        def _add_current_run_to_interval(start: datetime):
            intervals.append((start, start + self.runtime))

        match self.schedule.type:
            case ScheduleType.ONETIME:
                _add_current_run_to_interval(self.schedule.start_time)
            case ScheduleType.POISSON:
                current_start = self.schedule.start_time
                lam = 1. / self.schedule.interval.total_seconds()
                # Draw poisson distribution every second
                distribution_size = int(timedelta(days=1).total_seconds())
                poisson_distributions = np.random.poisson(lam, distribution_size)
                for i in range(poisson_distributions):
                    for _ in range(poisson_distributions[i]):
                        time_elapsed = timedelta(seconds=i)
                        start = self.schedule.start_time + time_elapsed
                        _add_current_run_to_interval(start)
            case ScheduleType.UNIFORM_RANDOM:
                current_start = self.schedule.start_time
                while current_start < self.schedule.start_time + timedelta(days=1):
                    _add_current_run_to_interval(current_start)
                    current_start += self.schedule.interval
            case _:
                raise NotImplementedError()
        return intervals

    def get_power_in_watts(self) -> float:
        return self.watts_per_core * self.core_count

    def get_energy_usage_24h(self) -> float:
        """Get the energy usage in a 24h period, in kWh."""
        cpu_usage_hours = self.get_cputime_in_24h().total_seconds() / timedelta(hours=1).total_seconds()
        return self.get_power_in_watts() / 1000 * cpu_usage_hours
