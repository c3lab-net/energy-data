#!/usr/bin/env python3

from enum import Enum
from dataclasses import dataclass
import math
import os.path
from pathlib import Path
import traceback
from flask import current_app

from shapely.wkt import loads
from geopy.distance import distance, lonlat
from shapely.geometry import Point, LineString, MultiLineString

from api.models.common import Coordinate, ISOName
from api.util import load_yaml_data, simple_cache


def get_network_device_energy_intensity_mapping(config_path: os.path) -> dict[str, float]:
    """Load the carbon intensity per fuel source map from config."""
    # Load device to power mapping from yaml config
    yaml_data = load_yaml_data(config_path)
    energy_intensity_map_name = 'network_device_energy_intensity'
    assert yaml_data is not None and energy_intensity_map_name in yaml_data, \
        f'Failed to load {energy_intensity_map_name} from config.'
    return yaml_data[energy_intensity_map_name]

MAP_DEVICE_ENERGY_INTENSITY_BY_DEVICE_TYPE = get_network_device_energy_intensity_mapping(
    os.path.join(Path(__file__).parent.absolute(), 'network_device.yaml'))

# Define an Enum for device types
class NetworkDeviceType(str, Enum):
    ROUTER = "Router"
    WDM_SWITCH = "WDM Switch"
    TRANSPONDER = "Transponder"
    MUXPONDER = "Muxponder"
    AMPLIFIER = "Amplifier"
    REGENERATOR = "Regenerator"

    def __str__(self) -> str:
        return self.value

@dataclass(unsafe_hash=True)
class NetworkDevice:
    """This represent a network device, and the source of power for the device."""
    device_type: NetworkDeviceType
    gps: Coordinate
    # This is the distance from the device to the power source, used for submarine fiber device power loss calculation.
    #   It is 0 for land fiber devices.
    distance_from_route_start_km: float
    power_source_distance_km: float = 0.0
    iso: ISOName = None

    def __str__(self) -> str:
        return f'{self.device_type} at {self.gps} ({self.iso})'

    def get_energy_intensity_w_per_gbps(self) -> float:
        """Get the energy intensity in W/Gbps for this device."""
        if self.device_type not in MAP_DEVICE_ENERGY_INTENSITY_BY_DEVICE_TYPE:
            raise ValueError(f'Unknown energy intensity for device type {self.device_type}')
        energy_intensity = MAP_DEVICE_ENERGY_INTENSITY_BY_DEVICE_TYPE[self.device_type]

        # Account for power loss for non-zero distance from power source
        TRANSMISSION_LOSS_PER_100KM = 0.035     # From "HVDC Submarine Power Cables in the World"
        # Exponential decay, per 100km the device receives 3.5% less power
        power_ratio_received = pow(1 - TRANSMISSION_LOSS_PER_100KM, self.power_source_distance_km / 100.0)
        return energy_intensity / power_ratio_received

def create_network_devices_along_path(mls: MultiLineString,
                                      network_device_type: NetworkDeviceType,
                                      interval_km: float,
                                      fiber_types: list[str],
                                      router_hop_distances_out: list[float]) -> \
                                        list[NetworkDevice]:
    """Create a list of network devices along a route, every interval_km.

    Args:
        mls: MultiLineString representing the route, or a list of paths (linestrings) between routers.
        network_device_type: Type of network device to insert.
        interval_km: Distance between each device.
        fiber_types: List of fiber types for each path of mls, either "land" or "submarine".
        router_hop_distances_out: List to assign the total distance of each path to.

    Returns:
        List of inserted network devices.
    """
    def _interpolate_point(path, start, end, distance_km, total_distance_km):
        # Function to interpolate a point at a specific distance between two points.
        fraction = distance_km / total_distance_km
        return path.interpolate(path.project(Point(start)) + fraction * path.project(Point(end)))

    all_network_devices = []

    path: LineString
    total_distance_covered = 0.
    cumulative_path_distance = 0.
    for i in range(len(mls.geoms)):
        path = mls.geoms[i]
        network_devices = []
        fiber_type = fiber_types[i]
        current_point = path.coords[0]
        distance_from_path_start = 0.
        total_path_distance_km = 0.

        # Iterate through each pair of points in the line string
        for next_point in path.coords[1:]:
            segment_distance_km = distance(lonlat(*current_point),
                                           lonlat(*next_point)).km
            total_path_distance_km += segment_distance_km

            # Add points every interval along the segment
            while total_distance_covered + interval_km < segment_distance_km:
                total_distance_covered += interval_km
                distance_from_path_start += interval_km
                point = _interpolate_point(path, current_point, next_point, total_distance_covered, segment_distance_km)
                if fiber_type == 'submarine':
                    # Assume insertion at the start, but calculate distance to power source to adjust later.
                    insert_point = path.coords[0]
                    power_source_distance_km = distance_from_path_start
                else:
                    # If the fiber is land, insert the device as is. We assume power is sourced locally.
                    insert_point = point.coords[0]
                    power_source_distance_km = 0.
                gps: Coordinate = (insert_point[1], insert_point[0])  # (lon, lat) -> (lat, lon)
                distance_from_route_start = cumulative_path_distance + distance_from_path_start
                network_devices.append(NetworkDevice(network_device_type, gps,
                                                     distance_from_route_start,
                                                     power_source_distance_km))

            # Reset the distance counter after each segment
            total_distance_covered -= segment_distance_km
            current_point = next_point

        # Adjust submarine fiber device insertion points if it's closer to the end than the start.
        if fiber_type == 'submarine':
            for network_device in network_devices:
                if network_device.power_source_distance_km > total_path_distance_km / 2:
                    network_device.lat, network_device.lon = path.coords[-1]
                    network_device.power_source_distance_km = total_path_distance_km - network_device.power_source_distance_km

        all_network_devices += network_devices
        router_hop_distances_out[i] = total_path_distance_km
        cumulative_path_distance += total_path_distance_km

    return all_network_devices


@simple_cache.memoize(timeout=0)
def create_network_devices(router_latlons: list[Coordinate], fiber_wkt_paths: str, fiber_types: list[str]) -> \
        list[NetworkDevice]:
    """Create an ordered list of network devices given the router locations and the fiber paths among them.

    Args:
        router_latlons: List of router locations, in (lat, lon) format.
        fiber_wkt_paths: WKT representation of the fiber paths, in MULTILINESTRING((lon lat)) format.
        fiber_types: List of fiber types, either "land" or "submarine".

    Returns:
        List of network devices, in the order they appear along the path.
    """
    if len(router_latlons) == 0:
        return []

    try:
        # If no fiber paths are provided, create a simple network with only routers and router-attached devices.
        if not fiber_wkt_paths or not fiber_types:
            network_devices = []
            distance_from_route_start_km = 0.
            last_coordinate = router_latlons[0]
            for gps in router_latlons:
                # Already in (lat, lon) format
                distance_from_route_start_km += distance(last_coordinate, gps).km
                last_coordinate = gps
                network_devices.append(NetworkDevice(NetworkDeviceType.ROUTER, gps, distance_from_route_start_km))
                network_devices.append(NetworkDevice(NetworkDeviceType.WDM_SWITCH, gps, distance_from_route_start_km))
                network_devices.append(NetworkDevice(NetworkDeviceType.WDM_SWITCH, gps, distance_from_route_start_km))
                network_devices.append(NetworkDevice(NetworkDeviceType.TRANSPONDER, gps, distance_from_route_start_km))
                network_devices.append(NetworkDevice(NetworkDeviceType.MUXPONDER, gps, distance_from_route_start_km))
            return network_devices

        mls: MultiLineString = loads(fiber_wkt_paths)
        assert len(router_latlons) == len(mls.geoms) + 1, \
            f'Number of routers ({len(router_latlons)}) must be 1 + length of multi line string ({len(mls.geoms)})'
        assert len(mls.geoms) == len(fiber_types), \
            f'Number of fiber paths ({len(mls.geoms)}) must be equal to number of fiber types ({len(fiber_types)})'
        assert all(fiber_type in ['land', 'submarine'] for fiber_type in fiber_types), \
            f'Fiber type must be either "land" or "submarine"'

        network_devices = []

        # Add amplifiers and regenerators along the path, every 80km and 1500km respectively.
        # Also calculate the distance between each consecutive routers.
        router_hop_distances = [0.] * (len(router_latlons) - 1)
        network_devices += create_network_devices_along_path(mls, NetworkDeviceType.AMPLIFIER, 80,
                                                             fiber_types, router_hop_distances)
        network_devices += create_network_devices_along_path(mls, NetworkDeviceType.REGENERATOR, 1500,
                                                             fiber_types, router_hop_distances)

        # At each router's location, add 2x WDM switches, 1x transponder, 1x muxponder.
        distance_from_route_start_km = 0.
        for i in range(len(router_latlons)):
            gps = router_latlons[i] # Already in (lat, lon) format
            network_devices.append(NetworkDevice(NetworkDeviceType.ROUTER, gps, distance_from_route_start_km))
            network_devices.append(NetworkDevice(NetworkDeviceType.WDM_SWITCH, gps, distance_from_route_start_km))
            network_devices.append(NetworkDevice(NetworkDeviceType.WDM_SWITCH, gps, distance_from_route_start_km))
            network_devices.append(NetworkDevice(NetworkDeviceType.TRANSPONDER, gps, distance_from_route_start_km))
            network_devices.append(NetworkDevice(NetworkDeviceType.MUXPONDER, gps, distance_from_route_start_km))
            if i < len(router_latlons) - 1:
                distance_from_route_start_km += router_hop_distances[i]

        return sorted(network_devices, key=lambda d: d.distance_from_route_start_km)
    except Exception as ex:
        current_app.logger.error(f'Error creating network devices: {ex}')
        current_app.logger.error(traceback.format_exc())
        raise ValueError(f'Error creating network devices: {ex}')
