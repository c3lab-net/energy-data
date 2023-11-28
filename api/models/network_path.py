#!/usr/bin/env python3

from enum import Enum
from dataclasses import dataclass
import traceback
from flask import current_app

from shapely.wkt import loads
from geopy.distance import distance, lonlat
from shapely.geometry import Point, LineString, MultiLineString

from api.models.common import Coordinate

# Define an Enum for device types
class NetworkDeviceType(str, Enum):
    ROUTER = "Router"
    WDM_SWITCH = "WDM Switch"
    TRANSPONDER = "Transponder"
    MUXPONDER = "Muxponder"
    AMPLIFIER = "Amplifier"
    REGENERATOR = "Regenerator"

@dataclass
class NetworkDevice:
    """This represent a network device, and the source of power for the device."""
    device_type: NetworkDeviceType
    lat: float
    lon: float
    # This is the distance from the device to the power source, used for submarine fiber device power loss calculation.
    #   It is 0 for land fiber devices.
    power_source_distance_km: float = 0.0

    # TODO: not working in print()
    def __str__(self) -> str:
        return f'{self.device_type} at ({self.lat}, {self.lon})'

def create_network_devices_along_path(mls: MultiLineString,
                                      network_device_type: NetworkDeviceType,
                                      interval_km: float,
                                      fiber_types: list[str]) -> \
                                        list[NetworkDevice]:
    """Create a list of network devices along a path, every interval_km.

    Args:
        mls: MultiLineString representing the path.
        network_device_type: Type of network device to insert.
        interval_km: Distance between each device.
        fiber_types: List of fiber types for each path of mls, either "land" or "submarine".

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
                point = _interpolate_point(path, current_point, next_point, total_distance_covered, segment_distance_km)
                if fiber_type == 'submarine':
                    # Assume insertion at the start, but calculate distance to power source to adjust later.
                    insert_point = path.coords[0]
                    distance_from_path_start += interval_km
                    power_source_distance_km = distance_from_path_start
                else:
                    # If the fiber is land, insert the device as is. We assume power is sourced locally.
                    insert_point = point.coords[0]
                    power_source_distance_km = 0.
                network_devices.append(NetworkDevice(network_device_type, insert_point[1], insert_point[0],
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

    return all_network_devices


def create_network_devices(router_latlons: list[Coordinate], fiber_wkt_paths: str, fiber_types: list[str]) -> \
        list[NetworkDevice]:
    """Create a list of network devices given the router locations and the fiber paths among them.

    Args:
        router_latlons: List of router locations, in (lat, lon) format.
        fiber_wkt_paths: WKT representation of the fiber paths, in MULTILINESTRING((lon lat)) format.
        fiber_types: List of fiber types, either "land" or "submarine".

    Returns:
        List of network devices, NOT necessarily in the order they appear along the path.
    """
    try:
        mls: MultiLineString = loads(fiber_wkt_paths)
        assert len(router_latlons) == len(mls.geoms) + 1, \
            f'Number of routers ({len(router_latlons)}) must be 1 + length of multi line string ({len(mls.geoms)})'
        assert len(mls.geoms) == len(fiber_types), \
            f'Number of fiber paths ({len(mls.geoms)}) must be equal to number of fiber types ({len(fiber_types)})'
        assert all(fiber_type in ['land', 'submarine'] for fiber_type in fiber_types), \
            f'Fiber type must be either "land" or "submarine"'
        for i in range(len(fiber_types)):
            if fiber_types[i] == 'submarine':
                if i > 0:
                    assert fiber_types[i - 1] != 'submarine', 'Submarine fiber must be surrounded by land fiber'
                if i < len(fiber_types) - 1:
                    assert fiber_types[i + 1] != 'submarine', 'Submarine fiber must be surrounded by land fiber'

        network_devices = []

        # At each router's location, add 2x WDM switches, 1x transponder, 1x muxponder.
        for lat, lon in router_latlons:
            network_devices.append(NetworkDevice(NetworkDeviceType.ROUTER, lat, lon))
            network_devices.append(NetworkDevice(NetworkDeviceType.WDM_SWITCH, lat, lon))
            network_devices.append(NetworkDevice(NetworkDeviceType.WDM_SWITCH, lat, lon))
            network_devices.append(NetworkDevice(NetworkDeviceType.TRANSPONDER, lat, lon))
            network_devices.append(NetworkDevice(NetworkDeviceType.MUXPONDER, lat, lon))

        # Add amplifiers and regenerators along the path, every 80km and 1500km respectively
        network_devices += create_network_devices_along_path(mls, NetworkDeviceType.AMPLIFIER, 80, fiber_types)
        network_devices += create_network_devices_along_path(mls, NetworkDeviceType.REGENERATOR, 1500, fiber_types)

        return network_devices
    except Exception as ex:
        current_app.logger.error(f'Error creating network devices: {ex}')
        current_app.logger.error(traceback.format_exc())
        raise ValueError(f'Error creating network devices: {ex}')
