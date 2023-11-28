#!/usr/bin/env python3

from enum import Enum
from dataclasses import dataclass

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
    device_type: NetworkDeviceType
    lat: float
    lon: float

    # TODO: not working in print()
    def __str__(self) -> str:
        return f'{self.device_type} at ({self.lat}, {self.lon})'

def create_network_devices_along_path(mls: MultiLineString,
                                      network_device_type: NetworkDeviceType,
                                      interval_km: float) -> \
                                        list[NetworkDevice]:
    # Function to interpolate a point at a specific distance
    def _interpolate_point(path, start, end, distance_km, total_distance_km):
        fraction = distance_km / total_distance_km
        return path.interpolate(path.project(Point(start)) + fraction * path.project(Point(end)))

    network_devices = []

    path: LineString
    total_distance_covered = 0.
    for path in mls.geoms:
        current_point = path.coords[0]

        # Iterate through each pair of points in the line string
        for next_point in path.coords[1:]:
            segment_distance_km = distance(lonlat(*current_point),
                                           lonlat(*next_point)).km

            # Add points every interval along the segment
            while total_distance_covered + interval_km < segment_distance_km:
                total_distance_covered += interval_km
                point = _interpolate_point(path, current_point, next_point, total_distance_covered, segment_distance_km)
                insert_point = point.coords[0]
                network_devices.append(NetworkDevice(network_device_type, insert_point[1], insert_point[0]))

            # Reset the distance counter after each segment
            total_distance_covered -= segment_distance_km
            current_point = next_point

    return network_devices


# Modify the create_network_devices function to insert amplifiers and regenerators
def create_network_devices(router_latlons: list[Coordinate], wkt_paths_mls: str):
    mls: MultiLineString = loads(wkt_paths_mls)
    assert len(router_latlons) == len(mls.geoms), \
        f'Number of routers ({len(router_latlons)}) must be one more than length of multi line string ({len(mls.geoms)})'

    network_devices = []

    # At each router's location, add 2x WDM switches, 1x transponder, 1x muxponder.
    for lat, lon in router_latlons:
        network_devices.append(NetworkDevice(NetworkDeviceType.ROUTER, lat, lon))
        network_devices.append(NetworkDevice(NetworkDeviceType.WDM_SWITCH, lat, lon))
        network_devices.append(NetworkDevice(NetworkDeviceType.WDM_SWITCH, lat, lon))
        network_devices.append(NetworkDevice(NetworkDeviceType.TRANSPONDER, lat, lon))
        network_devices.append(NetworkDevice(NetworkDeviceType.MUXPONDER, lat, lon))

    # Add amplifiers and regenerators along the path, every 80km and 1500km respectively
    network_devices += create_network_devices_along_path(mls, NetworkDeviceType.AMPLIFIER, 80)
    network_devices += create_network_devices_along_path(mls, NetworkDeviceType.REGENERATOR, 1500)

    return network_devices
