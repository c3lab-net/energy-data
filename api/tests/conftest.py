#!/usr/bin/env python3

# Source: https://flask.palletsprojects.com/en/2.1.x/testing/

import pytest
from api import create_app
from api.util import CustomJSONEncoder


@pytest.fixture()
def app():
    app = create_app()
    app.config.update({
        "TESTING": True,
    })
    app.json_encoder = CustomJSONEncoder
    yield app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def runner(app):
    return app.test_cli_runner()


@pytest.fixture()
def get_gps_coordinate():
    m_test_location_to_gps_coordinate: dict[str, tuple[float, float]] = {
        'UCSD': (32.8801, -117.2340),
        'Milwaukee, WI': (43.0389, -87.9065),
        'Columbus, OH': (39.9833, -82.9833),
        'Ashburn, VA': (39.0438, -77.4874),
        'Boardman, OR': (45.8399, -119.7006),
        'Eugene, OR': (44.0521, -123.0868),
        'Austin, TX': (30.2672, -97.7431),
    }

    def _method_impl(location_name) -> tuple[float, float]:
        assert location_name in m_test_location_to_gps_coordinate, f"Unknown test location {location_name}"
        return m_test_location_to_gps_coordinate[location_name]

    return _method_impl
