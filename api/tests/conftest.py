#!/usr/bin/env python3

# Source: https://flask.palletsprojects.com/en/2.1.x/testing/

import pytest
from api import create_app

import logging

@pytest.fixture
def logger():
    return logging.getLogger(__name__)

@pytest.fixture()
def app():
    app = create_app()
    app.config.update({
        "TESTING": True,
    })
    yield app

@pytest.fixture()
def client(app):
    return app.test_client()

@pytest.fixture()
def runner(app):
    return app.test_cli_runner()
