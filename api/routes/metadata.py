#!/usr/bin/env python3

from flask_restful import Resource
from webargs.flaskparser import use_kwargs

from api.models.cloud_location import CloudLocationManager
from api.models.dataclass_extensions import *

g_cloud_manager = CloudLocationManager()


class Metadata_CloudLocation_Providers(Resource):
    def get(self):
        """Get all cloud providers."""
        return g_cloud_manager.get_all_cloud_providers()


class Metadata_CloudLocation_Locations(Resource):
    def get(self, provider: str):
        """Get all cloud locations for a provider."""
        all_cloud_provider = g_cloud_manager.get_all_cloud_providers()
        if provider not in all_cloud_provider:
            return f'Provider {provider} not found', 404
        return g_cloud_manager.get_cloud_region_codes(provider)
