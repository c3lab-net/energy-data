#!/usr/bin/env python3

from datetime import datetime
from flask import current_app
from flask_restful import Resource
from webargs.flaskparser import use_args
from typing import Optional
import marshmallow_dataclass
from marshmallow import validates_schema, ValidationError

from api.helpers.balancing_authority import get_iso_from_gps
from api.helpers.carbon_intensity import get_carbon_intensity_list
from api.models.common import ISO_PREFIX_C3LAB, CarbonDataSource, IsoFormat, get_iso_format_for_carbon_source
from api.models.dataclass_extensions import *


@marshmallow_dataclass.dataclass
class CarbonIntensityRequest:
    start: datetime = field_default()
    end: datetime = field_default()

    carbon_data_source: CarbonDataSource = field_enum(CarbonDataSource, CarbonDataSource.C3Lab)
    use_prediction: bool = field(default=False)
    desired_renewable_ratio: Optional[float] = \
        optional_field_with_validation(lambda ratio: 0. <= ratio <= 1.)

    latitude: Optional[float] = optional_field_with_validation(lambda x: abs(x) <= 90.)
    longitude: Optional[float] = optional_field_with_validation(lambda x: abs(x) <= 180.)
    iso: Optional[str] = optional_field_with_validation(lambda x: ':' in x)

    @validates_schema
    def validate_schema(self, data, **kwargs):
        errors = dict()
        iso = data.get('iso', None)
        if iso is None and (data.get('latitude', None) is None or data.get('longitude', None) is None):
            errors['iso'] = 'iso must be specified if latitude and longitude are not specified'
        if errors:
            raise ValidationError(errors)

class CarbonIntensity(Resource):
    @use_args(marshmallow_dataclass.class_schema(CarbonIntensityRequest)(), location='query')
    def get(self, request: CarbonIntensityRequest):
        orig_request = { 'request': request }
        current_app.logger.info("CarbonIntensity.get(%s)" % request)

        iso_format = get_iso_format_for_carbon_source(request.carbon_data_source)
        if request.iso is not None:
            iso = request.iso
            region = None
        else:
            iso = get_iso_from_gps(request.latitude, request.longitude, iso_format)
            region = get_iso_from_gps(request.latitude, request.longitude, IsoFormat.C3Lab).removeprefix(ISO_PREFIX_C3LAB)
        l_carbon_intensity = get_carbon_intensity_list(iso, request.start, request.end,
                                                       request.carbon_data_source, request.use_prediction,
                                                       request.desired_renewable_ratio)

        return orig_request | {
            'region': region,
            'iso': iso,
            'carbon_intensities': l_carbon_intensity,
        }
