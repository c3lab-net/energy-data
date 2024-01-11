#!/usr/bin/env python3

import time
import traceback
from flask import Flask, g, current_app, jsonify
from flask_restful import Api
import webargs
import secrets
import logging
from werkzeug.exceptions import UnprocessableEntity, HTTPException

from api.util import DocstringDefaultException, CustomJSONEncoder, simple_cache, carbon_data_cache


class CustomApi(Api):
    def handle_error(self, e: Exception):
        handled_exceptions = [UnprocessableEntity]
        if any([isinstance(e, handled_ex) for handled_ex in handled_exceptions]):
            return super().handle_error(e)

        # Skip re-thrown wrapped exceptions
        if not isinstance(e, DocstringDefaultException):
            current_app.logger.error("%s: %s", type(e), e)
            current_app.logger.error(traceback.format_exc())
        status_code = e.code if isinstance(e, HTTPException) else 500
        return jsonify({'error': str(e)}), status_code


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = secrets.token_hex()
    app.config['RESTFUL_JSON'] = {
        'cls': CustomJSONEncoder
    }
    simple_cache.init_app(app)
    carbon_data_cache.init_app(app)
    if __name__ != '__main__':
        # Source: https://trstringer.com/logging-flask-gunicorn-the-manageable-way/
        gunicorn_logger = logging.getLogger('gunicorn.error')
        app.logger.handlers = gunicorn_logger.handlers
        app.logger.setLevel(gunicorn_logger.level)

    from api.routes.balancing_authority import BalancingAuthority
    from api.routes.carbon_intensity import CarbonIntensity
    from api.routes.carbon_aware_scheduler import CarbonAwareScheduler
    from api.routes.energy_mixture import EnergyMixture
    from api.routes.metadata import Metadata_CloudLocation_Providers, Metadata_CloudLocation_Locations

    # Alternatively, use this and `from varname import nameof`.
    errors_custom_responses = {
        # nameof("PSqlExecuteException"): {
        #     'message': 'An unknown database exception has occurred',
        #     'status': 500
        # }
    }

    api = CustomApi(app, errors=errors_custom_responses)
    api.add_resource(BalancingAuthority, '/balancing-authority/')
    api.add_resource(CarbonIntensity, '/carbon-intensity/')
    api.add_resource(CarbonAwareScheduler, '/carbon-aware-scheduler/')
    api.add_resource(EnergyMixture, '/energy-mixture/')

    api.add_resource(Metadata_CloudLocation_Providers, '/metadata/cloud-location/providers/')
    api.add_resource(Metadata_CloudLocation_Locations, '/metadata/cloud-location/locations/<string:provider>/')

    # Source: https://github.com/marshmallow-code/webargs/issues/181#issuecomment-621159812
    @webargs.flaskparser.parser.error_handler
    def webargs_validation_handler(error, req, schema, *, error_status_code, error_headers):
        """Handles errors during parsing. Aborts the current HTTP request and
        responds with a 422 error.
        """
        status_code = error_status_code or webargs.core.DEFAULT_VALIDATION_STATUS
        webargs.flaskparser.abort(
            status_code,
            exc=error,
            messages=error.messages,
        )

    @app.before_request
    def before_request():
        g.start = time.time()

    @app.teardown_request
    def teardown_request(exception=None):
        diff = time.time() - g.start
        app.logger.debug(f'Request took {1000 * diff:.0f}ms')

    return app
