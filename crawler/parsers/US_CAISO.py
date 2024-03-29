#!/usr/bin/env python3

# Source: https://github.com/electricitymap/electricitymap-contrib/blob/master/parsers/US_CA.py

from datetime import datetime, timedelta
import io
import sys
import arrow
import pandas
import requests
from bs4 import BeautifulSoup
from collections import defaultdict
import logging

# CAISO_PROXY = 'https://us-ca-proxy-jfnx5klx2a-uw.a.run.app'
# FUEL_SOURCE_CSV = f'{CAISO_PROXY}/outlook/SP/fuelsource.csv'

def fetch_production(zone_key='US-CAISO', session=None, target_datetime=None,
                     logger: logging.Logger = logging.getLogger(__name__)) -> list:
    """Requests the last known production mix (in MW) of a given country."""
    target_datetime = arrow.get(target_datetime) if target_datetime else arrow.get()
    target_datetime = target_datetime.to('US/Pacific')
    target_date = target_datetime.strftime('%Y%m%d')

    # Get the production from the CSV
    url = f'http://www.caiso.com/outlook/SP/History/{target_date}/fuelsource.csv'
    response = requests.get(url)
    response.raise_for_status()
    # NOTE: temporary check, as CAISO seems to return 404 HTML page with a 200 status code
    if '404 - Page Not Found' in response.text:
        raise ValueError('404 response with a success status code')
    csv = pandas.read_csv(io.StringIO(response.text))
    csv.columns = csv.columns.str.lower()
    latest_index = len(csv) - 1
    production_map = {
        'Solar': 'solar',
        'Wind': 'wind',
        'Geothermal': 'geothermal',
        'Biomass': 'biomass',
        'Biogas': 'biomass',
        'Small hydro': 'hydro',
        'Coal': 'coal',
        'Nuclear': 'nuclear',
        'Natural Gas': 'gas',
        'Large Hydro': 'hydro',
        'Batteries': 'battery',
        'Imports': 'import',
        'Other': 'unknown',
    }
    storage_map = {
        'Batteries': 'battery'
    }
    daily_data = []
    for i in range(0, latest_index + 1):
        h, m = map(int, csv['time'][i].split(':'))
        date = target_datetime.replace(hour=h, minute=m,
                                                       second=0, microsecond=0)
        data = {
            'zoneKey': zone_key,
            'production': defaultdict(float),
            'storage': defaultdict(float),
            'source': 'caiso.com',
            'datetime': date.datetime
        }

        # map items from names in CAISO CSV to names used in Electricity Map
        for ca_gen_type, mapped_gen_type in production_map.items():
            production = float(csv[ca_gen_type.lower()][i])
            
            if production < 0 and (mapped_gen_type == 'solar' or mapped_gen_type == 'nuclear'):
                # logger.warn(ca_gen_type + ' production for US_CA was reported as less than 0 and was clamped')
                production = 0.0
            
            # if another mean of production created a value, sum them up
            data['production'][mapped_gen_type] += production

        for ca_storage_type, mapped_storage_type in storage_map.items():
            storage = -float(csv[ca_storage_type.lower()][i])

            # if another mean of storage created a value, sum them up
            data['storage'][mapped_storage_type] += storage

        daily_data.append(data)

    return daily_data


if __name__ == '__main__':
    "Main method, not used by Electricity Map backend, but handy for testing"

    from pprint import pprint

    date_delta = -1
    print(f'fetch_production({date_delta}) ->')
    pprint(fetch_production(target_datetime=arrow.now().shift(days=date_delta)))
