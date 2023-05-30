#!/usr/bin/env python3

# Source: https://github.com/electricitymap/electricitymap-contrib/blob/master/parsers/US_PJM.py

"""Parser for the PJM area of the United States."""

import json
import re

import arrow
import demjson3 as demjson
import requests
from bs4 import BeautifulSoup
from dateutil import parser, tz

# Used for both production and price data.
url = 'http://www.pjm.com/markets-and-operations.aspx'

mapping = {
    'Coal': 'coal',
    'Gas': 'gas',
    'Hydro': 'hydro',
    'Multiple Fuels': 'unknown',
    'Nuclear': 'nuclear',
    'Oil': 'oil',
    'Other': 'unknown',
    'Other Renewables': 'unknown-renewables',
    'Solar': 'solar',
    'Wind': 'wind',
    'Storage': 'battery',
}

def extract_data(session=None) -> tuple:
    """
    Makes a request to the PJM data url.
    Finds timestamp of current data and converts into a useful form.
    Finds generation data inside script tag.

    :return: tuple of generation data and datetime.
    """

    s = session or requests.Session()
    req = requests.get(url)
    soup = BeautifulSoup(req.content, 'html.parser')

    try:
        time_div = soup.find("div", id="asOfDate").text
    except AttributeError:
        raise LookupError('No data is available for US-PJM.')

    time_pattern = re.compile(r"""(\d{1,2}     #Hour can be 1/2 digits.
                                   :           #Separator.
                                   \d{2})\s    #Minutes must be 2 digits with a space after.
                                   (a.m.|p.m.) #Either am or pm allowed.""", re.X)

    latest_time = re.search(time_pattern, time_div)

    time_data = latest_time.group(1).split(":")
    am_or_pm = latest_time.group(2)
    hour = int(time_data[0])
    minute = int(time_data[1])

    # Time format used by PJM is slightly unusual and needs to be converted so arrow can use it.
    if am_or_pm == "p.m." and hour != 12:
        # Time needs to be in 24hr format
        hour += 12
    elif am_or_pm == "a.m." and hour == 12:
        # Midnight is 12 a.m.
        hour = 0

    arr_dt = arrow.now('America/New_York').replace(hour=hour, minute=minute)
    future_check = arrow.now('America/New_York')

    if arr_dt > future_check:
        # Generation mix lags 1-2hrs behind present.
        # This check prevents data near midnight being given the wrong date.
        arr_dt = arr_dt.shift(days=-1)

    dt = arr_dt.floor('minute').datetime

    generation_mix_div = soup.find("div", id="rtschartallfuelspjmGenFuelM_container")
    generation_mix_script = generation_mix_div.next_sibling

    pattern = r'series: \[(.*)\]'
    script_data = re.search(pattern, str(generation_mix_script)).group(1)

    # demjson is required because script data is javascript not valid json.
    raw_data = demjson.decode(script_data)
    data = raw_data["data"]

    return data, dt


def data_processer(data) -> dict:
    """Takes a list of dictionaries and extracts generation type and value from each."""

    production = {}
    for point in data:
        gen_type = mapping[point['name']]
        gen_value = float(point['y'])
        production[gen_type] = production.get(gen_type, 0.0) + gen_value

    return production


def fetch_production(zone_key='US-PJM', session=None, target_datetime=None, logger=None) -> dict:
    """Requests the last known production mix (in MW) of a given country."""
    if target_datetime is not None:
        raise NotImplementedError('This parser is not yet able to parse past dates')

    extracted = extract_data(session=None)
    production = data_processer(extracted[0])

    datapoint = {
        'zoneKey': zone_key,
        'datetime': extracted[1],
        'production': production,
        'storage': {'hydro': None, 'battery': None},
        'source': 'pjm.com'
    }

    return datapoint

if __name__ == '__main__':
    print('fetch_production() ->')
    print(fetch_production())
