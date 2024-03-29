#!usr/bin/env python3

# Source: https://github.com/electricitymap/electricitymap-contrib/blob/master/parsers/US_SPP.py

"""Parser for the Southwest Power Pool area of the United States."""

from dateutil import parser, tz
from io import StringIO
from logging import getLogger
from pandas.tseries.offsets import DateOffset
import datetime
import pandas as pd
import requests
import urllib3
from urllib3.util.ssl_ import create_urllib3_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class CustomHTTPAdapter(requests.adapters.HTTPAdapter):
    """
    A TransportAdapter that get around DH_KEY_TOO_SMALL in Requests.
    """
    CIPHERS = "HIGH:!DH:!aNULL"

    def init_poolmanager(self, *args, **kwargs):
        context = create_urllib3_context(ciphers=CustomHTTPAdapter.CIPHERS)
        kwargs['ssl_context'] = context
        return super(CustomHTTPAdapter, self).init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, *args, **kwargs):
        context = create_urllib3_context(ciphers=CustomHTTPAdapter.CIPHERS)
        kwargs['ssl_context'] = context
        return super(CustomHTTPAdapter, self).proxy_manager_for(*args, **kwargs)

HISTORIC_GENERATION_BASE_URL = 'https://portal.spp.org/file-browser-api/download/generation-mix-historical?path=/'
# HISTORIC_GENERATION_BASE_URL = 'https://portal.spp.org/file-browser-api/download/hourly-generation-capacity-by-fuel-type?path='

GENERATION_URL = 'https://portal.spp.org/file-browser-api/download/generation-mix-historical?path=%2FGenMix2Hour.csv'

# GENERATION_URL = 'https://portal.spp.org/file-browser-api/download/hourly-generation-capacity-by-fuel-type?path=/HRLY-GEN-CAP-BY-FUEL-TYPE-LATEST-INTERVAL.csv'

MAPPING = {'Wind': 'wind',
           'Nuclear': 'nuclear',
           'Hydro': 'hydro',
           'Solar': 'solar',
           'Natural Gas': 'gas',
           'Diesel Fuel Oil': 'oil',
           'Waste Disposal Services': 'biomass',
           'Coal': 'coal'
            }

TIE_MAPPING = {'US-MISO->US-SPP': ['AMRN', 'DPC', 'GRE', 'MDU', 'MEC', 'NSP', 'OTP']}

TIMESTAMP_COLUMN = 'GMT MKT Interval'

# NOTE
# Data sources return timestamps in GMT.
# Energy storage situation unclear as of 16/03/2018, likely to change quickly in future.


def get_data(url, session=None):
    """Returns a pandas dataframe."""

    s=session or requests.Session()
    # s.mount("https://marketplace.spp.org", CustomHTTPAdapter())
    # req = s.get(url, verify=False)
    req = s.get(url)
    df = pd.read_csv(StringIO(req.text))

    return df


def data_processor(df, logger) -> list:
    """
    Takes a dataframe and logging instance as input.
    Checks for new generation types and logs a warning if any are found.
    Parses the dataframe row by row removing unneeded keys.

    :return: list of tuples containing a datetime object and production dictionary.
    """

    # Remove leading whitespace in column headers.
    df = df.dropna(thresh=2)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={'Gas Self': 'Natural Gas Self'}) #Fix naming error which otherwise misclassifies Gas Self as Unknown
    #Some historical csvs split the production into 'Market' and 'Self',
    #So first we need to combine those.
    for col in df.columns:
        if 'Market' in col:
            combined_col = col.replace('Market','').strip()
            self_col = col.replace('Market','Self')
            if self_col in df.columns:
                df[combined_col] = df[col] + df[self_col]
                df.drop(self_col, inplace=True, axis=1)
            else:
                logger.warning(f'Corresponding column "{self_col}" to "{col}" not found in file', extra={'key':'US-SPP'})
                df[combined_col] = df[col]

            df.drop(col, inplace=True, axis=1)

    keys_to_remove = {TIMESTAMP_COLUMN, 'Average Actual Load', 'Load'}

    # Check for new generation columns.
    known_keys = MAPPING.keys() | keys_to_remove
    column_headers = set(df.columns)

    unknown_keys = column_headers - known_keys

    for heading in unknown_keys:
        if heading not in ['Other','Waste Heat']:
            logger.warning('New column \'{}\' present in US-SPP data source.'.format(
                heading), extra={'key': 'US-SPP'})

    keys_to_remove = keys_to_remove | unknown_keys

    processed_data = []
    for index, row in df.iterrows():
        production = row.to_dict()
        production['unknown'] = sum([production[k] for k in unknown_keys])

        dt_aware = production[TIMESTAMP_COLUMN].to_pydatetime()
        for k in keys_to_remove:
            production.pop(k, None)

        mapped_production = {MAPPING.get(k,k):v for k,v in production.items()}

        processed_data.append((dt_aware, mapped_production))
    return processed_data


def fetch_production(zone_key = 'US-SPP', session=None, target_datetime:datetime.datetime=None, logger=getLogger(__name__)) -> dict:
    """Requests the last known production mix (in MW) of a given zone."""

    if target_datetime is not None:
        current_year = datetime.datetime.now().year
        target_year = target_datetime.year

        #Check if datetime is too far in the past
        if target_year < 2011:
            raise NotImplementedError('Data before 2011 not available from this source')

        #Check if datetime in current year, or past year
        if target_year == current_year:
            filename = 'GenMixYTD.csv'
        else:
            filename = f'GenMix_{target_year}.csv'

        # # E.g. '/2022/01/HRLY-GEN-CAP-BY-FUEL-TYPE-20220101.csv'
        # filepath = '/%s/%s/HRLY-GEN-CAP-BY-FUEL-TYPE-%s.csv' % (
        #     target_datetime.strftime('%Y'),
        #     target_datetime.strftime('%m'),
        #     target_datetime.strftime('%Y%m%d')
        # )
        # historic_generation_url = HISTORIC_GENERATION_BASE_URL + filepath

        historic_generation_url = HISTORIC_GENERATION_BASE_URL + filename
        raw_data = get_data(historic_generation_url, session=session)
        #In some cases the timeseries column is named differently, so we standardize it
        raw_data.rename(columns={'GMTTime': TIMESTAMP_COLUMN},inplace=True)

        raw_data[TIMESTAMP_COLUMN] = pd.to_datetime(raw_data[TIMESTAMP_COLUMN])
        end = target_datetime
        start = target_datetime - datetime.timedelta(days=1)
        start = max(start, raw_data[TIMESTAMP_COLUMN].min())
        raw_data = raw_data[(raw_data[TIMESTAMP_COLUMN] >= start)&(raw_data[TIMESTAMP_COLUMN]<= end)]
    else:
        raw_data = get_data(GENERATION_URL, session=session)
        raw_data[TIMESTAMP_COLUMN] = pd.to_datetime(raw_data[TIMESTAMP_COLUMN])

    processed_data = data_processor(raw_data, logger)

    data = []
    for item in processed_data:
        dt = item[0].replace(tzinfo=tz.gettz('Etc/GMT'))
        datapoint = {
          'zoneKey': zone_key,
          'datetime': dt,
          'production': item[1],
          'storage': {},
          'source': 'spp.org'
        }
        data.append(datapoint)

    return data


if __name__ == '__main__':
    print('fetch_production() -> ')
    print(fetch_production())
    # print(fetch_production(target_datetime=datetime.datetime(2023, 1, 1, tzinfo=tz.gettz('America/Chicago'))))
