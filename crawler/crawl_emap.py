#!/usr/bin/env python3

import argparse
import configparser
import logging
import os
import random
import sys
import traceback
from time import sleep
from typing import Callable

import requests
from crawl import get_db_connection

CONFIG_FILE = "electricitymap.ini"

class InfoFilter(logging.Filter):
    def filter(self, rec):
        return rec.levelno in (logging.DEBUG, logging.INFO, logging.WARNING)

def init_logging(level=logging.DEBUG):
    h1 = logging.StreamHandler(sys.stdout)
    h1.setLevel(logging.DEBUG)
    h1.addFilter(InfoFilter())
    h2 = logging.StreamHandler()
    h2.setLevel(logging.ERROR)

    logging.basicConfig(level=level,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[h1, h2])


def exponential_backoff(max_retries: int = 3,
                        base_delay_s: int = 1,
                        should_retry: Callable[[Exception], bool] = lambda _: True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while True:
                try:
                    result_func = func(*args, **kwargs)
                    return result_func
                except Exception as ex:
                    logging.warning(f"Attempt {retries + 1} failed: {ex}")
                    if retries < max_retries and should_retry(ex):
                        delay = (base_delay_s * 2 ** retries +
                                 random.uniform(0, 1))
                        logging.warning(f"Retrying in {delay:.2f} seconds...")
                        sleep(delay)
                        retries += 1
                    else:
                        raise
        return wrapper
    return decorator


def get_auth_token():
    try:
        parser = configparser.ConfigParser()
        config_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), CONFIG_FILE)
        parser.read(config_filepath)
        return parser['auth']['token']
    except Exception as ex:
        raise ValueError(f'Failed to retrieve watttime credentials: {ex}') from ex


def prepare_insert_query(data):
    carbon_intensity = data.get('carbonIntensity', 0.0)
    if carbon_intensity is None:
        carbon_intensity = 0.0

    low_carbon_percentage = 0.00
    renewable_percentage = 0.00

    datetime = data['datetime']

    zone_id = data['zone']

    query = """
    INSERT INTO EMapCarbonIntensity(DateTime, ZoneId, CarbonIntensity, LowCarbonPercentage, RenewablePercentage)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (DateTime, ZoneId) DO UPDATE SET
    CarbonIntensity = EXCLUDED.CarbonIntensity,
    LowCarbonPercentage = EXCLUDED.LowCarbonPercentage,
    RenewablePercentage = EXCLUDED.RenewablePercentage;
    """

    return query, (datetime, zone_id, carbon_intensity, low_carbon_percentage, renewable_percentage)


@exponential_backoff(should_retry=lambda ex: ex.response.status_code == 429)
def fetch(zone, session: requests.Session):
    logging.info(f"Fetching {zone} carbon data from electricity map API ...")
    url = f"https://api-access.electricitymaps.com/free-tier/carbon-intensity/history?zone={zone}"
    headers = {
        "auth-token": get_auth_token()
    }

    history_response = session.get(url, headers=headers)
    history_response.raise_for_status()
    return history_response


def update(conn, history_response_json):
    logging.info(f"Updating database ...")
    assert 'history' in history_response_json, "history not found in response"
    for history in history_response_json['history']:
        query, data = prepare_insert_query(history)
        with conn, conn.cursor() as cur:
            cur.execute(query, data)


def fetch_and_update(zone, session: requests.Session, conn):
    try:
        logging.info(f"Working on {zone} ...")
        history_response = fetch(zone, session)
        assert history_response.ok, "Request failed %d: %s" % (
            history_response.status_code, history_response.text)
        history_response_json = history_response.json()

        update(conn, history_response_json)
    except Exception as ex:
        logging.error(f"Error occurred while processing {zone}: {ex}")
        logging.error(traceback.format_exc())


def get_all_electricity_zones():
    logging.info(f"Getting all zones from electricity map API ...")

    url = "https://api-access.electricitymaps.com/free-tier/zones"
    headers = {
        "auth-token": get_auth_token()
    }

    zone_response = requests.get(url, headers=headers)

    assert zone_response.ok, f"Error in EMap request for getting all zones in electricity map ({zone_response.status_code}): {zone_response.text}"

    zone_response_json = zone_response.json()
    logging.info(f"Got {len(zone_response_json)} zones from electricity map API ...")

    return zone_response_json.keys()


def crawl():
    session = requests.Session()
    conn = get_db_connection()

    for zone in get_all_electricity_zones():
        fetch_and_update(zone, session, conn)
    conn.close()


def backfill(zones):
    logging.info(f"Backfilling {zones} ...")
    raise NotImplementedError()


def parse_args():
    parser = argparse.ArgumentParser(description='Crawl electricity map API and update database.')
    parser.add_argument('-B', '--backfill', action='store_true', help='Run backfill')
    return parser.parse_args()


if __name__ == '__main__':
    init_logging(level=logging.INFO)

    args = parse_args()
    if args.backfill:
        backfill()
    else:
        crawl()
