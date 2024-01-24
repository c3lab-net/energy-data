#!/usr/bin/env python3

import argparse
import configparser
import csv
import io
import logging
import math
import os
import random
import re
import sys
import time
import traceback
from time import sleep
from typing import Callable, Optional
import psycopg2
import psycopg2.extras
from psycopg2 import sql

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import chromedriver_binary  # Adds chromedriver binary to path
from selenium.webdriver.chrome.options import Options

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

BULK_INSERT_QUERY_BASE = sql.SQL("""
WITH t AS (
    INSERT INTO EMapCarbonIntensity(DateTime, ZoneId, CarbonIntensity, LowCarbonPercentage, RenewablePercentage)
    VALUES %s
    ON CONFLICT (DateTime, ZoneId) DO {}
    RETURNING xmax
)
SELECT COUNT(*) AS count_all,
    SUM(CASE WHEN xmax = 0 THEN 1 ELSE 0 END) AS count_insert,
    SUM(CASE WHEN xmax::text::int > 0 THEN 1 ELSE 0 END) AS count_update
FROM t;
""")

BULK_INSERT_QUERY_WITH_UPDATE = BULK_INSERT_QUERY_BASE.format(
    sql.SQL("""UPDATE SET
            CarbonIntensity = EXCLUDED.CarbonIntensity,
            LowCarbonPercentage = EXCLUDED.LowCarbonPercentage,
            RenewablePercentage = EXCLUDED.RenewablePercentage"""))

BULK_INSERT_QUERY_NO_UPDATE = BULK_INSERT_QUERY_BASE.format(
    # WHERE FALSE clause never executes, but maintains return value syntax
    sql.SQL("""UPDATE SET
                CarbonIntensity = EMapCarbonIntensity.CarbonIntensity WHERE FALSE"""))

def upload_to_database(rows: list[tuple], update_existing: bool,
                       conn: Optional[psycopg2.extensions.connection] = None, set_utc: bool = False):
    logging.info(f"Uploading {len(rows)} rows to database ...")
    if len(rows) == 0:
        return
    try:
        if conn is None:
            conn = get_db_connection()
        with conn, conn.cursor() as cur:
            if set_utc:
                cur.execute("SET TIME ZONE 'UTC';")
            query = BULK_INSERT_QUERY_WITH_UPDATE if update_existing else BULK_INSERT_QUERY_NO_UPDATE
            result = psycopg2.extras.execute_values(cur, query, rows, fetch=True, page_size=len(rows))
    except psycopg2.Error as ex:
        raise ValueError ("Failed to upload new data") from ex
    (count_all, count_insert, count_update) = result[0]
    logging.info(f"Inserted {count_insert} rows, updated {count_update} rows, total {count_all} rows")


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


def prepare_insert_query_args_from_history(history: list[dict]) -> list[tuple]:
    rows = []
    for entry in history:
        datetime = entry['datetime']
        zone_id = entry['zone']
        carbon_intensity = entry.get('carbonIntensity') or 'NaN'
        low_carbon_percentage = 'NaN'
        renewable_percentage = 'NaN'
        rows.append((datetime, zone_id, carbon_intensity, low_carbon_percentage, renewable_percentage))
    return rows


def update(conn, history_response_json, update_existing: bool):
    logging.info(f"Updating database ...")

    assert 'history' in history_response_json, "history not found in response"
    rows = prepare_insert_query_args_from_history(history_response_json['history'])
    upload_to_database(rows, update_existing, conn)


def fetch_and_update(zone, session: requests.Session, conn, update_existing: bool):
    try:
        logging.info(f"Working on {zone} ...")
        history_response = fetch(zone, session)
        assert history_response.ok, "Request failed %d: %s" % (history_response.status_code, history_response.text)

        update(conn, history_response.json(), update_existing)
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


def crawl(update_existing: bool):
    session = requests.Session()
    conn = get_db_connection()

    for zone in get_all_electricity_zones():
        fetch_and_update(zone, session, conn, update_existing)
    conn.close()


def save_data_to_csv(url: str, data: bytes):
    filename = url.split('/')[-1]
    if not filename.endswith('.csv'):
        filename += '.csv'
    logging.info(f'\tSaving to {filename} ...')
    with open(filename, 'wb') as file:
        file.write(data)

def prepare_insert_query_args_from_csv(csv_data: bytes) -> list[tuple]:
    rows = []
    with io.StringIO(csv_data.decode('utf-8')) as f:
        reader = csv.DictReader(f)
        for row in reader:
            datetime = row['Datetime (UTC)']
            zone_id = row['Zone Id']
            carbon_intensity = row['Carbon Intensity gCO₂eq/kWh (LCA)'] or 'NaN'
            low_carbon_percentage = row['Low Carbon Percentage'] or 'NaN'
            renewable_percentage = row['Renewable Percentage'] or 'NaN'
            rows.append((datetime, zone_id, carbon_intensity, low_carbon_percentage, renewable_percentage))
    return rows

def get_chrome_driver():
    options = Options()
    options.add_argument('--headless')
    return webdriver.Chrome(options=options)

driver = None

def get_dynamic_website_html(url: str):
    wait_s = 5
    logging.info(f'Loading dynamic website {url} with {wait_s}s wait period ...')
    driver.get(url)
    time.sleep(wait_s)
    return driver.page_source

def backfill(zones: list[str], save_csvs: bool, year: str, update_existing: bool):
    logging.info(f"Backfilling {zones if zones else 'all zones'} ...")
    HOST = "https://www.electricitymaps.com"
    BACKFILL_HOME_URL = f"{HOST}/data-portal"

    global driver
    driver = get_chrome_driver()

    page_html = get_dynamic_website_html(BACKFILL_HOME_URL)
    soup = BeautifulSoup(page_html, 'html.parser')

    # Find all the 'Get Data' buttons or their equivalent to extract the URLs
    data_urls = []
    zone_counts = []
    for a_tag in soup.find_all('a'):
        if not a_tag.find('div', text='Get data'):
            continue
        data_urls.append(a_tag['href'])
        div_country_name = a_tag.find('div', class_='text-block-31 country-name truncate')
        if div_country_name:
            # regex_zones_count = re.compile(r'(\d+) zones available')
            div_tag = a_tag.find('div', class_='div-block-13')
            zone_count = div_tag.find('div').text
        else:
            zone_count = math.nan
        zone_counts.append(zone_count)
        logging.info(f'{div_country_name.text} ({zone_count}): {a_tag["href"]}')


    # Iterate through each URL, navigate to the page, and find the CSV download link
    for url, zone_count in zip(data_urls, zone_counts):
        if zones and all(zone not in url for zone in zones):
            # logging.info(f'Skipping {url} ...')
            continue

        zone_name = url.split('/')[-1]

        logging.info(f'Loading {HOST + url} ...')
        page_html = get_dynamic_website_html(HOST + url)
        soup = BeautifulSoup(page_html, 'html.parser')

        csv_urls = set()
        for a_tag in soup.find_all('a', attrs={'data-link': True}):
            if a_tag.find('div', text='CSV'):
                csv_urls.add(a_tag['data-link'])

        logging.debug(f'Found {len(csv_urls)} unique CSV files ...')

        # Download each CSV file
        filtered_urls = []
        for csv_url in csv_urls:
            if 'hourly' not in csv_url:
                logging.debug(f'\tSkipping non-hourly file {csv_url} ...')
                continue

            filename = csv_url.split('/')[-1]
            if str(year) not in filename:
                logging.debug(f'\tSkipping non-{year} file {csv_url} ...')
                continue

            filtered_urls.append(csv_url)

        logging.info(f'Expected {zone_count} zones. Found {len(filtered_urls)} hourly CSV files for {zone_name} ...')
        for csv_url in filtered_urls:
            logging.info(f'\tDownloading {csv_url} ...')
            csv_data = requests.get(csv_url).content

            if save_csvs:
                save_data_to_csv(csv_url, csv_data)
            else:
                rows = prepare_insert_query_args_from_csv(csv_data)
                # Let psycopg2 know timestamps are in UTC
                upload_to_database(rows, update_existing, set_utc=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Crawl electricity map API and update database.')
    parser.add_argument('-B', '--backfill', action='store_true', help='Run backfill')
    parser.add_argument('-Y', '--year', type=int, help='Year to backfill')
    parser.add_argument('-z', '--zones', type=str, nargs='*', help='Zones to crawl')
    parser.add_argument('-s', '--save-csvs', action='store_true', help='Save CSVs')
    parser.add_argument('-u', '--update-existing', action='store_true', help='Update existing records in database')

    args = parser.parse_args()

    if not args.backfill:
        if args.zones is not None:
            parser.error('--zones can only be specified when specifying --backfill')
        if args.save_csvs:
            parser.error('--save-csvs can only be specified when specifying --backfill')
        if args.year is not None:
            parser.error('--year can only be specified when specifying --backfill')

    return args


if __name__ == '__main__':
    init_logging(level=logging.INFO)

    args = parse_args()
    if args.backfill:
        backfill(args.zones, args.save_csvs, args.year, args.update_existing)
    else:
        crawl(args.update_existing)
