#!/usr/bin/env python3

import argparse
import requests_cache
from urllib.parse import urlparse, parse_qs


BA_QUERY_URL = 'https://api-access.electricitymaps.com/free-tier/home-assistant'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir-path', '-d', type=str, required=True, help='Directory path of the cache')
    parser.add_argument('--output', '-o', type=argparse.FileType('w'), required=True,
                        help='Output file name')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    args.output.write('lat,lon,ba\n')
    session = requests_cache.CachedSession(backend='filesystem', cache_name=args.dir_path)
    for key in session.cache.responses:
        response = session.cache.responses[key]
        if response.url.startswith(BA_QUERY_URL) and response.ok:
            params = parse_qs(urlparse(response.url).query)
            lat = params['lat'][0]
            lon = params['lon'][0]
            ba = response.json()['countryCode']
            args.output.write(f'{lat},{lon},{ba}\n')
        else:
            print('Ignored', key, response.url, response.status_code)

if __name__ == '__main__':
    main()
