#!/bin/zsh

cd "$(dirname "$0")"/..

set -e
source "$HOME/anaconda3/bin/activate"
conda activate crawler
./crawler/crawl_emap.py -u >> ./logs/crawler.emap.log 2>> >(tee -a ./logs/crawler.emap.err >&2)
