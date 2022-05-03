#!/bin/zsh

cd "$(dirname "$0")"/..

set -e
source $HOME/anaconda3/bin/activate
conda activate py39
./crawler/crawl.py >> ./logs/crawler.log 2>> >(tee -a ./logs/crawler.err >&2)
