#!/bin/zsh

cd "$(dirname "$0")/.." || exit 1

while [[ $# -gt 0 ]]; do
case $1 in
  -D|--debug)
    DEBUG=true
    shift
    ;;
  *)
    shift
    ;;
esac
done

set -e
source "$HOME/anaconda3/bin/activate"
conda activate flask
if [[ -n $DEBUG ]]; then
  gunicorn --workers=1 \
    --log-level=debug \
     --bind=localhost:8082 \
     -t=3600 \
     --reload \
    'api:create_app()'
else
  gunicorn --workers=32 \
    --log-level=info \
    -t=120 \
    --access-logfile - \
    --access-logformat '%({X-Real-IP}i)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s %(M)s "%(f)s" "%(a)s"' \
    'api:create_app()'
fi
