#!/usr/bin/zsh

# This script allows local access from certain users.

typeset -a psql_users
psql_users=(
    restapi_ro
    crawler_rw
)

for psql_user in "${psql_users[@]}"; do
    echo -e "local\t""electricity-data\t$psql_user\t""trust" | sudo tee -a /etc/postgresql/14/main/pg_hba.conf > /dev/null
done

sudo service postgresql reload
