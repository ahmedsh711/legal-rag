#!/bin/sh
# Read-only role for Grafana's "Drift history" data source: SELECT on the monitoring database
# only. Grafana allows anonymous viewers, so it must never connect as the superuser.
# Runs once on a fresh volume (the zz- prefix sorts it after init.sql). On an existing volume:
#   docker exec -e GRAFANA_DB_PASSWORD legal-rag-postgres-1 sh /docker-entrypoint-initdb.d/zz-grafana-ro.sh
set -eu
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname monitoring \
     -v pw="$GRAFANA_DB_PASSWORD" <<'SQL'
SELECT 'CREATE ROLE grafana_ro LOGIN' WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'grafana_ro')
\gexec
ALTER ROLE grafana_ro PASSWORD :'pw';
GRANT CONNECT ON DATABASE monitoring TO grafana_ro;
GRANT USAGE ON SCHEMA public TO grafana_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO grafana_ro;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO grafana_ro;
SQL
