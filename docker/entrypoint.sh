#!/bin/sh
# On machines with HTTPS inspection (antivirus, corporate proxy), mount the inspector's root CA at
# /run/local-ca.pem. It is added to a private copy of the trust bundle at start-up: never baked
# into the image, and a no-op everywhere else.
set -e
if [ -f /run/local-ca.pem ] && [ -s /run/local-ca.pem ]; then  # -f: a missing bind source becomes a dir
  cat /etc/ssl/certs/ca-certificates.crt /run/local-ca.pem > /tmp/ca-bundle.pem
  export SSL_CERT_FILE=/tmp/ca-bundle.pem REQUESTS_CA_BUNDLE=/tmp/ca-bundle.pem
fi
# Several uvicorn workers (e.g. --workers 2): each process writes its metrics to files in
# PROMETHEUS_MULTIPROC_DIR and /metrics adds them up. Files from a previous run would be added
# in too, so the folder starts empty.
if [ -n "${PROMETHEUS_MULTIPROC_DIR:-}" ]; then
  rm -rf "$PROMETHEUS_MULTIPROC_DIR" && mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
fi
exec "$@"
