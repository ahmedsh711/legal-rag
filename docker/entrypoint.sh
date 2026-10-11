#!/bin/sh
# Behind HTTPS inspection (antivirus, corporate proxy), mount the root CA at /run/local-ca.pem.
# It is appended to a copy of the system bundle at start-up, never baked into the image.
set -e
# -f as well as -s: Docker mounts a missing bind source as an empty directory
if [ -f /run/local-ca.pem ] && [ -s /run/local-ca.pem ]; then
  cat /etc/ssl/certs/ca-certificates.crt /run/local-ca.pem > /tmp/ca-bundle.pem
  export SSL_CERT_FILE=/tmp/ca-bundle.pem REQUESTS_CA_BUNDLE=/tmp/ca-bundle.pem
fi
# Multi-worker metrics: /metrics sums the per-process files in PROMETHEUS_MULTIPROC_DIR, so
# clear files left over from a previous run.
if [ -n "${PROMETHEUS_MULTIPROC_DIR:-}" ]; then
  rm -rf "$PROMETHEUS_MULTIPROC_DIR" && mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
fi
exec "$@"
