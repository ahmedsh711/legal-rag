#!/bin/sh
# First start: send everything to stable until promote.sh opens the canary.
set -eu
SPLIT=/etc/nginx/release/split.conf
if [ ! -f "$SPLIT" ]; then
    printf 'split_clients "${request_id}" $release {\n    * stable;\n}\n' > "$SPLIT"
fi
