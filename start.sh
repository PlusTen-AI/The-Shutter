#!/bin/bash
# Start shutter: the node containers and the UI container on the main node (image shutter:$TAG from shutter.conf).
# Whatever already runs is left alone, so running jobs keep going. Open the printed UI address in any browser on your
# network. Stop everything with ./stop.sh.
set -euo pipefail
cd "$(dirname "$0")"
exec ./deploy.sh start
