#!/bin/bash
# Start shutter from the machine you work on: the node containers (image shutter:$TAG from shutter.conf) and the UI
# (mac/mac.py, http://127.0.0.1:8310, log mac/mac.log). Whatever already runs is left alone, so running jobs keep going.
# Stop everything with ./stop.sh.
set -euo pipefail
cd "$(dirname "$0")"
./deploy.sh start
PORT=${SHUTTER_MAC_PORT:-8310}
if lsof -nP -iTCP:$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
  echo "UI already running: http://127.0.0.1:$PORT"
else
  nohup python3 mac/mac.py >> mac/mac.log 2>&1 &
  for _ in 1 2 3 4 5 6 7 8 9 10; do lsof -nP -iTCP:$PORT -sTCP:LISTEN -t >/dev/null 2>&1 && break; sleep 0.5; done
  lsof -nP -iTCP:$PORT -sTCP:LISTEN -t >/dev/null 2>&1 && echo "UI http://127.0.0.1:$PORT" || { echo "UI did not start: see mac/mac.log"; exit 1; }
fi
