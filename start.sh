#!/bin/bash
# Start shutter from the machine you work on (macOS, Linux or Windows WSL): the node containers (image shutter:$TAG from
# shutter.conf) and the UI (mac/mac.py, http://127.0.0.1:8310, log mac/mac.log). Whatever already runs is left alone,
# so running jobs keep going. Stop everything with ./stop.sh.
set -euo pipefail
cd "$(dirname "$0")"
./deploy.sh start
PORT=${SHUTTER_MAC_PORT:-8310}
PY=python3; [ -x .venv/bin/python ] && PY=.venv/bin/python   # Linux / WSL: the .venv with Pillow + PyAV
up(){ "$PY" -c "import socket, sys; s = socket.socket(); s.settimeout(1); sys.exit(s.connect_ex(('127.0.0.1', $PORT)) != 0)"; }
if up; then
  echo "UI already running: http://127.0.0.1:$PORT"
else
  nohup "$PY" mac/mac.py >> mac/mac.log 2>&1 &
  echo $! > mac/mac.pid
  for _ in 1 2 3 4 5 6 7 8 9 10; do up && break; sleep 0.5; done
  up && echo "UI http://127.0.0.1:$PORT" || { echo "UI did not start: see mac/mac.log"; exit 1; }
fi
