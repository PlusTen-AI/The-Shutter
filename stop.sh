#!/bin/bash
# Stop shutter: the node containers first (SIGTERM — a running unit goes back to the queue and continues after the next
# ./start.sh), then the UI. Results the UI has not fetched yet wait on the main node and are saved after the next start.
set -euo pipefail
cd "$(dirname "$0")"
./deploy.sh down
PORT=${SHUTTER_MAC_PORT:-8310}
P=$(lsof -nP -iTCP:$PORT -sTCP:LISTEN -t 2>/dev/null || true)
if [ -n "$P" ]; then
  kill $P
  for _ in 1 2 3 4 5 6 7 8 9 10; do lsof -nP -iTCP:$PORT -sTCP:LISTEN -t >/dev/null 2>&1 || break; sleep 0.5; done
fi
echo "stopped (nodes, UI)"
