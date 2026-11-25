#!/bin/bash
# Stop shutter: the node containers first (SIGTERM — a running unit goes back to the queue and continues after the next
# ./start.sh), then the UI. Results the UI has not fetched yet wait on the main node and are saved after the next start.
set -euo pipefail
cd "$(dirname "$0")"
./deploy.sh down
PORT=${SHUTTER_MAC_PORT:-8310}
ui(){ { ps -p "$1" -o command= 2>/dev/null || tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null; } | grep -q 'mac/mac.py'; }
P=$(cat mac/mac.pid 2>/dev/null || true)
[ -n "$P" ] && ! ui "$P" && P=   # stale pid file
if [ -z "$P" ] && command -v lsof >/dev/null; then   # a UI started before mac.pid existed
  P=$(lsof -nP -iTCP:$PORT -sTCP:LISTEN -t 2>/dev/null || true)
fi
if [ -n "$P" ]; then
  kill $P 2>/dev/null || true
  for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 $P 2>/dev/null || break; sleep 0.5; done
fi
rm -f mac/mac.pid
echo "stopped (nodes, UI)"
