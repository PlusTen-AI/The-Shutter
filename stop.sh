#!/bin/bash
# Stop shutter: the node containers first (SIGTERM — a running unit goes back to the queue and continues after the next
# ./start.sh), then the UI. Results the UI has not saved yet wait on the main node and are saved after the next start.
set -euo pipefail
cd "$(dirname "$0")"
./deploy.sh down
echo "stopped (nodes, UI)"
