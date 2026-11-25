#!/bin/bash
# Drive the shutter nodes from the Mac (design §0b): one image, one container per node, nothing else on the hosts.
#   ./deploy.sh build          dev: send this checkout's code to every node and build shutter:dev there
#   ./deploy.sh build <tag>    release: every node builds shutter:<tag> from the GitHub repository (REPO)
#   ./deploy.sh up [tag]       (re)start the containers (default TAG in shutter.conf; a dev build: up dev) — main node:
#                              server + worker, others: worker
#   ./deploy.sh start [tag]    start only the nodes without a running container (start.sh: running jobs keep going)
#   ./deploy.sh down           stop them, workers first
#   ./deploy.sh logs           tail of the container logs
# Settings in shutter.conf (template shutter.conf.example): nodes, how to reach them, data / models folders.
# Down never force-kills (docker rm -f = SIGKILL): docker stop (SIGTERM) → the worker hands the running unit back to
# the queue at the next step boundary, unloads the model and exits. SIGKILL mid GPU compute can leave the GPU stuck
# (session #5, C). Stop timeout 900 s = well above the longest step gap (SeedVR2 VAE decode ~160 s, H3 load ~120 s).
set -euo pipefail
cd "$(dirname "$0")"
[ -f shutter.conf ] || { echo "shutter.conf missing: copy shutter.conf.example and edit it"; exit 1; }
. ./shutter.conf
TAG=${TAG:-dev}
MAIN=${NODES%% *}
host() { local h=HOST_$1; echo "${!h}"; }
node() { local n=$1; shift; $SSH "$(host $n)" "bash -s -- $*"; }   # script on stdin
REV=; for n in $NODES; do REV="$n $REV"; done                     # workers first, main last
running() { [ -n "$($SSH "$(host $1)" "docker ps -q --filter label=project=shutter --filter name=^shutter\\$")" ]; }

start_node() {   # node, image tag — main node: server + worker, others: worker
  local n=$1 role server
  if [ $n = $MAIN ]; then role=main; server=http://127.0.0.1:$PORT; else role=worker; server=http://$SERVER_ADDR:$PORT; fi
  node $n $n $role $2 "$(date +%z)" $server $PORT $DATA $MODELS <<'EOF'
set -e
N=$1 ROLE=$2 TAG=$3 Z=$4 SERVER=$5 PORT=$6 DATA=$HOME/$7 MODELS=$HOME/$8
[ ${Z:0:1} = + ] && S=- || S=+; TZS="LOC$S${Z:1:2}:${Z:3:2}"   # Mac time zone (job ids and result folder names)
mkdir -p $DATA/cache/assets; [ $ROLE = main ] && mkdir -p $DATA/state $DATA/results
docker ps -aq --filter label=project=shutter --filter name=^shutter\$ --filter status=exited | xargs -r docker rm >/dev/null
docker run -d --name shutter --label project=shutter --gpus all --user 1000:1000 -e HOME=/tmp -e TZ=$TZS \
  --network host -e SHUTTER_NODE=$N -e SHUTTER_SERVER=$SERVER -e SHUTTER_PORT=$PORT \
  -v $DATA:/work -v $MODELS:/models:ro shutter:$TAG $ROLE >/dev/null
docker ps --filter label=project=shutter --filter name=^shutter\$ --format "$N {{.Names}} {{.Image}} {{.Status}}"
EOF
}

stop_all() {
  for n in $REV; do node $n <<'EOF'
docker ps -aq --filter label=project=shutter --filter name=^shutter\$ | xargs -r sh -c 'docker stop -t 900 "$@" >/dev/null && docker rm "$@" >/dev/null' _
EOF
  done
}

case "${1:-}" in
build)
  TAG=${2:-}
  for n in $NODES; do
    if [ -z "$TAG" ]; then
      COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf - --exclude __pycache__ docker engine server worker |
        $SSH "$(host $n)" "docker build -q --label project=shutter -t shutter:dev -f docker/Dockerfile -" >/dev/null
      echo "built shutter:dev → $n"
    else
      $SSH "$(host $n)" "docker build -q --label project=shutter -t shutter:$TAG -f docker/Dockerfile $REPO#$TAG" >/dev/null
      echo "built shutter:$TAG → $n"
    fi
  done ;;
up)
  stop_all
  for n in $NODES; do start_node $n ${2:-$TAG}; done ;;   # main first: the workers call its server
start)
  for n in $NODES; do if running $n; then echo "$n shutter already running"; else start_node $n ${2:-$TAG}; fi; done ;;
down)
  stop_all ;;
logs)
  for n in $NODES; do echo "== $n"; node $n <<'EOF'
docker ps -aq --filter label=project=shutter --filter name=^shutter\$ | grep -q . && docker logs --tail 30 shutter 2>&1 || echo "(no container)"
EOF
  done ;;
*) sed -n '2,10p' "$0"; exit 1 ;;
esac
