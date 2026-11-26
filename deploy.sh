#!/bin/bash
# Drive the shutter nodes from the machine you deploy from (design §0b): one image, one container per node (the main node
# also runs the UI container shutter-ui), nothing else on the hosts.
#   ./deploy.sh build          dev: send this checkout's code to every node and build shutter:dev there
#   ./deploy.sh build <tag>    release: every node builds shutter:<tag> from the GitHub repository (REPO)
#   ./deploy.sh up [tag]       (re)start the containers (default TAG in shutter.conf; a dev build: up dev) — main node:
#                              server + worker and the UI, others: worker
#   ./deploy.sh start [tag]    start only what is not running (start.sh: running jobs keep going)
#   ./deploy.sh down           stop them, workers first, the UI last
#   ./deploy.sh logs           tail of the container logs
# Settings in shutter.conf (template shutter.conf.example): nodes, how to reach them, data / models folders, UI address,
# storage (the main node's folder, or a folder shared from your computer over SMB).
# Down never force-kills (docker rm -f = SIGKILL): docker stop (SIGTERM) → the worker hands the running unit back to
# the queue at the next step boundary, unloads the model and exits. SIGKILL mid GPU compute can leave the GPU stuck
# (session #5, C). Stop timeout 900 s = well above the longest step gap (SeedVR2 VAE decode ~160 s, H3 load ~120 s).
set -euo pipefail
cd "$(dirname "$0")"
[ -f shutter.conf ] || { echo "shutter.conf missing: copy shutter.conf.example and edit it"; exit 1; }
. ./shutter.conf
TAG=${TAG:-dev}
MAIN=${NODES%% *}
UI_PORT=${UI_PORT:-8310}
host() { local h=HOST_$1; echo "${!h}"; }
node() { local n=$1; shift; $SSH "$(host $n)" "bash -s -- $*"; }   # script on stdin
REV=; for n in $NODES; do REV="$n $REV"; done                     # workers first, main last
running() { [ -n "$($SSH "$(host $1)" "docker ps -q --filter label=project=shutter --filter name=^${2:-shutter}\\$")" ]; }
ui_url() { local h; h=$(host $MAIN); echo "http://${UI_ADDR:-${h#*@}}:$UI_PORT"; }

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

start_ui() {   # image tag — the UI container on the main node, storage mounted at /storage (shutter.conf STORAGE)
  # storage settings go in the script, not as arguments: empty ones vanish through a jump host, a password shows in ps
  { printf 'ST=%q SU=%q PASS=%q\n' "${STORAGE:-}" "${STORAGE_USER:-}" "${STORAGE_PASS:-}"; cat <<'EOF'
set -e
TAG=$1 Z=$2 PORT=$3 UI_PORT=$4 DATA=$HOME/$5 N=$6
[ ${Z:0:1} = + ] && S=- || S=+; TZS="LOC$S${Z:1:2}:${Z:3:2}"
mkdir -p $DATA/ui
docker ps -aq --filter label=project=shutter --filter name=^shutter-ui\$ --filter status=exited --filter status=created | xargs -r docker rm >/dev/null
case "$ST" in   # --mount, not -v: a missing folder (external disk unplugged) is an error, never created empty
  //*) H=${ST#//}; H=${H%%/*}; LABEL=$ST M=type=volume,src=shutter-storage,dst=/storage   # SMB: mounted at container start
       docker volume rm shutter-storage >/dev/null 2>&1 || true
       docker volume create --label project=shutter --driver local --opt type=cifs --opt "device=$ST" \
         --opt "o=addr=$H,username=$SU,password=$PASS,uid=1000,gid=1000,file_mode=0644,dir_mode=0755,vers=3.0,iocharset=utf8,noserverino" \
         shutter-storage >/dev/null ;;
  '') LABEL="$N ~/studio/works" M=type=bind,src=$HOME/studio/works,dst=/storage; mkdir -p $HOME/studio/works ;;   # apart from DATA (cache, queue)
  /*) LABEL="$N $ST" M=type=bind,src=$ST,dst=/storage ;;
  *:/*) H=${ST%%:*}; LABEL=$ST M=type=volume,src=shutter-storage,dst=/storage   # NFS: no password, the export allows this node
       docker volume rm shutter-storage >/dev/null 2>&1 || true
       docker volume create --label project=shutter --driver local --opt type=nfs --opt "device=:${ST#*:}" \
         --opt "o=addr=$H,rw,nfsvers=3,nolock,soft,timeo=50,retrans=3,nocto,actimeo=30" shutter-storage >/dev/null ;;
  *) LABEL="$N ~/$ST" M=type=bind,src=$HOME/$ST,dst=/storage ;;
esac
run() { docker run -d --name shutter-ui --label project=shutter --user 1000:1000 -e HOME=/tmp -e TZ=$TZS --network host \
  -e SHUTTER_MAC_HOST=0.0.0.0 -e SHUTTER_MAC_PORT=$UI_PORT -e SHUTTER_MAC_SERVER=http://127.0.0.1:$PORT \
  -e SHUTTER_MAC_SETTINGS=/ui/settings.json -e SHUTTER_MAC_STORAGE=/storage -e "SHUTTER_MAC_STORAGE_LABEL=$LABEL" \
  -v $DATA/ui:/ui "$@" shutter:$TAG ui >/dev/null; }
if ! run --mount "$M" 2>/dev/null; then   # storage cannot be reached: the UI runs without it (red dot, results wait)
  docker rm shutter-ui >/dev/null 2>&1 || true
  echo "storage not reachable: $LABEL — the UI starts without it; ./deploy.sh up once it is back"
  run
fi
docker ps --filter label=project=shutter --filter name=^shutter-ui\$ --format "{{.Names}} {{.Image}} {{.Status}}"
EOF
  } | $SSH "$(host $MAIN)" "bash -s -- $1 $(date +%z) $PORT $UI_PORT $DATA $MAIN"
  echo "UI $(ui_url)"
}

stop_ui() {
  node $MAIN <<'EOF'
docker ps -aq --filter label=project=shutter --filter name=^shutter-ui\$ | xargs -r sh -c 'docker stop -t 30 "$@" >/dev/null && docker rm "$@" >/dev/null' _
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
      X=(); tar --version 2>/dev/null | grep -q bsdtar && X=(--no-xattrs --no-mac-metadata)   # macOS: no ._ files
      COPYFILE_DISABLE=1 tar ${X[@]+"${X[@]}"} -czf - --exclude __pycache__ docker engine server worker mac/mac.py mac/script.py mac/ui |
        $SSH "$(host $n)" "docker build -q --label project=shutter -t shutter:dev -f docker/Dockerfile -" >/dev/null
      echo "built shutter:dev → $n"
    else
      $SSH "$(host $n)" "docker build -q --label project=shutter -t shutter:$TAG -f docker/Dockerfile $REPO#$TAG" >/dev/null
      echo "built shutter:$TAG → $n"
    fi
  done ;;
up)
  stop_all; stop_ui
  for n in $NODES; do start_node $n ${2:-$TAG}; done   # main first: the workers call its server
  start_ui ${2:-$TAG} ;;
start)
  for n in $NODES; do if running $n; then echo "$n shutter already running"; else start_node $n ${2:-$TAG}; fi; done
  if running $MAIN shutter-ui; then echo "UI already running: $(ui_url)"; else start_ui ${2:-$TAG}; fi ;;
down)
  stop_all; stop_ui ;;
logs)
  for n in $NODES; do echo "== $n"; node $n <<'EOF'
docker ps -aq --filter label=project=shutter --filter name=^shutter\$ | grep -q . && docker logs --tail 30 shutter 2>&1 || echo "(no container)"
EOF
  done
  echo "== UI"; node $MAIN <<'EOF'
docker ps -aq --filter label=project=shutter --filter name=^shutter-ui\$ | grep -q . && docker logs --tail 30 shutter-ui 2>&1 || echo "(no container)"
EOF
  ;;
*) sed -n '2,10p' "$0"; exit 1 ;;
esac
