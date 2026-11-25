#!/bin/bash
# shutter container: main = queue server + worker (the main node), worker = worker only.
# docker stop (SIGTERM): the worker goes first — it hands its running unit back to the server at the next step boundary,
# unloads the model and exits — then the server. If either process ends on its own, the other is stopped too.
cd /opt/shutter || exit 1
role=${1:-worker}
S=
if [ "$role" = main ]; then
  python -u server/server.py & S=$!
  export SHUTTER_SHARED_CACHE=1   # this worker uses the server's cache folder; the server cleans it
elif [ "$role" != worker ]; then
  exec "$@"   # anything else: run it (e.g. python /opt/shutter/fetch_models.py)
fi
python -u worker/worker.py & W=$!
stop() {
  kill -TERM "$W" 2>/dev/null; wait "$W"
  [ -n "$S" ] && { kill -TERM "$S" 2>/dev/null; wait "$S"; }
  exit 0
}
trap stop TERM INT
wait -n
r=$?
stop_rest() { kill -TERM "$W" ${S:+"$S"} 2>/dev/null; wait; }
stop_rest
exit $r
