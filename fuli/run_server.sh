#!/usr/bin/env bash
# start/stop/restart/status for orchestrator_server, pinned to CPUs 1-12
# — Redis (in Docker) already owns 13-15, CPU 0 is left free for the OS
# and everything else, so this server never contends with either.
#
# Usage: ./run_server.sh {start|stop|restart|status|logs [n]}
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" # main.cpp's config paths (config/, include/VAD_DB/, data/)
                  # are resolved relative to the launch directory, not to
                  # this script's location — this cd is load-bearing.

BINARY="./build/orchestrator_server"
CPU_LIST="${CPU_LIST:-1-12}"
RUN_DIR="$SCRIPT_DIR/.run"
PID_FILE="$RUN_DIR/orchestrator_server.pid"
LOG_FILE="$RUN_DIR/orchestrator_server.log"

mkdir -p "$RUN_DIR"

is_running() {
  [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

start() {
  if is_running; then
    echo "orchestrator_server already running (pid $(cat "$PID_FILE"))"
    return 0
  fi
  if [[ ! -x "$BINARY" ]]; then
    echo "error: $BINARY not found or not executable — build it first (cd build && make)" >&2
    exit 1
  fi

  # setsid makes taskset the leader of a brand new session AND process
  # group, so it (and orchestrator_server underneath it) survive this
  # script exiting, and later a single `kill -TERM -<pid>` (negative =
  # process group) reaches orchestrator_server too — taskset forks and
  # waits on it rather than exec-replacing itself, so $! is taskset's
  # pid, not the server's.
  setsid taskset -c "$CPU_LIST" "$BINARY" >> "$LOG_FILE" 2>&1 < /dev/null &
  echo $! > "$PID_FILE"
  disown

  sleep 1
  if is_running; then
    echo "orchestrator_server started (pid $(cat "$PID_FILE"), cpus $CPU_LIST) — logging to $LOG_FILE"
  else
    echo "orchestrator_server failed to start — check $LOG_FILE" >&2
    rm -f "$PID_FILE"
    exit 1
  fi
}

stop() {
  if ! is_running; then
    echo "orchestrator_server is not running"
    rm -f "$PID_FILE"
    return 0
  fi

  local pid
  pid="$(cat "$PID_FILE")"
  echo "stopping orchestrator_server (pid $pid)..."
  # SIGTERM to the whole process group -> main.cpp's signal handler
  # saves the Faiss index before exiting (see main()'s signals.async_wait).
  kill -TERM "-$pid" 2>/dev/null || true

  for _ in $(seq 1 20); do
    is_running || break
    sleep 0.5
  done

  if is_running; then
    echo "still alive after graceful stop — sending SIGKILL" >&2
    kill -KILL "-$pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
  echo "stopped"
}

status() {
  if is_running; then
    echo "running (pid $(cat "$PID_FILE"), cpus $CPU_LIST)"
  else
    echo "not running"
  fi
}

case "${1:-}" in
  start)   start ;;
  stop)    stop ;;
  restart) stop; start ;;
  status)  status ;;
  logs)    tail -n "${2:-50}" -f "$LOG_FILE" ;;
  *)
    echo "usage: $0 {start|stop|restart|status|logs [n]}" >&2
    exit 1
    ;;
esac
