#!/bin/bash
# Launch the R2C single-window environment: `opencode serve` in background +
# `opencode attach` in foreground. When the TUI exits, the server is killed.
#
# Usage:
#   ./r2c-start.sh [port]
#
# Default port: 4096. Override if 4096 is in use:
#   ./r2c-start.sh 4097

set -euo pipefail

PORT="${1:-4096}"
WORKSPACE="$(pwd)"

# Repo-root .env (gitignored, see .env.example) carries R2C_MODEL and the
# provider API keys; its values apply to everything this launch starts.
if [ -f "${WORKSPACE}/.env" ]; then
  set -a; . "${WORKSPACE}/.env"; set +a
fi
# The model every R2C agent runs on, as opencode's <provider>/<model> id.
export R2C_MODEL="${R2C_MODEL:-google/gemini-flash-latest}"

# Export the port for the TUI agent's bash environment: the /r2c-run and
# /r2c-paradigms commands read R2C_PORT to route their drivers to THIS server.
export R2C_PORT="${PORT}"

# Loopback traffic must never go through an HTTP proxy. Some corporate
# setups export http_proxy without a localhost exemption, which breaks
# `opencode attach`. Prepend the loopback names to both spellings so serve,
# attach, and the driver all inherit them. No-op without proxy vars.
_no_proxy_existing="${no_proxy:-${NO_PROXY:-}}"
export no_proxy="localhost,127.0.0.1${_no_proxy_existing:+,${_no_proxy_existing}}"
export NO_PROXY="${no_proxy}"

# TCP probe from inside this bash process, deliberately not curl: on WSL
# `curl` can resolve to the Windows curl.exe or route through ${http_proxy},
# and either way reports a healthy server as down. bash's /dev/tcp cannot
# be shimmed or proxied.
_port_open() {
  (exec 3<>"/dev/tcp/127.0.0.1/${1}") 2>/dev/null
}

# --- live-view auto-launch: the read-only viewer (fleet page at /, per-run
# live view at /run/<slug>) starts with the environment. Reuses a viewer
# already on the port; R2C_NO_WATCH=1 skips it. The browser open is
# best-effort; the echoed URL is the fallback.
WATCH_PORT="${R2C_WATCH_PORT:-8765}"
WATCH_PID=""
_start_watch() {
  [ -n "${R2C_NO_WATCH:-}" ] && return 0
  if _port_open "${WATCH_PORT}"; then
    echo "r2c-watch live view already up on port ${WATCH_PORT}; reusing it."
  else
    python3 "${WORKSPACE}/scripts/r2c_watch.py" --host 127.0.0.1 \
      --port "${WATCH_PORT}" > "/tmp/r2c-watch-${WATCH_PORT}.log" 2>&1 &
    WATCH_PID=$!
    echo "r2c-watch live view up on http://127.0.0.1:${WATCH_PORT}/ (PID ${WATCH_PID}, log: /tmp/r2c-watch-${WATCH_PORT}.log)"
  fi
  python3 -c "import webbrowser; webbrowser.open('http://127.0.0.1:${WATCH_PORT}/')" \
    2>/dev/null || true
}

# If the chosen port is already serving, just attach. A watch server
# started on this branch is not torn down on exit (exec replaces this
# shell); the next r2c-start reuses it by port, so at most one outlives
# its TUI.
if _port_open "${PORT}"; then
  echo "opencode server already up on port ${PORT}; attaching."
  _start_watch
  exec opencode attach "http://127.0.0.1:${PORT}" --dir "${WORKSPACE}"
fi

# Otherwise, boot serve in the background and tear it down on TUI exit,
# along with the watch server if this launch started one.
LOGFILE="/tmp/r2c-server-${PORT}.log"
opencode serve --port "${PORT}" > "${LOGFILE}" 2>&1 &
SERVE_PID=$!
trap 'kill "${SERVE_PID}" 2>/dev/null || true; [ -n "${WATCH_PID}" ] && kill "${WATCH_PID}" 2>/dev/null || true' EXIT INT TERM

# Wait up to 60s for the server to start accepting connections. Generous on
# purpose: on networks where outbound fetches are silently dropped, opencode
# can stall on its startup models.dev fetch before binding. The loop exits
# the moment the port answers, so healthy machines pay nothing extra.
READY=""
for _ in $(seq 1 120); do
  if ! kill -0 "${SERVE_PID}" 2>/dev/null; then
    echo "ERROR: opencode serve (PID ${SERVE_PID}) exited during startup." >&2
    echo "  Check ${LOGFILE} for details." >&2
    exit 1
  fi
  if _port_open "${PORT}"; then
    READY=1
    break
  fi
  sleep 0.5
done

if [ -z "${READY}" ]; then
  echo "ERROR: opencode serve is still running (PID ${SERVE_PID}) but nothing" >&2
  echo "  accepted a connection on 127.0.0.1:${PORT} within 60s." >&2
  echo "  Listener state for port ${PORT}:" >&2
  { ss -ltn 2>/dev/null || netstat -an 2>/dev/null; } | grep "${PORT}" >&2 \
    || echo "  (no listener reported — ss/netstat missing or port not bound)" >&2
  echo "  Last lines of ${LOGFILE}:" >&2
  tail -n 15 "${LOGFILE}" >&2 || true
  echo "  Report the output above plus the newest file in" >&2
  echo "  ~/.local/share/opencode/log when escalating." >&2
  exit 1
fi

echo "opencode server up on port ${PORT} (PID ${SERVE_PID}, log: ${LOGFILE})"
_start_watch
echo "attaching TUI..."
opencode attach "http://127.0.0.1:${PORT}" --dir "${WORKSPACE}"
