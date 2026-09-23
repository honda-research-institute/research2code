#!/usr/bin/env bash
# Serial R2C batch runner — one run at a time (the concurrency protocol),
# survives individual failures, one summary line per run.
#
# Usage:
#   export R2C_SERVER_URL=http://127.0.0.1:<port>   # a running opencode server
#   caffeinate -is bash scripts/run_batch.sh <manifest> [logdir]
#
# (caffeinate keeps the Mac awake for the whole batch; runs are unattended
# and halts simply move on to the next manifest row.)
#
# Manifest: one row per run, `#` comments and blank lines ignored.
#   <paper-path> [wipe] [stop-after=<stage_id>]
# Example:
#   input_papers/bayesian-active-learning.pdf wipe
#   input_papers/ADAM.pdf wipe stop-after=stage_1x
#
# `wipe` removes r2c_runs/<slug> first (slug = paper filename stem) so the
# driver starts fresh instead of resuming. Default log dir:
# r2c_runs/_batch_logs (per-run logs + batch_summary.txt).
# DRY_RUN=1 prints the commands instead of executing.

set -u

MANIFEST="${1:?usage: run_batch.sh <manifest> [logdir]}"
LOGDIR="${2:-r2c_runs/_batch_logs}"
: "${R2C_SERVER_URL:?export R2C_SERVER_URL to a running opencode server first}"

# Batch context for the run-history ledger: the driver stamps each
# invocation's ledger row with the manifest it ran under.
export R2C_BATCH_MANIFEST="$MANIFEST"

# Systemic-failure guard: a real run takes minutes even to its first halt;
# two CONSECUTIVE runs dying this fast means the environment is broken
# (down parse service, dead server, bad env) — stop instead of wiping and
# burning the whole manifest (the 2026-07-02 Marker outage burned 11 rows
# in two minutes).
FAST_FAIL_S="${FAST_FAIL_S:-120}"
fast_fail_streak=0

mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/batch_summary.txt"
echo "=== batch start $(date '+%Y-%m-%d %H:%M:%S') manifest=$MANIFEST ===" | tee -a "$SUMMARY"

while IFS= read -r line || [[ -n "$line" ]]; do
  line="${line%%#*}"
  [[ -z "${line// /}" ]] && continue
  set -- $line
  paper="$1"; shift
  wipe=0
  extra=()
  for tok in "$@"; do
    case "$tok" in
      wipe) wipe=1 ;;
      stop-after=*) extra+=(--stop-after "${tok#stop-after=}") ;;
      *) echo "WARN: unknown manifest token '$tok' (ignored)" | tee -a "$SUMMARY" ;;
    esac
  done
  slug="$(basename "$paper")"; slug="${slug%.*}"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY: slug=$slug wipe=$wipe python3 scripts/run_pipeline.py --paper $paper ${extra[*]:-}"
    continue
  fi

  if [[ $wipe -eq 1 && -d "r2c_runs/$slug" ]]; then
    rm -rf "r2c_runs/$slug"
    echo "[$(date '+%H:%M:%S')] wiped r2c_runs/$slug" | tee -a "$SUMMARY"
  fi

  # Rotate a prior run's log instead of clobbering it — the 7/3 GBALD
  # verified roll's driver log was overwritten by the 7/5 matrix rows,
  # leaving run_events.jsonl as the only behavioral record for the audit.
  # Latest run keeps the stable <slug>.log name; history gets a stamp.
  if [[ -f "$LOGDIR/$slug.log" ]]; then
    mv "$LOGDIR/$slug.log" \
       "$LOGDIR/$slug.log.$(date -r "$LOGDIR/$slug.log" '+%Y%m%d-%H%M%S')"
  fi

  echo "[$(date '+%H:%M:%S')] START $slug ${extra[*]:-}" | tee -a "$SUMMARY"
  t_start=$(date +%s)
  python3 scripts/run_pipeline.py --paper "$paper" ${extra[@]:+"${extra[@]}"} \
    > "$LOGDIR/$slug.log" 2>&1
  rc=$?
  t_run=$(( $(date +%s) - t_start ))

  if [[ $rc -ne 0 && $t_run -lt $FAST_FAIL_S ]]; then
    fast_fail_streak=$((fast_fail_streak + 1))
    if [[ $fast_fail_streak -ge 2 ]]; then
      echo "[$(date '+%H:%M:%S')] ABORT: $fast_fail_streak consecutive runs" \
           "failed in under ${FAST_FAIL_S}s — systemic failure, stopping" \
           "before wiping more rows (see $LOGDIR/$slug.log)" | tee -a "$SUMMARY"
      exit 3
    fi
  else
    fast_fail_streak=0
  fi

  verdict="$(python3 - "$slug" <<'PY'
import json, sys
slug = sys.argv[1]
try:
    try:
        m = json.load(open(f"r2c_runs/{slug}/details/final_manifest.json"))
    except FileNotFoundError:
        # Pre-consolidation layout (runs produced before 2026-07-07).
        m = json.load(open(f"r2c_runs/{slug}/final_manifest.json"))
    d = m.get("delivery") or {}
    print(f"delivery={d.get('label')} run_status={m.get('run_status')}")
except Exception:
    try:
        p = json.load(open(f"r2c_runs/{slug}/.pipeline/progress.json"))
        print(f"no final manifest; progress run_status={p.get('run_status')}")
    except Exception:
        print("no final manifest / progress artifact")
PY
)"
  echo "[$(date '+%H:%M:%S')] DONE  $slug rc=$rc $verdict" | tee -a "$SUMMARY"
done < "$MANIFEST"

echo "=== batch end $(date '+%Y-%m-%d %H:%M:%S') ===" | tee -a "$SUMMARY"
