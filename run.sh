#!/usr/bin/env bash
# Wraps a pipeline module run with the systemd-oomd isolation + logging setup
# worked out in report.md ("Root cause of 'it killed my whole tmux session'"):
#   - systemd-run --user --scope puts the job in its own cgroup, separate from
#     the tmux server's, so an oom-kill takes down only the job, not the
#     whole tmux session.
#   - python -u for unbuffered stdout, so the log reflects real progress
#     instead of sitting in python's buffer until exit.
#   - tee to a timestamped file under logs/, so output survives even if the
#     shell/tmux pane itself goes away.
# RETRIES=N (env var, default 1) retries an oom-killed run up to N times --
# ONLY safe for a module that's resumable (process_scene()'s _scene_done()
# skips already-finished scenes, so a kill partway through prepare_
# training_data just costs the one scene in progress, not the whole run).
# Do NOT set RETRIES>1 for train_nbdnet.py -- it has no checkpoint-resume,
# a retry would silently restart training from epoch 0, discarding
# everything the killed run had done.
#
# Before each attempt (not just the first), waits for system-wide memory
# pressure (/proc/pressure/memory, avg10) to actually be low, instead of a
# blind fixed sleep. Confirmed via journalctl on the real box: a
# systemd-oomd kill of a COMPLETELY UNRELATED unit
# (snap.snapd-desktop-integration...) landed 16 seconds before our job got
# killed in the same window, and during a later RETRIES=15/sleep-15s run,
# systemd-oomd eventually killed the tmux-spawn scope itself too -- i.e.
# this box was in a genuine system-wide memory-pressure episode (other
# tenants), and retrying on a fixed short timer just adds another
# memory-hungry process into that same bad window, worsening odds of
# oomd escalating to a bigger unit (like tmux) instead of just ours.
# Waiting for pressure to actually drop first is the fix, not a longer
# fixed delay.
PRESSURE_FILE=/proc/pressure/memory
PRESSURE_THRESHOLD=2.0   # percent, avg10 -- comfortably above idle noise,
                         # well below the levels seen during the bad window
PRESSURE_MAX_WAIT=1800   # give up waiting after 30min and try anyway

wait_for_low_pressure() {
    [ -r "$PRESSURE_FILE" ] || return 0  # no PSI support -- nothing to gate on
    local waited=0
    while true; do
        local avg10
        avg10=$(awk -F'avg10=' '/^some/{split($2,a," "); print a[1]}' "$PRESSURE_FILE")
        if awk -v v="$avg10" -v t="$PRESSURE_THRESHOLD" 'BEGIN{exit !(v < t)}'; then
            return 0
        fi
        if [ "$waited" -ge "$PRESSURE_MAX_WAIT" ]; then
            echo "memory pressure still elevated (avg10=${avg10}) after ${waited}s, proceeding anyway"
            return 0
        fi
        echo "memory pressure elevated (avg10=${avg10} >= ${PRESSURE_THRESHOLD}), waiting 30s before (re)trying..."
        sleep 30
        waited=$((waited + 30))
    done
}
#
# Usage: ./run.sh <pipeline-module> [args...]
#   e.g. ./run.sh evaluate_nbdnet
#        ./run.sh train_nbdnet --epochs 20 --wishart-weight 0.05
#        RETRIES=10 ./run.sh prepare_training_data
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ $# -lt 1 ]; then
    echo "usage: $0 <pipeline-module> [args...]" >&2
    exit 1
fi

module="$1"
shift
retries="${RETRIES:-1}"

mkdir -p logs
log_file="logs/$(date +%Y%m%d_%H%M%S)_${module}.log"

echo "running pipeline.${module} $* (isolated scope, log: ${log_file}, retries: ${retries})"
attempt=0
while true; do
    attempt=$((attempt + 1))
    wait_for_low_pressure
    echo "--- attempt ${attempt}/${retries} ---" | tee -a "${log_file}"
    systemd-run --user --scope -p MemoryMax=40G -- \
        python -u -m "pipeline.${module}" "$@" 2>&1 | tee -a "${log_file}"
    status="${PIPESTATUS[0]}"
    if [ "$status" -eq 0 ]; then
        echo "pipeline.${module} finished successfully (attempt ${attempt})"
        exit 0
    fi
    if [ "$attempt" -ge "$retries" ]; then
        echo "pipeline.${module} failed after ${attempt} attempt(s), giving up" >&2
        exit 1
    fi
    echo "pipeline.${module} died (exit ${status}), will wait for memory pressure to drop before retrying..." | tee -a "${log_file}"
done
