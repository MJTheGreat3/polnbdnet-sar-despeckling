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
# Observed on the real box: the kill point moves around run to run (died
# at a different step each time with no code change in between) -- a sign
# of variable external contention (other tenants' jobs) on top of our own
# allocations, not a single deterministic bug to keep chasing in our code.
# Do NOT set RETRIES>1 for train_nbdnet.py -- it has no checkpoint-resume,
# a retry would silently restart training from epoch 0, discarding
# everything the killed run had done.
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
    echo "--- attempt ${attempt}/${retries} ---" | tee -a "${log_file}"
    systemd-run --user --scope -p MemoryMax=30G -- \
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
    echo "pipeline.${module} died (exit ${status}), retrying in 15s..." | tee -a "${log_file}"
    sleep 15
done
