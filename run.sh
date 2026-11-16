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
#
# Usage: ./run.sh <pipeline-module> [args...]
#   e.g. ./run.sh evaluate_nbdnet
#        ./run.sh train_nbdnet --epochs 20 --wishart-weight 0.05
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ $# -lt 1 ]; then
    echo "usage: $0 <pipeline-module> [args...]" >&2
    exit 1
fi

module="$1"
shift

mkdir -p logs
log_file="logs/$(date +%Y%m%d_%H%M%S)_${module}.log"

echo "running pipeline.${module} $* (isolated scope, log: ${log_file})"
systemd-run --user --scope -p MemoryMax=30G -- \
    python -u -m "pipeline.${module}" "$@" 2>&1 | tee "${log_file}"
