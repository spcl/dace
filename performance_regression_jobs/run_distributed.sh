#!/usr/bin/env bash
# Multi-rank launcher for the performance_regression_jobs scripts.
#
# The scripts self-slice their kernel list via OMPI_COMM_WORLD_RANK/SIZE or
# SLURM_PROCID/SLURM_NTASKS (see engine.py's get_world_rank/get_world_size),
# so this launcher does no enumeration or splitting itself -- it just starts
# N ranks of the same script.
#
# Usage:
#   ./run_distributed.sh <script.py> <num-ranks> [script args...]
#
# Examples:
#   ./run_distributed.sh tsvc2_perf.py 8
#   ./run_distributed.sh npbench_polybench_perf.py 4 --reps 50
#
# Under a real scheduler, prefer its own launcher instead (it already sets
# the rank env vars), e.g.:
#   srun -n 8 python3 tsvc2_perf.py
#   mpirun -np 8 python3 tsvc2_perf.py
set -euo pipefail

script="$1"
num_ranks="$2"
shift 2

if command -v srun &>/dev/null; then
    exec srun -n "$num_ranks" python3 "$script" "$@"
fi
if command -v mpirun &>/dev/null; then
    exec mpirun -np "$num_ranks" python3 "$script" "$@"
fi

# No scheduler available: fall back to plain background processes, setting
# the same rank env vars engine.py already knows how to read.
echo "no srun/mpirun found; running $num_ranks local ranks via OMPI_COMM_WORLD_RANK/SIZE"
pids=()
for ((rank = 0; rank < num_ranks; rank++)); do
    OMPI_COMM_WORLD_RANK="$rank" OMPI_COMM_WORLD_SIZE="$num_ranks" python3 "$script" "$@" &
    pids+=($!)
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=$?
done
exit "$status"
