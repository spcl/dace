#!/bin/bash
# Submit run_perf_ab.sh to SLURM on a normal node with a 2 hour limit.
#
#   tests/perf/submit_perf_ab.sh [base-ref] [new-ref] [reps] [outdir] [corpus-ref] [harness-ref]
#   tests/perf/submit_perf_ab.sh                      # all defaults
#
# Every argument is forwarded untouched to run_perf_ab.sh. Override the SLURM side with the
# environment, e.g. PARTITION=debug TIME=00:30:00 tests/perf/submit_perf_ab.sh
#
# --exclusive is not optional: these are wall-clock medians and a shared node moves them by tens of
# percent, which is larger than the effect being measured.
set -euo pipefail

REPO=$(git rev-parse --show-toplevel)
SCRIPT="$REPO/tests/perf/run_perf_ab.sh"
OUTDIR=${4:-$PWD/perf_ab}

PARTITION=${PARTITION:-normal}
TIME=${TIME:-02:00:00}
JOBNAME=${JOBNAME:-dace-perf-ab}

mkdir -p "$OUTDIR"

sbatch \
    --job-name="$JOBNAME" \
    --partition="$PARTITION" \
    --nodes=1 \
    --exclusive \
    --time="$TIME" \
    --output="$OUTDIR/%x-%j.out" \
    --error="$OUTDIR/%x-%j.out" \
    --wrap="$SCRIPT $*"

echo "submitted; output goes to $OUTDIR/$JOBNAME-<jobid>.out"
echo "watch with: tail -f $OUTDIR/$JOBNAME-*.out"
