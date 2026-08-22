#!/bin/bash
# Local CPU driver: one corpus (or a group), no Slurm.
#
# Usage:
#   cd /home/primrose/Work/dace
#   bash canon_perf_jobs/run_local.sh <poly|np|tsvc|tsvc25|array|loops|all>
#
# Override env:
#   PYENV_VERSION=py12 OMP_NUM_THREADS=8 PRESET=S bash canon_perf_jobs/run_local.sh <poly|np|tsvc|tsvc25|array|loops|all>

set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
cd "$REPO"

export PYENV_VERSION="${PYENV_VERSION:-py12}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONHASHSEED=0
export DACE_JOB_SCRATCH="${DACE_JOB_SCRATCH:-/dev/shm/dace-corpus-perf}"
SUITE="${1:-all}"
PRESET="${PRESET:-S}"

# Only the two DaCe CPU pipelines by default; set CANON_PERF_ARMS=1 for the full baseline table.
export CANON_PERF_ARMS="${CANON_PERF_ARMS:-0}"

OUT="${OUT_DIR:-$HERE/results_local_${SUITE}_${PRESET}}"

echo "=== $SUITE sweep: OMP=$OMP_NUM_THREADS preset=$PRESET -> $OUT"
python3 - "$HERE" "$OUT" "$PRESET" "$SUITE" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from pathlib import Path
import subprocess

here = Path(sys.argv[1])
out = Path(sys.argv[2])
preset = sys.argv[3]
suite = sys.argv[4]

subprocess.run([
    sys.executable, str(here / 'corpus_perf_job.py'),
    '--facet', 'perf',
    '--preset', preset,
    '--suites', suite,
    '--out', str(out),
], check=True)
PY

echo "=== done: $OUT"
