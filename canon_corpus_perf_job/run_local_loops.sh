#!/bin/bash
# Local CPU driver: tsvc + tsvc_2_5.
#
# Usage:
#   cd /home/primrose/Work/dace
#   bash canon_corpus_perf_job/run_local_loops.sh

set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
cd "$REPO"

export PYENV_VERSION="${PYENV_VERSION:-py12}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONHASHSEED=0
export DACE_JOB_SCRATCH="${DACE_JOB_SCRATCH:-/dev/shm/dace-corpus-perf}"
PRESET="${PRESET:-S}"

export CANON_PERF_ARMS="${CANON_PERF_ARMS:-0}"

OUT="${OUT_DIR:-$HERE/corpus_perf_results_loops_${PRESET}}"

echo "=== loops sweep: OMP=$OMP_NUM_THREADS preset=$PRESET -> $OUT"
python3 - "$HERE" "$OUT" "$PRESET" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from pathlib import Path
import subprocess

here = Path(sys.argv[1])
out = Path(sys.argv[2])
preset = sys.argv[3]

subprocess.run([
    sys.executable, str(here / 'corpus_perf_job.py'),
    '--facet', 'perf',
    '--preset', preset,
    '--suites', 'loops',
    '--out', str(out),
], check=True)
PY

echo "=== done: $OUT"
