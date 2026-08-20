#!/bin/bash
# Local CPU driver: polybench + npbench.
#
# Usage:
#   cd /home/primrose/Work/dace
#   bash canon_corpus_perf_job/run_local_array.sh
#
# Override env:
#   PYENV_VERSION=py12 OMP_NUM_THREADS=8 PRESET=S bash canon_corpus_perf_job/run_local_array.sh

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

# Only the two DaCe CPU pipelines by default; set CANON_PERF_ARMS=1 for the full baseline table.
export CANON_PERF_ARMS="${CANON_PERF_ARMS:-0}"

OUT="$HERE/corpus_perf_results_array_${PRESET}"

echo "=== array sweep: OMP=$OMP_NUM_THREADS preset=$PRESET -> $OUT"
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
    '--suites', 'array',
    '--out', str(out),
], check=True)
PY

echo "=== done: $OUT"
