#!/bin/bash
# Local, single-node, single-rank runner for the CODEGEN BUILD perf comparison (codegen time,
# C++ compile time, generated-code size + readability metrics INLINE, plus runtime), CPU-only,
# on THIS machine. No SLURM / sbatch: run it directly
#     ./run_local_buildperf.sh
# Mirrors the daint sbatch's CPU phases, single-rank:
#   * single_core : preset S     (small dataset), OMP_NUM_THREADS=1
#   * multi_core  : preset paper (full dataset),  OMP_NUM_THREADS=THREADS  (default 8)
#
# Knobs (env-overridable):
#   PHASES    which phases to run                    (default "single_core multi_core")
#   THREADS   OpenMP threads for the multi_core lane (default 8; single_core is always 1)
#   CODEGEN   legacy | experimental | both           (default both)
#   REPS      timed reps per lane (best-of)          (default 5)
#   CXX       host compiler                          (default g++; try clang++)
#   CPP_STD   C++ standard, digits only              (default 20)
#   ONLY      substring filter on kernel name        (default none)
#   PYTHON    python interpreter                     (default python3)
#   OUT       merged TSV path                        (default local/buildperfresults_local.tsv)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$HERE")"
cd "$REPO_ROOT/codegen_buildperf_jobs"

THREADS="${THREADS:-8}"
CODEGEN="${CODEGEN:-both}"
REPS="${REPS:-5}"
CXX="${CXX:-g++}"
CPP_STD="${CPP_STD:-20}"
PYTHON="${PYTHON:-python3}"
ONLY="${ONLY:-}"
PHASES="${PHASES:-single_core multi_core}"
OUT="${OUT:-$HERE/buildperfresults_local.tsv}"

export PYTHONPATH="$REPO_ROOT"
export DACE_PERF_CXX="$CXX"
export DACE_compiler_cpp_standard="$CPP_STD"
export DACE_PERF_CXX_STD="c++$CPP_STD"
export MPI4PY_RC_INITIALIZE=0 OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader UCX_VFS_ENABLE=n
export PYTHONUNBUFFERED=1
export OMP_PROC_BIND=close OMP_PLACES=cores
unset SLURM_PROCID SLURM_NTASKS 2>/dev/null || true
RANK_ARGS=(--rank 0 --num-ranks 1)
[ -n "$ONLY" ] && RANK_ARGS+=(--only "$ONLY")

if ! command -v "$CXX" >/dev/null 2>&1; then
    echo "error: compiler '$CXX' not found on PATH" >&2; exit 1
fi

echo "== local codegen build-perf run =="
echo "   repo=$REPO_ROOT  cxx=$CXX  c++$CPP_STD  phases='$PHASES'  codegen=$CODEGEN  reps=$REPS  multi_core threads=$THREADS"

BASES=()
run_phase() {
    local phase="$1" preset threads
    case "$phase" in
        single_core) preset=S;     threads=1 ;;
        multi_core)  preset=paper; threads="$THREADS" ;;
        *) echo "unknown phase '$phase' (want single_core|multi_core)"; return 1 ;;
    esac
    export OMP_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS="$threads"
    local out="$HERE/${phase}_buildperf.tsv"
    rm -f "$out"
    echo "--- phase=$phase preset=$preset threads=$threads ---"
    # run_buildperf_all builds ONE combined kernel list spanning all four corpora; single rank sweeps all.
    "$PYTHON" run_buildperf_all.py --preset "$preset" --codegen "$CODEGEN" --reps "$REPS" \
        --phase "$phase" "${RANK_ARGS[@]}" --out "$out" \
        || echo "warning: run_buildperf_all.py (phase=$phase) returned nonzero; keeping completed rows"
    BASES+=("$out")
}

rm -f "$OUT"
for ph in $PHASES; do run_phase "$ph"; done

echo "merging -> $OUT"
"$PYTHON" - "$OUT" "${BASES[@]}" <<'PYMERGE'
import os, sys
out, parts = sys.argv[1], [p for p in sys.argv[2:] if os.path.isfile(p)]
header, rows = None, []
for p in parts:
    lines = open(p).read().splitlines()
    if not lines:
        continue
    if header is None:
        header = lines[0]
    elif lines[0] != header:
        print(f'WARNING: header mismatch in {p}, skipping'); continue
    rows += lines[1:]
with open(out, 'w') as f:
    if header:
        f.write(header + '\n')
    f.write('\n'.join(rows) + ('\n' if rows else ''))
print(f'merged {len(parts)} file(s), {len(rows)} rows -> {out}')
PYMERGE

echo "done: $OUT"
echo "plot with:"
echo "  $PYTHON $REPO_ROOT/codegen_buildperf_jobs/plot_buildperf.py --tsv $OUT --out-dir $HERE/plots"
