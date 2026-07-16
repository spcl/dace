#!/bin/bash
# Local, single-node, single-rank runner for the readable-codegen RUNTIME + readability perf
# comparison (legacy vs experimental C++ codegen) on THIS machine.
#
# No SLURM / sbatch: run it directly
#     ./run_local.sh
# It runs ONE rank on ONE node, using this machine's compilers and BLAS, and writes ONE merged
# TSV (+ a readability-metrics CSV). Mirrors the daint sbatch's CPU phases, single-rank:
#   * single_core : preset S     (small dataset), OMP_NUM_THREADS=1
#   * multi_core  : preset paper (full dataset),  OMP_NUM_THREADS=THREADS  (default 8)
# Every phase runs the same pipeline (dace + simplify + LoopToMap + MapFusion +
# ConvertLengthOneArraysToScalars) and varies only compiler.cpu.implementation (legacy|experimental).
#
# Knobs (env-overridable, e.g. `PHASES=multi_core REPS=10 CXX=clang++ ./run_local.sh`):
#   PHASES    which phases to run                   (default "single_core multi_core")
#   THREADS   OpenMP threads for the multi_core lane (default 8; single_core is always 1)
#   CODEGEN   legacy | experimental | both          (default both)
#   CORPUS    npbench | polybench | both            (default both)
#   TARGET    cpu | gpu  (gpu -> paper only)         (default cpu)
#   REPS      timed reps per lane (best-of)          (default 5)
#   CXX       host compiler                          (default g++; try clang++)
#   CPP_STD   C++ standard, digits only              (default 20 -> consteval; 17 -> constexpr)
#   ONLY      substring filter on kernel name        (default none; e.g. ONLY=atax)
#   PYTHON    python interpreter                     (default python3)
#   OUT       merged TSV path                        (default local/perfresults_local.tsv)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$HERE")"
cd "$REPO_ROOT/cpu_codegen_perf_jobs"

THREADS="${THREADS:-8}"
CODEGEN="${CODEGEN:-both}"
CORPUS="${CORPUS:-both}"
TARGET="${TARGET:-cpu}"
REPS="${REPS:-5}"
CXX="${CXX:-g++}"
CPP_STD="${CPP_STD:-20}"
PYTHON="${PYTHON:-python3}"
ONLY="${ONLY:-}"
OUT="${OUT:-$HERE/perfresults_local.tsv}"
METRICS_CSV="${METRICS_CSV:-$HERE/codegen_metrics_local.csv}"
# The GPU target is paper-only (single_core is a CPU concept); default both CPU phases otherwise.
if [ "$TARGET" = "gpu" ]; then PHASES="${PHASES:-gpu}"; else PHASES="${PHASES:-single_core multi_core}"; fi

# --- process-wide local env (compiler, C++ std, single rank) -------------------------------
export PYTHONPATH="$REPO_ROOT"
# engine.configure_dace_process reads DACE_PERF_CXX -> compiler.cpu.executable (and the `cxx` column).
export DACE_PERF_CXX="$CXX"
# One knob pins the C++ standard for BOTH the CMake correctness build and engine's direct timed compile.
export DACE_compiler_cpp_standard="$CPP_STD"
export DACE_PERF_CXX_STD="c++$CPP_STD"
export MPI4PY_RC_INITIALIZE=0 OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader UCX_VFS_ENABLE=n
export PYTHONUNBUFFERED=1
export OMP_PROC_BIND=close OMP_PLACES=cores
# One rank, single node (unset any stray SLURM_* so it can't re-partition; pass rank explicitly).
unset SLURM_PROCID SLURM_NTASKS 2>/dev/null || true
RANK_ARGS=(--rank 0 --num-ranks 1)
[ -n "$ONLY" ] && RANK_ARGS+=(--only "$ONLY")
# System OpenBLAS/LAPACK are on the ldconfig cache; the SDFG's own BLAS/LAPACK environments carry the
# link flags (engine resolves a Debian soname to a path / -l<stem>). No DACE_PERF_LAPACK_LIBDIR (a
# removed no-op that force-linked names a single-lib OpenBLAS does not have).

if ! command -v "$CXX" >/dev/null 2>&1; then
    echo "error: compiler '$CXX' not found on PATH" >&2; exit 1
fi

echo "== local readable-codegen perf run =="
echo "   repo=$REPO_ROOT  cxx=$CXX ($($CXX -dumpversion 2>/dev/null))  c++$CPP_STD"
echo "   phases='$PHASES'  target=$TARGET  codegen=$CODEGEN  corpus=$CORPUS  reps=$REPS  multi_core threads=$THREADS"

BASES=()

# run_phase <phase>: pick (preset, threads) by the phase and run the 3 runners on one rank.
run_phase() {
    local phase="$1" preset threads
    case "$phase" in
        single_core) preset=S;     threads=1 ;;
        multi_core)  preset=paper; threads="$THREADS" ;;
        gpu)         preset=paper; threads="$THREADS" ;;
        *) echo "unknown phase '$phase' (want single_core|multi_core|gpu)"; return 1 ;;
    esac
    export OMP_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS="$threads"

    local npb="$HERE/${phase}_npbench_polybench.tsv"
    local t2="$HERE/${phase}_tsvc2.tsv"
    local t25="$HERE/${phase}_tsvc2_5.tsv"
    rm -f "$npb" "$t2" "$t25"

    echo "--- phase=$phase preset=$preset threads=$threads ---"
    "$PYTHON" run_readable_perf.py --preset "$preset" --target "$TARGET" --codegen "$CODEGEN" \
        --corpus "$CORPUS" --reps "$REPS" --phase "$phase" "${RANK_ARGS[@]}" --out "$npb" \
        || echo "warning: run_readable_perf.py (phase=$phase) returned nonzero; keeping completed rows"
    BASES+=("$npb")

    if [ "$TARGET" = "cpu" ]; then
        "$PYTHON" run_readable_tsvc2_perf.py --preset "$preset" --codegen "$CODEGEN" \
            --reps "$REPS" --phase "$phase" "${RANK_ARGS[@]}" --out "$t2" \
            || echo "warning: run_readable_tsvc2_perf.py (phase=$phase) returned nonzero; keeping completed rows"
        "$PYTHON" run_readable_tsvc2_5_perf.py --preset "$preset" --codegen "$CODEGEN" \
            --reps "$REPS" --phase "$phase" "${RANK_ARGS[@]}" --out "$t25" \
            || echo "warning: run_readable_tsvc2_5_perf.py (phase=$phase) returned nonzero; keeping completed rows"
        BASES+=("$t2" "$t25")
    fi
}

rm -f "$OUT"
for ph in $PHASES; do run_phase "$ph"; done

# --- merge every produced TSV under ONE header (single rank -> each base is written direct) ----
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

# --- readability metrics from the persistent paper (multi_core) frames --------------------------
readable_srcs=$(find "/dev/shm/dace_perf_jobs_$(id -u)_rank0" \
                    -path '*paper*/src/cpu/*.cpp' \
                    ! -path '*_gpu_*' ! -path '*_timed*' ! -path '*_compilebench*' 2>/dev/null || true)
if [ -n "$readable_srcs" ]; then
    # shellcheck disable=SC2086
    "$PYTHON" readability_metrics.py $readable_srcs --csv "$METRICS_CSV" \
        || echo "warning: readability_metrics.py returned nonzero; $METRICS_CSV may be partial"
    echo "readability metrics -> $METRICS_CSV"
fi

echo "done: $OUT"
echo "plot with:"
echo "  $PYTHON $REPO_ROOT/cpu_codegen_perf_jobs/plot_codegen_perf.py --tsv $OUT --metrics-csv $METRICS_CSV --out-dir $HERE/plots"
