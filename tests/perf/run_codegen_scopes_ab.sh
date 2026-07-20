#!/bin/bash
# Compare optimization and codegen wall time between two checkouts.
#
#   tests/perf/run_codegen_scopes_ab.sh <base-ref> <new-ref> [reps] [outdir] [harness-ref]
#   tests/perf/run_codegen_scopes_ab.sh origin/main HEAD 5 /scratch/ab extended
#
# Builds a throwaway worktree per ref, then in each one:
#   * links the git submodules (worktrees do NOT get them, and without
#     dace/external/{cub,moodycamel} everything that compiles C++ fails)
#   * checks tests/corpus and tests/perf out of <harness-ref>, so BOTH arms use the identical
#     workload AND the identical measurement code. Neither is the thing under test; letting them
#     vary per branch would compare two different benchmarks. It is also what makes a base ref
#     that predates these files (e.g. origin/main) runnable at all.
# Then runs the benchmark in each and writes markdown tables and a plot.
#
# cloudsc caveat: tests/corpus/cloudsc imports dace.transformation.layout, so it can only be built
# on refs that carry that subsystem. Against an upstream base ref the run still produces the
# npbench/polybench workloads and simply reports no cloudsc rows.
#
# Run it on an idle node: the numbers are medians of wall time and a busy box moves them by tens of
# percent.
set -euo pipefail

BASE_REF=${1:?usage: $0 <base-ref> <new-ref> [reps] [outdir]}
NEW_REF=${2:?usage: $0 <base-ref> <new-ref> [reps] [outdir]}
REPS=${3:-5}
OUTDIR=${4:-$PWD/codegen_scopes_ab}
CORPUS_REF=${5:-extended}
# The scripts live on the branch under test; the corpus lives on `extended`. They are
# separate refs because a ref can easily have one and not the other.
HARNESS_REF=${6:-$NEW_REF}

REPO=$(git rev-parse --show-toplevel)
mkdir -p "$OUTDIR"

# Anti-hang settings for anything that pulls in mpi4py.
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader UCX_VFS_ENABLE=n MPI4PY_RC_INITIALIZE=0
export DACE_compiler_use_cache=0

checkout_path() {
    local wt=$1 ref=$2 path=$3
    if ! git -C "$wt" cat-file -e "$ref:$path" 2>/dev/null; then
        echo "ERROR: '$path' does not exist in ref '$ref'." >&2
        echo "       Pass a ref that has it, e.g. a branch where it is committed AND pushed." >&2
        echo "       corpus-ref (arg 5) supplies tests/corpus; harness-ref (arg 6) supplies tests/perf." >&2
        exit 1
    fi
    git -C "$wt" checkout "$ref" -- "$path"
}

run_one() {
    local label=$1 ref=$2 wt="$OUTDIR/wt-$1"

    rm -rf "$wt"
    git -C "$REPO" worktree add -q --detach "$wt" "$ref"
    for sub in cub moodycamel; do
        rm -rf "$wt/dace/external/$sub"
        ln -s "$REPO/dace/external/$sub" "$wt/dace/external/$sub"
    done

    # Identical workload and identical measurement code on both arms.
    checkout_path "$wt" "$CORPUS_REF" tests/corpus
    checkout_path "$wt" "$HARNESS_REF" tests/perf

    echo "=== $label ($ref @ $(git -C "$wt" rev-parse --short HEAD)) ==="
    ( cd "$wt" && PYTHONPATH="$wt" DACE_default_build_folder="$OUTDIR/dc-$label" \
        python tests/perf/bench_codegen_scopes.py --label "$label" --out "$OUTDIR/$label.csv" --reps "$REPS" ) \
        2>&1 | tee "$OUTDIR/$label.log"
}

run_one base "$BASE_REF"
run_one new "$NEW_REF"

python "$REPO/tests/perf/plot_codegen_scopes.py" \
    --base "$OUTDIR/base.csv" --new "$OUTDIR/new.csv" \
    --out-md "$OUTDIR/speedup.md" --out-plot "$OUTDIR/speedup.png"

echo
echo "results in $OUTDIR:"
echo "  base.csv new.csv   raw medians"
echo "  speedup.md         markdown tables + geometric means"
echo "  speedup.png        per-workload speedup plot"

git -C "$REPO" worktree prune
