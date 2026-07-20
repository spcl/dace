#!/bin/bash
# Optimization and code-generation wall time, old analysis path vs new, in one run.
#
#   tests/perf/run_perf_ab.sh [base-ref] [new-ref] [reps] [outdir] [corpus-ref] [harness-ref]
#
# Workloads, in order, per ref:
#   opt / codegen                 simplify and generate_code over every npbench+polybench kernel
#   cloudsc_simplify              initial simplify on the cloudsc graph
#   cloudsc_unroll                LoopUnroll on the simplified graph
#   cloudsc_codegen_after_unroll  generate_code on the unrolled graph
#
# Only Python time is measured. The C++ toolchain is untouched by these passes and is 90%+ of a full
# compile, so including it would bury the signal; nothing here calls sdfg.compile().
#
# Both arms take tests/corpus from <corpus-ref> and tests/perf from <harness-ref>, so they run the
# identical workload with identical measurement code. That is also what makes an upstream base ref
# runnable at all -- neither directory exists there.
#
# The cloudsc SDFG takes ~10 minutes of frontend parsing to build and is identical for both refs, so
# it is built once and cached; the second arm loads it in seconds.
set -euo pipefail

BASE_REF=${1:-origin/main}
NEW_REF=${2:-origin/rm-redundant-getattr-hasattr}
REPS=${3:-3}
OUTDIR=${4:-$PWD/perf_ab}
CORPUS_REF=${5:-origin/extended}
HARNESS_REF=${6:-$NEW_REF}

REPO=$(git rev-parse --show-toplevel)
mkdir -p "$OUTDIR"

export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader UCX_VFS_ENABLE=n MPI4PY_RC_INITIALIZE=0
export DACE_compiler_use_cache=0
# Shared across both arms: the build is the input, not the thing under test.
export DACE_BENCH_SDFG_CACHE="${DACE_BENCH_SDFG_CACHE:-$OUTDIR/cloudsc.sdfgz}"

checkout_path() {
    local wt=$1 ref=$2 path=$3
    if ! git -C "$wt" cat-file -e "$ref:$path" 2>/dev/null; then
        echo "ERROR: '$path' does not exist in ref '$ref'." >&2
        echo "       corpus-ref (arg 5) supplies tests/corpus; harness-ref (arg 6) supplies tests/perf." >&2
        echo "       Both must be refs where the path is committed AND pushed." >&2
        exit 1
    fi
    git -C "$wt" checkout "$ref" -- "$path"
}

run_one() {
    local label=$1 ref=$2 wt="$OUTDIR/wt-$1"

    # rm -rf leaves git's registration behind, so a rerun collides with the stale entry.
    rm -rf "$wt"
    git -C "$REPO" worktree prune
    git -C "$REPO" worktree add -f -q --detach "$wt" "$ref"

    # git worktree does not populate submodules, and without these every C++ build fails.
    for sub in cub moodycamel; do
        rm -rf "$wt/dace/external/$sub"
        ln -s "$REPO/dace/external/$sub" "$wt/dace/external/$sub"
    done

    checkout_path "$wt" "$CORPUS_REF" tests/corpus
    checkout_path "$wt" "$HARNESS_REF" tests/perf

    echo "=== $label: $ref @ $(git -C "$wt" rev-parse --short HEAD) ==="
    ( cd "$wt" && PYTHONPATH="$wt" DACE_default_build_folder="$OUTDIR/dc-$label" \
        python tests/perf/bench_codegen_scopes.py \
            --label "$label" --out "$OUTDIR/$label.csv" --reps "$REPS" ) \
        2>&1 | tee "$OUTDIR/$label.log"
}

echo "base=$BASE_REF  new=$NEW_REF  reps=$REPS  corpus=$CORPUS_REF  harness=$HARNESS_REF"
echo "outdir=$OUTDIR  sdfg-cache=$DACE_BENCH_SDFG_CACHE"
echo

run_one base "$BASE_REF"
run_one new "$NEW_REF"

python "$REPO/tests/perf/plot_codegen_scopes.py" \
    --base "$OUTDIR/base.csv" --new "$OUTDIR/new.csv" \
    --out-md "$OUTDIR/speedup.md" --out-plot "$OUTDIR/speedup.png"

echo
echo "results in $OUTDIR:"
echo "  base.csv new.csv   raw medians per workload"
echo "  speedup.md         tables + geometric means"
echo "  speedup.png        per-workload plot"
echo
cat "$OUTDIR/speedup.md"

git -C "$REPO" worktree prune
