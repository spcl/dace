#!/bin/bash
#SBATCH --job-name=cloudsc-codegen-ab
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=cloudsc-codegen-ab-%j.out
#
# cloudsc code generation, old analysis path vs new, on the same input.
#
#   tests/perf/run_cloudsc_codegen_ab.sh [base-ref] [new-ref] [reps] [outdir] [corpus-ref] [harness-ref]
#   tests/perf/run_cloudsc_codegen_ab.sh origin/main origin/rm-redundant-getattr-hasattr 5 /scratch/ab origin/extended
#
# corpus-ref supplies tests/corpus, harness-ref supplies tests/perf (default: the new ref, which
# is where these scripts live). They are separate because a ref often has one and not the other.
#
# Submit with sbatch, or just run it -- the SBATCH lines are inert outside SLURM.
#
# What it measures, per ref, on the SAME cloudsc SDFG:
#   cloudsc_simplify              initial simplify on the freshly built graph
#   cloudsc_unroll                LoopUnroll on the simplified graph
#   cloudsc_codegen_after_unroll  generate_code on the unrolled graph  <-- the interesting one
#
# Only Python time is measured. The C++ toolchain is untouched by these passes and is 90%+ of a
# full compile, so including it would bury the signal.
#
# Both arms take tests/corpus AND tests/perf from <harness-ref> (default: extended), so they run the
# identical workload with identical measurement code. That is also what makes an upstream base ref
# runnable: neither directory exists there.
#
# REQUIREMENT: corpus-ref must contain the fix that stops tests/corpus/cloudsc/cloudsc.py from
# importing ScalarToSymbolPromotion/SplitArray at module scope. Without it the corpus drags in
# dace.transformation.layout, which does not exist upstream, and the base arm cannot import
# cloudsc at all.
set -euo pipefail

BASE_REF=${1:-origin/main}
NEW_REF=${2:-origin/rm-redundant-getattr-hasattr}
REPS=${3:-5}
OUTDIR=${4:-$PWD/cloudsc_codegen_ab}
CORPUS_REF=${5:-extended}
# The scripts live on the branch under test; the corpus lives on `extended`. They are
# separate refs because a ref can easily have one and not the other.
HARNESS_REF=${6:-$NEW_REF}

REPO=$(git rev-parse --show-toplevel)
mkdir -p "$OUTDIR"

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

    # git worktree does not populate submodules, and without these every C++ build fails.
    for sub in cub moodycamel; do
        rm -rf "$wt/dace/external/$sub"
        ln -s "$REPO/dace/external/$sub" "$wt/dace/external/$sub"
    done

    # Identical workload and identical measurement code on both arms.
    checkout_path "$wt" "$CORPUS_REF" tests/corpus
    checkout_path "$wt" "$HARNESS_REF" tests/perf

    echo "=== $label: $ref @ $(git -C "$wt" rev-parse --short HEAD) ==="
    ( cd "$wt" && PYTHONPATH="$wt" DACE_default_build_folder="$OUTDIR/dc-$label" \
        python tests/perf/bench_codegen_scopes.py \
            --label "$label" --out "$OUTDIR/$label.csv" --reps "$REPS" --cloudsc-only ) \
        2>&1 | tee "$OUTDIR/$label.log"
}

run_one base "$BASE_REF"
run_one new "$NEW_REF"

python "$REPO/tests/perf/plot_codegen_scopes.py" \
    --base "$OUTDIR/base.csv" --new "$OUTDIR/new.csv" \
    --out-md "$OUTDIR/cloudsc_speedup.md" --out-plot "$OUTDIR/cloudsc_speedup.png"

echo
echo "results in $OUTDIR:"
echo "  base.csv new.csv        raw medians per stage"
echo "  cloudsc_speedup.md      table + geometric mean"
echo "  cloudsc_speedup.png     plot"
echo
cat "$OUTDIR/cloudsc_speedup.md"

git -C "$REPO" worktree prune
