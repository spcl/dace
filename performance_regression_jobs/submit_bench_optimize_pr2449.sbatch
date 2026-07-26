#!/bin/bash
#SBATCH --job-name=bench-opt-pr2449
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8      # kernels are sliced round-robin across ranks (SLURM_PROCID)
#SBATCH --cpus-per-task=1        # the optimizer is single-threaded Python
#SBATCH --time=02:00:00
#SBATCH --output=bench_opt_pr2449_%j.out
#SBATCH --error=bench_opt_pr2449_%j.err
#
# Optimization-time A/B for PR #2449, over npbench + polybench.
#
# Corpus comes from `extended` (it carries the seeded generators the kernels need); the two
# trees under test are the PR branch and its base. All three are materialised read-only from
# git, so the tree you are sitting in is never touched.
#
# Reps are INTERLEAVED (pr, base, pr, base, ...) so both variants see the same machine load.
#
#   sbatch submit_bench_optimize_pr2449.sbatch      # 10 reps + 100x repeat, the defaults
#   REPS=2 ./submit_bench_optimize_pr2449.sbatch    # runs fine without slurm too
set -euo pipefail

REPO="${REPO:-$(git -C "$(dirname "$(readlink -f "$0")")" rev-parse --show-toplevel)}"
PR_BRANCH="${PR_BRANCH:-rm-getattr-hasattr-sweep}"
BASE_BRANCH="${BASE_BRANCH:-main}"
CORPUS_REF="${CORPUS_REF:-extended}"
REPS="${REPS:-10}"
HEAVY_ITERS="${HEAVY_ITERS:-100}"
HEAVY_KERNEL="${HEAVY_KERNEL:-poly:gemm}"
WORK="${WORK:-$HOME/.cache/dace_bench/pr2449}"   # on disk: /tmp is a RAM disk on some hosts
LAUNCH="${LAUNCH:-$(command -v srun >/dev/null && echo srun || echo '')}"

DRIVER="$(dirname "$(readlink -f "$0")")/bench_optimize_pr2449.py"
RESULTS="$WORK/results"
mkdir -p "$RESULTS"

# Corpus and both trees, materialised once and reused on re-runs.
[[ -d "$WORK/corpus/tests/corpus" ]] || {
    mkdir -p "$WORK/corpus"
    git -C "$REPO" archive "$CORPUS_REF" tests/__init__.py tests/corpus | tar -x -C "$WORK/corpus"
}
for spec in "pr:$PR_BRANCH" "base:$BASE_BRANCH"; do
    variant="${spec%%:*}" ref="${spec#*:}"
    [[ -d "$WORK/tree-$variant/dace" ]] && continue
    git -C "$REPO" worktree add --detach "$WORK/tree-$variant" "$ref" >/dev/null
    # Worktrees do not populate submodules, and `import dace` walks the package.
    for sub in cub moodycamel; do
        rmdir "$WORK/tree-$variant/dace/external/$sub" 2>/dev/null || true
        ln -sfn "$REPO/dace/external/$sub" "$WORK/tree-$variant/dace/external/$sub"
    done
done

run() {  # run <variant> <driver args...>
    local variant="$1"; shift
    # The corpus must precede the tree on PYTHONPATH: both ship a `tests` package and we want
    # the corpus one, while `dace` resolves onward into the tree.
    PYTHONPATH="$WORK/corpus:$WORK/tree-$variant" DACE_default_build_folder="$WORK/dacecache-$variant" \
        $LAUNCH python "$DRIVER" --variant "$variant" "$@"
}

: > "$RESULTS/corpus.csv"
for ((rep = 1; rep <= REPS; rep++)); do
    echo "=== corpus rep $rep/$REPS ==="
    for variant in pr base; do
        run "$variant" --rep "$rep" --mode corpus --out "$RESULTS/corpus.csv" \
            > "$RESULTS/corpus.$variant.$rep.log" 2>&1
    done
done

: > "$RESULTS/heavy.csv"
echo "=== repeat optimization: $HEAVY_KERNEL x $HEAVY_ITERS ==="
for variant in pr base; do
    SLURM_NTASKS=1 run "$variant" --mode heavy --kernel "$HEAVY_KERNEL" --iters "$HEAVY_ITERS" \
        --out "$RESULTS/heavy.csv" > "$RESULTS/heavy.$variant.log" 2>&1
done

PYTHONPATH="$WORK/corpus:$WORK/tree-base" python "$DRIVER" --summarize "$RESULTS/corpus.csv" "$RESULTS/heavy.csv"
echo "raw results in $RESULTS"
