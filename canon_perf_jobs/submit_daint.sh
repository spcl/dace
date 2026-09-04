#!/bin/bash
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#
# THE Daint/Alps submitter. One corpus per job, one node = 4 ranks.
#
#   ./submit_daint.sh smoke                 # debug partition, 2 kernels/corpus/rank, ~3 min. RUN FIRST.
#   ./submit_daint.sh tsvc                  # one corpus, 1 node
#   ./submit_daint.sh poly 2                # one corpus, 2 nodes = 8 ranks
#   ./submit_daint.sh all                   # all four corpora, as four independent jobs
#
# A Grace-Hopper node is four modules, each a 72-core Grace CPU with its own H100, so one node is
# 4 ranks x 72 threads with nothing left over. corpus_perf_job.py shards the corpus's kernels
# across the ranks positionally (SLURM_PROCID of SLURM_NTASKS) and gives each a private build
# scratch, so no two ranks share a .dacecache.
#
# One job per corpus rather than one job for a group: each then gets its own queue slot, time limit
# and results directory, so a slow polybench sweep cannot consume the wall clock npbench needed.
#
# The PREFLIGHT is not optional courtesy. A missing package degrades a column SILENTLY: no clang
# kills four arms, no Polly makes the two llvm-autopar arms measure plain -O3 while still printing a
# number, and no OpenBLAS drops the DaCe arms to `pure` BLAS (-20-40x). Set PREFLIGHT=0 only on a
# site that is not CSCS, where the spack specs below do not resolve.
#
# ⛔ DO NOT replace the sbatch below with a direct `srun`. slurmstepd starts every task with SIGCHLD
# BLOCKED; the mask survives fork+exec into cmake, whose KWSys learns its helpers exited by
# RECEIVING SIGCHLD and otherwise waits in select() FOREVER -- a configure that looks stuck but is
# really a lost wakeup. The only place the unblock works is corpus_perf_job.py's module scope,
# because srun execs it directly with no shell in between; submit_corpus_perf.sh preserves that
# chain. A wrapper shell, setsid, nohup, timeout or `env --ignore-signal=CHLD` all bring the hang
# back. The rank re-checks the mask at entry and refuses to start if it is still set, so a broken
# chain fails in the first second instead of after an hour.
#
# Env: PRESET (default paper), OMP_NUM_THREADS (72), TIMELIMIT, ACCOUNT (g34), PARTITION (normal),
#      CANON_PERF_ARMS (1 = the nine-arm table), OUT_DIR, PREFLIGHT=0 to skip the toolchain probe.

set -euo pipefail

WHAT="${1:-}"
NODES="${2:-1}"
HERE=$(cd "$(dirname "$0")" && pwd)
PRESET="${PRESET:-paper}"

case "$WHAT" in
    poly|np|tsvc|tsvc25|all|smoke) ;;
    *)
        echo "usage: $(basename "$0") <smoke|poly|np|tsvc|tsvc25|all> [NODES]" >&2
        echo "  poly, np      -- array corpora, divided by the corpus numpy reference" >&2
        echo "  tsvc, tsvc25  -- loop corpora, divided by seq-cpp" >&2
        echo "  smoke         -- 2 kernels/corpus/rank on the debug partition; run this first" >&2
        exit 2
        ;;
esac

# 72 threads is one rank's whole Grace CPU. Set here AND passed to srun as --cpus-per-task, because
# Slurm binds the task to that many cores and OpenMP has to agree with the binding, or the four
# ranks oversubscribe each other's cores and every timing is noise.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-72}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export PYTHONHASHSEED=0
export CANON_PERF_ARMS="${CANON_PERF_ARMS:-1}"

if [ "${PREFLIGHT:-1}" = "1" ]; then
    export SPACK_ROOT="${SPACK_ROOT:-/capstor/scratch/cscs/ybudanaz/aarch64/spack}"
    . "$SPACK_ROOT/share/spack/setup-env.sh"

    die() {
        echo "FATAL: $*" >&2
        exit 1
    }

    TMP=$(mktemp -d)
    trap 'rm -rf "$TMP"' EXIT

    # Spack's own stderr IS the diagnosis and must survive: an ambiguous spec ("matches multiple
    # packages" -- this tree already holds two gcc@14.2.0) is a different problem from a missing one,
    # and discarding the message reports both as "not installed".
    spack_prefix() {
        local out
        out=$(spack location -i "$@" 2>"$TMP/spack.err") || die "spack location -i $*: $(cat "$TMP/spack.err")"
        printf '%s\n' "$out"
    }

    # Variant pins because both packages are installed twice: the other gcc@16.1.0 is ~graphite (kills
    # the autopar arms) and the other llvm@22.1.5 is ~mlir. Per-COMPILER OpenBLAS: %clang links libomp,
    # %gcc links libgomp, so no arm's BLAS drags the other OpenMP runtime into its process.
    spack_prefix gcc@16.1.0 +graphite >/dev/null
    spack_prefix llvm@22.1.5 +mlir >/dev/null
    # Pinned to the exact compiler VERSION each arm builds with: openblas now exists for both
    # gcc@14.2.0 and gcc@16.1.0, so a bare %gcc is ambiguous and resolves to nothing.
    CANON_PERF_OPENBLAS_GCC=$(spack_prefix openblas threads=openmp %gcc@16.1.0)
    CANON_PERF_OPENBLAS_LLVM=$(spack_prefix openblas threads=openmp %llvm@22.1.5)
    export CANON_PERF_OPENBLAS_GCC CANON_PERF_OPENBLAS_LLVM

    spack load gcc@16.1.0 +graphite
    spack load llvm@22.1.5 +mlir
    g++ --version | head -1 | grep -q '16\.1\.0' || die "g++ on PATH is not 16.1.0: $(g++ --version | head -1)"
    clang++ --version | head -1 | grep -q '22\.1\.5' || die "clang++ on PATH is not 22.1.5"

    # Polly must EMIT a region, not merely accept the flag: a clang without Polly takes it silently.
    # BOTH measured llvm arms are proven here, each on the shape it needs: forced Polly parallelizes a
    # flat 1-D loop, while the -default arm declines one at any trip count and takes a 2-D nest.
    POLLY=(-mllvm -polly -mllvm -polly-parallel -mllvm -polly-omp-backend=LLVM -mllvm -polly-process-unprofitable)
    printf 'void k(double*__restrict__ a,double*__restrict__ b,long n){for(long i=0;i<n;++i)a[i]=b[i]+1.0;}\n' >"$TMP/flat.cpp"
    printf 'void k(double*__restrict__ a,double*__restrict__ b,double*__restrict__ c){for(int i=0;i<4096;i++)for(int j=0;j<4096;j++)c[i*4096+j]=a[i*4096+j]+b[i*4096+j]*3.0;}\n' >"$TMP/nest.cpp"
    polly_probe() {
        local arm=$1 src=$2
        shift 2
        clang++ -O3 -fopenmp "${POLLY[@]}" "$@" -c "$TMP/$src.cpp" -o "$TMP/$src.o" 2>/dev/null ||
            die "clang++ rejected the $arm flag set"
        nm -u "$TMP/$src.o" | grep -q __kmpc_fork_call ||
            die "$arm emitted no parallel region on the $src probe -- this clang has no working Polly"
    }
    polly_probe dace-simplify+llvm-autopar flat -mllvm -polly-parallel-force
    polly_probe dace-simplify+llvm-autopar-default nest

    echo "== preflight OK"
    echo "   $(g++ --version | head -1)"
    echo "   $(clang++ --version | head -1)"
    echo "   openblas gcc arms:  $CANON_PERF_OPENBLAS_GCC"
fi

submit_one() {
    local suite=$1 nodes=$2 deftime
    case "$suite" in
        poly)   deftime=04:00:00 ;;   # 30 kernels, but the paper preset makes them the big ones
        np)     deftime=04:00:00 ;;   # 24 kernels; softmax alone peaks around 12 GiB
        tsvc)   deftime=06:00:00 ;;   # 151 kernels, small each
        tsvc25) deftime=04:00:00 ;;   # 72 kernels
    esac
    local out="${OUT_DIR:-$HERE/results_${suite}_${PRESET}}"
    mkdir -p "$out"
    sbatch --job-name="dace-perf-${suite}" \
        --account="${ACCOUNT:-g34}" \
        --partition="${PARTITION:-normal}" \
        --exclusive \
        --time="${TIMELIMIT:-$deftime}" \
        --nodes="$nodes" \
        --ntasks-per-node=4 \
        --cpus-per-task="$OMP_NUM_THREADS" \
        --output="$out/dace-perf-${suite}-%j.out" \
        --error="$out/dace-perf-${suite}-%j.out" \
        "$HERE/submit_corpus_perf.sh" "$nodes" --suites "$suite" --preset "$PRESET" --facet perf --out "$out"
}

if [ "$WHAT" = "smoke" ]; then
    # Two kernels per corpus per rank on the debug partition. --force so it re-times kernels an
    # earlier run already has results for; otherwise a smoke on a populated directory measures
    # nothing and passes.
    PARTITION=debug TIMELIMIT=00:25:00 CANON_PERF_TIMEOUT=240 OUT_DIR="$HERE/results_smoke" \
        bash "$HERE/submit_corpus_perf.sh" 1 --suites all --facet perf --limit 2 --force
elif [ "$WHAT" = "all" ]; then
    for suite in poly np tsvc tsvc25; do
        echo "=== submitting $suite (${NODES} node(s) x 4 ranks x ${OMP_NUM_THREADS} threads)"
        submit_one "$suite" "$NODES"
    done
    echo
    echo "watch:  squeue --me"
else
    submit_one "$WHAT" "$NODES"
fi
