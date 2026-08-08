#!/bin/bash
# Preflight-gated submission for the corpus sweep.
#
#   ./run_sweep.sh smoke   debug partition, 2 kernels/corpus/rank, ~3 min -- run this FIRST
#   ./run_sweep.sh loops   normal partition, tsvc + tsvc_2_5 (223 kernels) -> results_loops
#   ./run_sweep.sh array   normal partition, polybench + npbench (54)     -> results_array
#
# A missing package degrades a column silently: no clang kills four arms, no OpenBLAS drops the
# DaCe arms to `pure` BLAS (-20-40x) while still printing a number. Hence the preflight.
set -euo pipefail

MODE="${1:-smoke}"
HERE=$(cd "$(dirname "$0")" && pwd)
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
CANON_PERF_OPENBLAS_GCC=$(spack_prefix openblas threads=openmp %gcc)
CANON_PERF_OPENBLAS_LLVM=$(spack_prefix openblas threads=openmp %clang)
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
echo "   openblas llvm arms: $CANON_PERF_OPENBLAS_LLVM"

case "$MODE" in
smoke)
    # 4 kernels a rank (2 corpora x --limit 2), so the per-kernel cap must leave the whole shard
    # inside the 25 min debug limit: 4 x 240s = 16 min, where the old 400s was 26.7 and overran it.
    PARTITION=debug TIMELIMIT=00:25:00 CANON_PERF_TIMEOUT=240 OUT_DIR="$HERE/verify_sync" \
        bash "$HERE/submit_corpus_perf.sh" 1 --suites loops --facet perf --limit 2 --force
    ;;
loops)
    TIMELIMIT=08:00:00 OUT_DIR="$HERE/results_loops" \
        bash "$HERE/submit_corpus_perf.sh" 1 --suites loops --facet perf
    ;;
array)
    TIMELIMIT=08:00:00 OUT_DIR="$HERE/results_array" \
        bash "$HERE/submit_corpus_perf.sh" 1 --suites array --facet perf
    ;;
*)
    die "usage: $0 {smoke|loops|array}"
    ;;
esac
