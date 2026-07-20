#!/bin/bash
#SBATCH --job-name=dace-perf-ab
#SBATCH --account=g34
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=perf_ab/dace-perf-ab-%j.out
#SBATCH --error=perf_ab/dace-perf-ab-%j.out
#
# A/B the DaCe optimization + codegen analysis path, old vs new, on one exclusive node.
#
#   sbatch tests/perf/submit_perf_ab.sh [base-ref] [new-ref] [reps] [outdir] [corpus-ref] [harness-ref]
#   sbatch tests/perf/submit_perf_ab.sh                          # all defaults
#
# Every argument is forwarded untouched to run_perf_ab.sh. Override the SLURM side with
# --account/--time/... on the sbatch command line, which win over the #SBATCH lines above.
#
# Runs directly too (bash tests/perf/submit_perf_ab.sh ...): outside SLURM the #SBATCH lines are
# inert comments, so it just executes the benchmark in the foreground.
set -euo pipefail

REPO=$(git rev-parse --show-toplevel)

# Toolchain: the cloudsc build and any C++ the corpus touches need a current compiler.
# `spack` is a shell function; in a non-interactive sbatch shell it may not be defined yet, so
# source the setup if `spack load` is not already available.
if ! command -v spack >/dev/null 2>&1 && [ -n "${SPACK_ROOT:-}" ]; then
    . "$SPACK_ROOT/share/spack/setup-env.sh"
fi
spack load llvm@22.1.5
spack load gcc@16.1.0

exec "$REPO/tests/perf/run_perf_ab.sh" "$@"
