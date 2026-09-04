#!/bin/bash
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#
# SLURM submission for tests/perf/run_cloudsc_backend.sh -- same shape as the existing
# tests/perf/submit_perf_ab.sh (account/partition/exclusive/time, spack toolchain load).
#
#   sbatch tests/perf/submit_cloudsc_backend.sh [reps] [outdir]
#   sbatch tests/perf/submit_cloudsc_backend.sh                  # all defaults (10 reps)
#
# Every argument is forwarded untouched to run_cloudsc_backend.sh. Override the SLURM side
# with --account/--time/... on the sbatch command line, which win over the #SBATCH lines below.
#
# Runs directly too (bash tests/perf/submit_cloudsc_backend.sh ...): outside SLURM the
# #SBATCH lines are inert comments, so it just executes the benchmark in the foreground.

#SBATCH --job-name=cloudsc-graph-backend
#SBATCH --account=g34
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=cloudsc_backend_results/dace-cloudsc-backend-%j.out
#SBATCH --error=cloudsc_backend_results/dace-cloudsc-backend-%j.out

set -euo pipefail

REPO=$(git rev-parse --show-toplevel)

# SBATCH --output writes to cloudsc_backend_results/ relative to the submit directory, so it
# must exist before the job is scheduled -- sbatch does not create it.
mkdir -p cloudsc_backend_results

# Toolchain: the cloudsc build needs a current compiler. `spack` is a shell function; in a
# non-interactive sbatch shell it may not be defined yet, so source the setup if `spack load`
# is not already available.
if ! command -v spack >/dev/null 2>&1 && [ -n "${SPACK_ROOT:-}" ]; then
    . "$SPACK_ROOT/share/spack/setup-env.sh"
fi
spack load llvm@22.1.5
spack load gcc@16.1.0

exec "$REPO/tests/perf/run_cloudsc_backend.sh" "$@"
