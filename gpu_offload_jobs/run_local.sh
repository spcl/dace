#!/bin/bash
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#
# The same two sweeps on this machine, no Slurm. One rank, one GPU.
#
#   ./run_local.sh cpu poly              # offload + validate + emit, no device needed
#   ./run_local.sh gpu poly gemm,doitgen # build, verify, time both arms for two kernels
#
#   ARMS=autoopt,canon ./run_local.sh gpu poly   # canon against DaCe's established pipeline
#
# Env: PRESET (default S here, not paper -- a laptop GPU does not hold the paper shapes), REPEATS,
# ARMS (baseline first; the driver's default compares the taskloop knob).

set -euo pipefail

KIND="${1:?cpu|gpu}"
SUITES="${2:-poly,np}"
ONLY="${3:-}"
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
PRESET="${PRESET:-S}"
OUT="${OUT_DIR:-$HERE/results_local_${KIND}}"
mkdir -p "$OUT"

export PYTHONHASHSEED=0 MPI4PY_RC_INITIALIZE=0 OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp \
       PMIX_MCA_gds=hash UCX_VFS_ENABLE=n HWLOC_COMPONENTS=-gl OMP_NUM_THREADS=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export DACE_default_build_folder="${DACE_default_build_folder:-$HOME/.cache/dace_gpu_offload_local}"

ARGS=(--preset "$PRESET" --suites "$SUITES" --csv "$OUT/rows.csv")
[ -n "${ARMS:-}" ] && ARGS+=(--arms "$ARMS")
[ -n "$ONLY" ] && ARGS+=(--only "$ONLY")
if [ "$KIND" = cpu ]; then
    # No device is touched, so hide it: a stray CUDA context here would only add a teardown that can
    # wedge on a broken driver.
    export CUDA_VISIBLE_DEVICES=
    ARGS+=(--stage codegen)
else
    ARGS+=(--repeats "${REPEATS:-20}")
fi

exec python3 -m tests.corpus.measure_gpu_arms "${ARGS[@]}"
