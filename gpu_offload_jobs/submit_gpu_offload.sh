#!/bin/bash
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#
# The job body, shared by the GPU and the CPU job. Pins the environment and sruns one rank per
# module. submit_daint.sh is the entry point; this is not meant to be run by hand.
#
#   submit_gpu_offload.sh --out <dir> [driver args...]
#
# ⛔ srun execs gpu_offload_job.py DIRECTLY. Do not wrap it in a shell, setsid, nohup or timeout:
# slurmstepd leaves SIGCHLD blocked and only that module's scope can clear it (see the file).

set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)

OUT="$HERE/results"
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --out) OUT="$2"; shift 2 ;;
        *)     ARGS+=("$1"); shift ;;
    esac
done
mkdir -p "$OUT"

# PYTHONHASHSEED pins str hashing, without which transformation order -- and with it the emitted
# program -- changes run to run. The MPI/UCX variables keep one rank's probing off another's.
export PYTHONHASHSEED=0
export MPI4PY_RC_INITIALIZE=0
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp
export PMIX_MCA_gds=hash
export UCX_VFS_ENABLE=n
export HWLOC_COMPONENTS=-gl
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
# One host thread. This sweep times GPU kernels, so a per-rank OpenMP pool only adds jitter to the
# launch path, which for the small kernels here is most of what is being measured.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

srun --cpu-bind=cores --kill-on-bad-exit=1 \
    python3 "$HERE/gpu_offload_job.py" --out "$OUT" "${ARGS[@]}"
