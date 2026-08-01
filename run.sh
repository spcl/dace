#!/bin/bash
# Test runner: pinned interpreter, isolated dacecache, MPI env, PYTHONHASHSEED.
# usage: run.sh <tree> <cachetag> <logfile> [pytest args...]
TREE="$1"; TAG="$2"; LOG="$3"; shift 3
export PYTHONPATH="$TREE"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader UCX_VFS_ENABLE=n MPI4PY_RC_INITIALIZE=0
export DACE_default_build_folder="/home/primrose/.cache/mlgpu_dc/$TAG"
mkdir -p "$DACE_default_build_folder"
cd "$TREE" || exit 1
exec /home/primrose/.pyenv/versions/py12/bin/python -m pytest -p no:cacheprovider --maxfail=10 -q "$@" >"$LOG" 2>&1
