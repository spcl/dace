#!/bin/bash
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
# Builds velocity_tendencies with and without the megakernel on one device, checks the numbers against the ICON dumps
# and times both. Sourced by run_<machine>.sbatch, after the machine's environment.
#
# Needs: DACE_FORTRAN_ROOT (dace-fortran checkout), DATA_DIR (data_r02b05, see README.md), OUT (results).
# Optional: MACHINE (the CSV's machine column), REPS (100), DEVICES ("gpu cpu"), BUILD_PARALLEL (2), BASELINES (the SDFGs
# of save_baselines.py, to skip the frontend), GPU_CXX (CXX of GPU builds, for a CXX nvcc cannot host).
set -u
HERE=${HERE:-$(pwd)}
PYTHON=${PYTHON:-python}
REPS=${REPS:-100}
MACHINE=${MACHINE:-$(hostname)}
DEVICES=${DEVICES:-gpu cpu}
BUILD_PARALLEL=${BUILD_PARALLEL:-2}
: "${DACE_FORTRAN_ROOT:?a dace-fortran checkout}" "${DATA_DIR:?the directory of the dumps}" "${OUT:?a results directory}"

# Every (lvn_only, istep) the data set has, with its timesteps; (1, 2) never occurs
VARIANTS=("0 1 1" "0 2 2,9,93,519,1140,1814,6519" "1 1 7,43,463,2593,5701")

# base: one launch per kernel. megakernel: a grid barrier between all stages. tuned: only between dependent stages.
KINDS=${KINDS:-base megakernel tuned}

# build <device> <base|megakernel|tuned> <lvn_only> <istep> <build dir> [build.py flags]: library and driver in the directory
build() {
  local device=$1 kind=$2 lvn=$3 istep=$4 dir=$5
  shift 5
  local flags=("$@")
  [ "$kind" = megakernel ] && flags+=(--megakernel)
  [ "$kind" = tuned ] && flags+=(--megakernel --fewer-syncs)
  [ "$device" = gpu ] && flags+=(--chiplets "${CHIPLETS:-0}")
  [ -n "${BASELINES:-}" ] && flags+=(--baselines "$BASELINES")
  local cxx=${CXX:-c++}
  [ "$device" = gpu ] && cxx=${GPU_CXX:-$cxx}
  CXX=$cxx "$PYTHON" "$HERE/build.py" --out "$dir" --lvn-only "$lvn" --istep "$istep" --device "$device" "${flags[@]}" \
    >"$dir.build.log" 2>&1
  # The library decides, not the exit code: a GPU build can crash in TLS destructors at interpreter exit
  [ -f "$dir/libvelocity_inner_wrap.so" ] || { echo "build of $dir failed, see $dir.build.log" >&2; return 1; }
  grep -E "megakernels" "$dir.build.log"
  (cd "$dir" && "$FC" -O2 -cpp -I. -ffree-line-length-none -fopenmp "$HERE/velocity_driver.F90" -o velocity_driver \
    ./libvelocity_inner_wrap.so sdfg/build/libvelocity_tendencies.so "-Wl,-rpath,$dir" "-Wl,-rpath,$dir/sdfg/build")
}

# run <build dir> <timesteps> <reps> <kernel> <log>
run() {
  (cd "$1" && GFORTRAN_UNBUFFERED_ALL=1 ./velocity_driver "$DATA_DIR" "$2" "$3" "$4") >"$5" 2>&1
}

build_all() {
  local device=$1 tag=$2
  shift 2
  local kind entry count=0
  for kind in $KINDS; do
    for entry in "${VARIANTS[@]}"; do
      read -r lvn istep _ <<<"$entry"
      mkdir -p "$OUT/$tag"
      build "$device" "$kind" "$lvn" "$istep" "$OUT/$tag/${device}_${kind}_l${lvn}i${istep}" "$@" &
      if ((++count % BUILD_PARALLEL == 0)); then wait; fi
    done
  done
  wait
}

verify_and_time() {
  local device=$1 kind entry
  build_all "$device" strict --strict
  build_all "$device" release
  mkdir -p "$OUT/verify" "$OUT/time"
  for kind in $KINDS; do
    for entry in "${VARIANTS[@]}"; do
      read -r lvn istep steps <<<"$entry"
      name=${device}_${kind}_l${lvn}i${istep}
      run "$OUT/strict/$name" "$steps" 2 both "$OUT/verify/$name.log"
      run "$OUT/release/$name" "$steps" "$REPS" dace "$OUT/time/$name.log"
    done
  done
}

export PYTHONHASHSEED=0
ulimit -s unlimited
export OMP_STACKSIZE=1G OMP_PLACES=cores OMP_PROC_BIND=close
for device in $DEVICES; do
  if [ "$device" = cpu ]; then
    . "$HERE/cpu_experiment.sh"
  else
    verify_and_time "$device"
  fi
done
"$PYTHON" "$HERE/summarize.py" "$OUT/verify" | tee "$OUT/verify.txt"
"$PYTHON" "$HERE/summarize.py" "$OUT/time" --csv "$OUT/times.csv" --machine "$MACHINE" | tee "$OUT/time.txt"
