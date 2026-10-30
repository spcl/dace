# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
# The CPU half of run_experiment.sh, sourced by it: the numbers on one schedule, then the time of every OpenMP schedule.
#
# Levels are what a CPU team shares. Stages cover different levels (nlev in full, or only nflatlev:nlev), so a thread must
# get levels in every stage, and neighboring levels on one thread keep the vertical stencils local. The chunks are 0 for
# equal blocks per stage, 1, 2, and the team's share of nlev.
NLEV=90
CHUNKS=${CHUNKS:-$(printf '%s\n' 0 1 2 $(((NLEV + OMP_NUM_THREADS - 1) / OMP_NUM_THREADS)) | sort -un | xargs)}

build_all cpu strict --strict
for chunk in $CHUNKS; do
  build_all cpu "release_static$chunk" --chunk "$chunk"
done
mkdir -p "$OUT/verify" "$OUT/time"
for kind in $KINDS; do
  for entry in "${VARIANTS[@]}"; do
    read -r lvn istep steps <<<"$entry"
    name=cpu_${kind}_l${lvn}i${istep}
    run "$OUT/strict/$name" "$steps" 2 both "$OUT/verify/$name.log"
    for chunk in $CHUNKS; do
      run "$OUT/release_static$chunk/$name" "$steps" "$REPS" dace "$OUT/time/${name}_static$chunk.log"
    done
  done
done
