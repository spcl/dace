#!/bin/bash
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
# One rank of run_beverin_ranks.sbatch: an MI300A node has four APUs, so each rank takes one, with its cores, memory and
# GPU to itself. The ranks do not share what they time.
rank=$SLURM_PROCID
case $rank in
  0) export DEVICES=cpu KINDS="base" ;;
  1) export DEVICES=cpu KINDS="megakernel tuned" ;;
  2) export DEVICES=gpu KINDS="base" ;;
  3) export DEVICES=gpu KINDS="megakernel tuned" ;;
esac
export OUT=$ROOT/rank$rank ROCR_VISIBLE_DEVICES=$rank
mkdir -p "$OUT"
exec numactl --cpunodebind="$rank" --membind="$rank" bash "$HERE/run_experiment.sh" >"$OUT/rank.log" 2>&1
