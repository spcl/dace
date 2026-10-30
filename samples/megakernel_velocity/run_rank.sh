#!/bin/bash
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
# One rank of run_<machine>_ranks.sbatch: an MI300A node has four APUs and a Daint node four GH200 modules, so each rank
# takes one, with its cores, memory and GPU to itself. The ranks do not share what they time.
rank=$SLURM_PROCID
# RANKS repeats some of them: the ranks left out exit, so each one keeps its module
case " ${RANKS:-0 1 2 3} " in *" $rank "*) ;; *) exit 0 ;; esac
case $rank in
  0) export DEVICES=cpu KINDS="base" ;;
  1) export DEVICES=cpu KINDS="megakernel tuned" ;;
  2) export DEVICES=gpu KINDS="base" ;;
  3) export DEVICES=gpu KINDS="megakernel tuned" ;;
esac
export OUT=$ROOT/rank$rank "${GPU_SELECT:?the variable that picks a GPU here}=$rank"
mkdir -p "$OUT"
# srun on Daint starts tasks with SIGCHLD blocked, children inherit that, and cmake then hangs in its configure step
exec "${PYTHON:-python}" -c '
import os, signal, sys
signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGCHLD})
os.execvp(sys.argv[1], sys.argv[1:])
' numactl --cpunodebind="$rank" --membind="$rank" bash "$HERE/run_experiment.sh" >"$OUT/rank.log" 2>&1
