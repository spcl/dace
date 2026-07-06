#!/bin/bash
#SBATCH --job-name=dace-canon-perf
#SBATCH --nodes=1
#SBATCH --ntasks=8              # X ranks -- one kernel-slice per task
#SBATCH --cpus-per-task=4       # matches OMP_NUM_THREADS below (also sizes the
                                 # native GCC autopar lane's -ftree-parallelize-loops)
#SBATCH --time=04:00:00
#SBATCH --output=canon_%j.out
#SBATCH --error=canon_%j.err
#
# Example SLURM job: runs the canonicalize performance regression across all
# 3 corpora, distributed over $SLURM_NTASKS ranks. Each srun'd task is a
# separate OS process, so engine.get_world_rank()/get_world_size() (which read
# SLURM_PROCID/SLURM_NTASKS) pick the right kernel slice automatically -- no
# manual splitting needed here, and dace's cache=unique policy (keyed on PID)
# means no two tasks' compiles ever collide even though they share this node's
# filesystem.
#
# A segfaulting SDFG, a crashing compile, or a failing transform inside any
# one kernel/lane is caught by engine.run_isolated's forked measurement (see
# engine.py) and only skips that one measurement -- it never kills the rank's
# srun task, and one rank dying never affects the others.
#
# Submit with:  sbatch example_slurm_canon_job.sh
# Adjust --ntasks to however many ranks (X) you want to spread the sweep over.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

srun python3 tsvc2_perf.py --reps 100
srun python3 tsvc2_5_perf.py --reps 100
srun python3 npbench_polybench_perf.py --reps 100

# This sbatch script's own body (as opposed to the srun'd tasks above) runs
# once, on the job's driver process -- exactly where the tables should be
# rebuilt, once, after every rank above has finished (write_tables just
# re-scans the whole results tree, so it doesn't matter which process runs
# this, only that it runs after all ranks are done).
python3 tsvc2_perf.py --tables-only
python3 tsvc2_5_perf.py --tables-only
python3 npbench_polybench_perf.py --tables-only
