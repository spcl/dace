#!/bin/bash
#SBATCH --job-name=dace-canon-perf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4      # 4 ranks/node by default -- see note below on sizing this
#SBATCH --cpus-per-task=72       # cores-per-node / ntasks-per-node -- run `nproc --all` on a
                                 # compute node to check its core count first and adjust
#SBATCH --time=06:00:00
#SBATCH --partition=normal
#SBATCH --account=g34
#SBATCH --output=canon_%j.out
#SBATCH --error=canon_%j.err
#SBATCH --chdir=/capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs
#
# Example SLURM job: runs the canonicalize performance regression across all
# 3 corpora, distributed over nodes * ntasks-per-node ranks. Each srun'd task
# is a separate OS process, so engine.get_world_rank()/get_world_size()
# (which read SLURM_PROCID/SLURM_NTASKS) pick the right kernel slice
# automatically -- no manual splitting needed here, and dace's cache=unique
# policy (keyed on PID) means no two tasks' compiles ever collide even
# though they share this node's filesystem.
#
# A segfaulting SDFG, a crashing compile, or a failing transform inside any
# one kernel/lane is caught by engine.run_isolated's forked measurement (see
# engine.py) and only skips that one measurement -- it never kills the rank's
# srun task, and one rank dying never affects the others.
#
# Submit with:  sbatch example_slurm_canon_job.sh
# Adjust --nodes / --ntasks-per-node for however many ranks (X) you want.

set -euo pipefail
cd /capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs

export OMP_NUM_THREADS="72"
export PYTHONUNBUFFERED=1  # otherwise stdout is fully buffered (not a tty), so progress prints
                           # don't show up in the log until a buffer fills -- looks like a hang

export PYTHONUSERBASE=/capstor/scratch/cscs/$USER/aarch64/python
export PATH=$PYTHONUSERBASE/bin:$PATH
#python3.11 -m venv /capstor/scratch/cscs/$USER/aarch64/venvs/myenv  # one-time setup; scratch can get purged
source /capstor/scratch/cscs/$USER/aarch64/venvs/myenv/bin/activate
alias python=python3.11

spack load gcc@16.1.0
spack load llvm@22.1.5

# --cpu-bind=cores keeps each rank pinned to its own allocated cores instead
# of letting the OS scheduler migrate/share them across ranks.
srun --cpu-bind=cores python3 tsvc2_perf.py --reps 100 --cxx=clang++
srun --cpu-bind=cores python3 tsvc2_5_perf.py --reps 100 --cxx=clang++
srun --cpu-bind=cores python3 npbench_polybench_perf.py --reps 100 --cxx=clang++

# This sbatch script's own body (as opposed to the srun'd tasks above) runs
# once, on the job's driver process -- exactly where the tables should be
# rebuilt, once, after every rank above has finished (write_tables just
# re-scans the whole results tree, so it doesn't matter which process runs
# this, only that it runs after all ranks are done).
python3 tsvc2_perf.py --tables-only
python3 tsvc2_5_perf.py --tables-only
python3 npbench_polybench_perf.py --tables-only
