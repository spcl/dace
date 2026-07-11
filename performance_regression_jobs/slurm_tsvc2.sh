#!/bin/bash
#SBATCH --job-name=dace-tsvc2-perf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4      # 4 ranks/node by default -- see note below on sizing this
#SBATCH --cpus-per-task=72       # cores-per-node / ntasks-per-node -- run `nproc --all` on a
                                 # compute node to check its core count first and adjust
#SBATCH --time=06:00:00
#SBATCH --partition=normal
#SBATCH --account=g34
#SBATCH --output=tsvc2_%j.out
#SBATCH --error=tsvc2_%j.err
#SBATCH --chdir=/capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs
#
# Example SLURM job: TSVC2 (single-core DaCe-seq vs. native-clang(-polly-
# autopar), multi-core DaCe-par vs. the same) canonicalize performance
# regression, distributed over nodes * ntasks-per-node ranks total.
#
# Each srun'd task is a separate OS process; engine.get_world_rank()/
# get_world_size() read SLURM_PROCID/SLURM_NTASKS and self-select an even
# kernel slice -- no manual splitting needed. dace's cache=name policy (every
# variant gets an explicit, corpus+kernel+pipeline+par/seq-qualified name) means
# no two ranks' compiles collide even sharing a node's filesystem, and each
# rank compiles its own rank-suffixed native .so once.
#
# A segfaulting SDFG, a crashing compile, or a failing transform inside any
# one kernel/lane is caught by engine.run_isolated's forked measurement and
# only skips that one measurement -- it never kills the rank's srun task, and
# one rank dying never affects the others.
#
# Sizing --ntasks-per-node: TSVC2 has 151 kernels x 10 lanes each (8 DaCe --
# baseline/auto-opt/canon/fast-canon x par/seq -- + 2 native -- clang serial
# and Polly-autopar, skipped if clang++ isn't on PATH); more ranks per node
# means more concurrent native compiles/DaCe builds competing for the same
# node's CPUs and filesystem during the build phase, but a shorter sweep
# overall. 4/node is a safe default -- raise it if your nodes have enough
# cores that --cpus-per-task would otherwise stay comfortably above 1.
#
# Submit with:  sbatch slurm_tsvc2.sh
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

# By default DaCe's own C++ codegen uses clang++ if it's on PATH, else g++
# (plain PATH lookup -- load/activate whatever module or spack env puts the
# compiler you want on PATH first) -- add --cxx <name-or-abs-path> below to
# pin a specific compiler instead (e.g. --cxx g++ or --cxx /path/to/clang++).
# DaCe-lane results are namespaced by compiler+hostname (see
# engine.result_tag), so mixing compilers/nodes in one --results-dir never
# corrupts a resumed sweep.
#
# Separately, native-clang(-polly-autopar) finds its own clang++ on PATH
# (plain PATH lookup, same as above) -- --cxx does not affect these lanes at
# all. If clang++ isn't found, both are skipped; native results are
# namespaced only by hostname (engine.native_result_tag), so they aren't
# needlessly re-measured every time --cxx changes.
#
# --cpu-bind=cores keeps each rank pinned to its own allocated cores instead
# of letting the OS scheduler migrate/share them across ranks.
srun --cpu-bind=cores python3 tsvc2_perf.py --reps 100 --cxx=clang++

# Runs once on the job's driver process, after every rank above has finished
# writing its slice -- write_tables just re-scans the whole results tree.
python3 tsvc2_perf.py --tables-only
