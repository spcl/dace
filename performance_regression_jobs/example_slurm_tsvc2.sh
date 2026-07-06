#!/bin/bash
#SBATCH --job-name=dace-tsvc2-perf
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4      # 4 ranks/node by default -- see note below on sizing this
#SBATCH --cpus-per-task=8        # cores-per-node / ntasks-per-node -- check your node's core
                                 # count first (see "checking cpus per node" below) and adjust
#SBATCH --time=04:00:00
#SBATCH --output=tsvc2_%j.out
#SBATCH --error=tsvc2_%j.err
#
# Example SLURM job: TSVC2 (single-core DaCe-seq vs. native-gcc/-clang/-icpx/
# -nvc, multi-core DaCe-par vs. each vendor's own autopar) canonicalize
# performance regression, distributed over nodes * ntasks-per-node ranks total.
#
# Each srun'd task is a separate OS process; engine.get_world_rank()/
# get_world_size() read SLURM_PROCID/SLURM_NTASKS and self-select an even
# kernel slice -- no manual splitting needed. dace's cache=unique policy
# (keyed on PID) means no two ranks' compiles collide even sharing a node's
# filesystem, and each rank compiles its own rank-suffixed native .so once.
#
# A segfaulting SDFG, a crashing compile, or a failing transform inside any
# one kernel/lane is caught by engine.run_isolated's forked measurement and
# only skips that one measurement -- it never kills the rank's srun task, and
# one rank dying never affects the others.
#
# Sizing --ntasks-per-node: TSVC2 has 151 kernels x 16 lanes each (8 DaCe --
# baseline/auto-opt/canon/fast-canon x par/seq -- + 8 native -- 4 vendors x
# serial/autopar, though a vendor with no compiler installed just skips its
# 2 lanes); more ranks per node means more
# concurrent native compiles/DaCe builds competing for the same node's CPUs
# during the (short) build phase, but a shorter sweep overall. 4/node is a
# safe default -- raise it if your nodes have enough cores that
# --cpus-per-task would otherwise stay comfortably above 1.
#
# Submit with:  sbatch example_slurm_tsvc2.sh
# Adjust --nodes / --ntasks-per-node for however many ranks (X) you want.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# By default DaCe's own C++ codegen uses clang++ if it's on PATH, else g++
# (plain PATH lookup -- load/activate whatever module or spack env puts the
# compiler you want on PATH first) -- add --cxx <name-or-abs-path> below to
# pin a specific compiler instead (e.g. --cxx g++ or --cxx /path/to/clang++).
# DaCe-lane results are namespaced by compiler+hostname (see
# engine.result_tag), so mixing compilers/nodes in one --results-dir never
# corrupts a resumed sweep.
#
# Separately, every native-gcc(-autopar)/native-clang(-polly-autopar)/
# native-icpx(-autopar)/native-nvc(-autopar) lane finds its OWN vendor's
# compiler on PATH (plain PATH lookup, same as above) -- --cxx does not
# affect these lanes at all. A vendor with no compiler installed is just
# skipped for that lane; native results are namespaced only by hostname
# (engine.native_result_tag), so they aren't needlessly re-measured every
# time --cxx changes.
srun python3 tsvc2_perf.py --reps 100

# Runs once on the job's driver process, after every rank above has finished
# writing its slice -- write_tables just re-scans the whole results tree.
python3 tsvc2_perf.py --tables-only
