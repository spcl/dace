#!/bin/bash
#SBATCH --job-name=dace-tsvc2-5-perf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4      # 4 ranks/node by default -- see note below on sizing this
#SBATCH --cpus-per-task=72       # cores-per-node / ntasks-per-node -- run `nproc --all` on a
                                 # compute node to check its core count first and adjust
#SBATCH --time=06:00:00
#SBATCH --partition=normal
#SBATCH --account=g34
#SBATCH --output=tsvc2_5_%j.out
#SBATCH --error=tsvc2_5_%j.err
#SBATCH --chdir=/capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs
#
# Example SLURM job: TSVC2.5 canonicalize performance regression, distributed
# over nodes * ntasks-per-node ranks total. Same isolation/timing/crash
# handling as tsvc2 (see slurm_tsvc2.sh for the full rationale) --
# this file only differs in which script it srun's.
#
# Submit with:  sbatch slurm_tsvc2_5.sh
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

# --cxx <name-or-abs-path> pins a specific compiler for DaCe's own codegen
# (default: clang++ on PATH, else g++ -- plain PATH lookup); DaCe-lane
# results are namespaced by compiler+hostname automatically. The native
# lanes (native-clang(-polly-autopar)) are unaffected by --cxx -- they find
# their own clang++ independently, and are just skipped if it isn't on PATH
# (see slurm_tsvc2.sh for the full rationale).
#
# --cpu-bind=cores keeps each rank pinned to its own allocated cores instead
# of letting the OS scheduler migrate/share them across ranks -- with many
# ranks per node this avoids extra scheduling contention on top of whatever
# the workload itself needs.
srun --cpu-bind=cores python3 tsvc2_5_perf.py --reps 25 --cxx=clang++

python3 tsvc2_5_perf.py --tables-only
