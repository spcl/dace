#!/bin/bash
#SBATCH --job-name=dace-npbench-polybench-gpu-perf
#SBATCH --nodes=2                 # X nodes -- raise for more aggregate throughput
#SBATCH --ntasks-per-node=4       # 4 ranks/node
#SBATCH --gpus-per-task=1         # 1 GPU per rank (4 GPUs/node); SLURM binds each rank
                                  # to its own device via a per-task CUDA_VISIBLE_DEVICES,
                                  # so DaCe's default device 0 IS that rank's GPU
#SBATCH --gpu-bind=single:1       # hard-pin one distinct GPU per rank (no GPU-0 contention)
#SBATCH --cpus-per-task=16        # 4 ranks x 16 = 64 cores/node (fits a 64-core node; raise on bigger)
#SBATCH --time=08:00:00
#SBATCH --partition=normal
#SBATCH --account=g34
#SBATCH --output=npbench_polybench_gpu_%j.out
#SBATCH --error=npbench_polybench_gpu_%j.err
#SBATCH --chdir=/capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs
#
# Example SLURM job: NPBench+PolyBench canonicalize performance regression on the
# GPU. Same script as the CPU sweep (npbench_polybench_perf.py) but restricted to
# the gpu device -- it times dace_parallel and canon vs. the dace_autoopt baseline
# with dace.DeviceType.GPU (auto-opt) and finalize_for_target(..., 'gpu') (canon).
# Host numpy arrays are passed to the compiled SDFG; DaCe inserts the host<->device
# copies (no cupy needed). The numpy timing lane is CPU-only and absent here.
#
# GPU degrades gracefully: with no CUDA toolchain/device a one-time crash-isolated
# probe skips the gpu device entirely, so this job never crashes on a CPU-only node
# (it just measures nothing). Run the CPU sweep (slurm_npbench_polybench.sh)
# for the cpu-device numbers; the two write into the SAME results tree under
# separate '<...>_paper-cpu' / '<...>_paper-gpu' folders.
#
# Submit with:  sbatch slurm_npbench_polybench_gpu.sh

set -euo pipefail
cd /capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs

export OMP_NUM_THREADS="16"   # = --cpus-per-task (4 ranks/node x 16)
export PYTHONUNBUFFERED=1

export PYTHONUSERBASE=/capstor/scratch/cscs/$USER/aarch64/python
export PATH=$PYTHONUSERBASE/bin:$PATH
source /capstor/scratch/cscs/$USER/aarch64/venvs/myenv/bin/activate
alias python=python3.11

spack load gcc@16.1.0
spack load llvm@22.1.5
spack load cuda   # nvcc must be on PATH for DaCe's GPU codegen

# --devices gpu restricts the sweep to the GPU device only.
srun --cpu-bind=cores python3 npbench_polybench_perf.py --reps 100 --devices gpu --cxx=clang++

# Re-scan the whole results tree (cpu + gpu folders) and rebuild the tables + summary.csv.
python3 npbench_polybench_perf.py --tables-only
