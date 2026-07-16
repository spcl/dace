#!/bin/bash
#SBATCH --job-name=dace-verify-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --time=00:25:00
#SBATCH --partition=debug
#SBATCH --account=g34
#SBATCH --output=/capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs/verify_gpu/verify_gpu_%j.out
#SBATCH --error=/capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs/verify_gpu/verify_gpu_%j.err
#SBATCH --chdir=/capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs
#
# GPU-path validation (debug node, 1 GH200): does `run_perf.py --device gpu` build, run and VALIDATE
# the dace lanes on the Hopper GPU? Runs canon_vs on 3 CPU-validated, GPU-friendly npbench kernels
# (arc_distance / azimint_naive / compute). Each dace-GPU candidate is checked against the CPU parallel
# oracle (adapters.check_dace_job keeps the reference on CPU). Success = dace-autoopt/canon/parallel
# rows with device=gpu, correct=True and a median time. Writes to a throwaway results dir (does NOT
# touch results/).

set -uo pipefail
cd /capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs

# --- MPI anti-hang + threading (mirror slurm_perf.sh) ---
export OMP_NUM_THREADS="72" OPENBLAS_NUM_THREADS="72"
export OMP_PROC_BIND="close"
export OMP_PLACES="cores"
export MPI4PY_RC_INITIALIZE="0"
export OMPI_MCA_pml="ob1"
export OMPI_MCA_btl="self,vader"
export UCX_VFS_ENABLE="n"
export PYTHONUNBUFFERED=1

export PYTHONUSERBASE=/capstor/scratch/cscs/$USER/aarch64/python
export PATH=$PYTHONUSERBASE/bin:$PATH
source /capstor/scratch/cscs/$USER/aarch64/venvs/myenv/bin/activate

spack load gcc@16.1.0
spack load llvm@22.1.5
spack load cmake
spack load openblas threads=pthreads
spack load cuda
spack load cutensor

# CUDA + cuTENSOR paths (spack load sets PATH, not LD_LIBRARY_PATH/CPATH) -- DaCe's nvcc discovery.
export CUDA_HOME="$(spack location -i cuda 2>/dev/null || echo "${CUDA_HOME:-}")"
if [ -n "$CUDA_HOME" ]; then
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
    export CPATH="$CUDA_HOME/include:${CPATH:-}"
fi
_CUTENSOR_ROOT="$(spack location -i cutensor 2>/dev/null || true)"
if [ -n "$_CUTENSOR_ROOT" ]; then
    for _d in "$_CUTENSOR_ROOT"/lib "$_CUTENSOR_ROOT"/lib64; do
        [ -d "$_d" ] && export LD_LIBRARY_PATH="$_d:${LD_LIBRARY_PATH:-}" LIBRARY_PATH="$_d:${LIBRARY_PATH:-}"
    done
    [ -d "$_CUTENSOR_ROOT/include" ] && export CPATH="$_CUTENSOR_ROOT/include:${CPATH:-}"
fi

# Runtime library paths for the CPU-side lanes' .so (OpenMP + spack-gcc C++ runtime).
_add_ldpath() { case ":${LD_LIBRARY_PATH:-}:" in *":$1:"*) ;; *) [ -d "$1" ] && export LD_LIBRARY_PATH="$1:${LD_LIBRARY_PATH:-}" ;; esac; }
_ldpath_for() { command -v "$1" >/dev/null 2>&1 || return 0; local _cc="$1"; shift; for _lib in "$@"; do _p="$("$_cc" -print-file-name="$_lib" 2>/dev/null)"; [ -f "$_p" ] && _add_ldpath "$(dirname "$_p")"; done; }
_ldpath_for clang++ libomp.so libgomp.so
_ldpath_for g++ libgomp.so libstdc++.so.6 libgcc_s.so.1
_ldpath_for gcc libstdc++.so.6 libgcc_s.so.1
unset -f _add_ldpath _ldpath_for
export HPTT_ROOT="$PWD/hptt"
[ -f "$HPTT_ROOT/lib/libhptt.so" ] && export LD_LIBRARY_PATH="$HPTT_ROOT/lib:${LD_LIBRARY_PATH:-}"

# Hopper GH200 is sm_90; DaCe's cuda_arch defaults to 60 (Pascal) -> override or nvcc targets the
# wrong ISA. This is the one GPU-specific knob the CPU job never needed.
export DACE_compiler_cuda_cuda_arch=90

# CUDA 13.3's nvcc rejects host gcc > 15 ("unsupported GNU version"), so the spack gcc 16.1.0 used for
# the CPU sweep cannot be nvcc's host compiler. Point both the DaCe C++ compiler AND nvcc's host
# compiler (CUDAHOSTCXX -> CMAKE_CUDA_HOST_COMPILER) at the system gcc 14, which CUDA 13.3 supports.
GPU_HOST_CXX=/usr/bin/g++-14
export CUDAHOSTCXX="$GPU_HOST_CXX"

# Streams: DaCe owns stream management/scheduling; for now pin everything to the single default stream
# (stream 0), no cross-kernel concurrency. max_concurrent_streams=0 is DaCe's default; set explicitly.
export DACE_compiler_cuda_max_concurrent_streams=0
# GPU default flags: always -O3 (device via -Xptxas, host via -Xcompiler) and fast-math (plus
# -march=native for the host glue). No bare -O3 -- nvcc only takes device/host opt via -Xptxas/-Xcompiler.
export DACE_compiler_cuda_args="-Xptxas -O3 -Xcompiler -O3 -Xcompiler -march=native --use_fast_math -Xcompiler -Wno-unused-parameter"

echo "[gpu] CUDA_HOME=$CUDA_HOME"
echo -n "[gpu] nvcc: "; command -v nvcc && nvcc --version 2>/dev/null | tail -1
echo "[gpu] nvidia-smi:"; nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader 2>&1 | head
python3 -c "import dace; print('[gpu] DaCe cuda_arch =', dace.Config.get('compiler','cuda','cuda_arch'))"

RD=/capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs/verify_gpu/results
rm -rf "$RD"; mkdir -p "$RD"
KERNELS="arc_distance,azimint_naive,compute"

echo "###################### canon_vs --device gpu (npbench: $KERNELS) ######################"
srun --ntasks=1 --cpu-bind=cores python3 run_perf.py \
    --experiment canon_vs --corpus npbench --device gpu --kernels "$KERNELS" \
    --reps 3 --force --cxx="$GPU_HOST_CXX" --timeout 600 --results-dir "$RD"
python3 run_perf.py --experiment canon_vs --corpus npbench --tables-only --results-dir "$RD"

echo "############################## GPU RESULT SUMMARY ##############################"
python3 - "$RD" <<'PY'
import csv, os, sys, collections
f = os.path.join(sys.argv[1], 'npbench', 'summary.csv')
if not os.path.isfile(f):
    print("NO summary.csv"); sys.exit()
rows = list(csv.DictReader(open(f)))
gpu = [r for r in rows if r['device'] == 'gpu']
print(f"total rows {len(rows)}, gpu-device rows {len(gpu)}")
per = collections.defaultdict(lambda: [0, 0])
for r in gpu:
    ok = r['correct'].strip().lower() == 'true'
    per[r['pipeline']][0] += ok; per[r['pipeline']][1] += 1
for pl, (o, t) in sorted(per.items()):
    print(f"  {pl:18s} {o}/{t} correct")
print("--- dace-GPU cells (device / correct / median_ms / speedup_vs_numpy) ---")
for r in gpu:
    if r['pipeline'].startswith('dace-'):
        print(f"  {r['kernel']:16s} {r['pipeline']:14s} dev={r['device']} correct={r['correct']:5s} "
              f"median_ms={r['median_ms'] or '-':>10} speedup={r['speedup_vs_baseline'] or '-'}")
PY
echo "GPU_VERIFY_DONE"
