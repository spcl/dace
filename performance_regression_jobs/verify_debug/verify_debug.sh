#!/bin/bash
#SBATCH --job-name=dace-verify-fix
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --partition=debug
#SBATCH --account=g34
#SBATCH --output=/capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs/verify_debug/verify_%j.out
#SBATCH --error=/capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs/verify_debug/verify_%j.err
#SBATCH --chdir=/capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs
#
# Debug-partition verification of the adapters.py fix across all four corpora.
# Tiny kernel subset per corpus, --reps 1 --force, under srun (exercises the
# SIGCHLD-blocking path too). Shared-scratch results dir. Does NOT touch results/.

set -uo pipefail
cd /capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs

export OMP_NUM_THREADS="72"
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
spack load openblas

_add_ldpath() {
    case ":${LD_LIBRARY_PATH:-}:" in
        *":$1:"*) ;;
        *) [ -d "$1" ] && export LD_LIBRARY_PATH="$1:${LD_LIBRARY_PATH:-}" ;;
    esac
}
_ldpath_for() {
    command -v "$1" >/dev/null 2>&1 || return 0
    local _cc="$1"; shift
    for _lib in "$@"; do
        _p="$("$_cc" -print-file-name="$_lib" 2>/dev/null)"
        [ -f "$_p" ] && _add_ldpath "$(dirname "$_p")"
    done
}
_ldpath_for clang++ libomp.so libgomp.so
_ldpath_for g++    libgomp.so
_ldpath_for g++ libstdc++.so.6 libgcc_s.so.1
_ldpath_for gcc libstdc++.so.6 libgcc_s.so.1
unset -f _add_ldpath _ldpath_for

export HPTT_ROOT="$PWD/hptt"
[ -f "$HPTT_ROOT/lib/libhptt.so" ] && export LD_LIBRARY_PATH="$HPTT_ROOT/lib:${LD_LIBRARY_PATH:-}"

CXX="$(command -v clang++)"
RD=/capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs/verify_debug/results
rm -rf "$RD"; mkdir -p "$RD"

declare -A K=(
  [npbench]="arc_distance,azimint_naive"
  [polybench]="covariance2,correlation,atax"
  [tsvc2]="s000_d_single,s1111_d_single"
  [tsvc2_5]="argmax_value,cond_reduce_sum"
)

for c in npbench polybench tsvc2 tsvc2_5; do
  echo "############################## CORPUS=$c kernels=${K[$c]} ##############################"
  srun --ntasks=1 --cpu-bind=cores python3 run_perf.py \
      --experiment canon_vs --corpus "$c" --kernels "${K[$c]}" \
      --reps 1 --force --cxx="$CXX" --timeout 240 --results-dir "$RD/$c"
done

echo "=================== CORRECTNESS SUMMARY ==================="
python3 - "$RD" <<'PY'
import csv, os, sys, collections
root = sys.argv[1]
for c in ('npbench','polybench','tsvc2','tsvc2_5'):
    f = os.path.join(root, c, c, 'summary.csv')
    if not os.path.isfile(f):
        print(f"[{c}] NO summary.csv"); continue
    rows = list(csv.DictReader(open(f)))
    per = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        ok = r['correct'].strip().lower() == 'true'
        per[r['pipeline']][0] += ok; per[r['pipeline']][1] += 1
    tot_ok = sum(v[0] for v in per.values()); tot = sum(v[1] for v in per.values())
    print(f"[{c}] {tot_ok}/{tot} rows correct")
    for pl,(o,t) in sorted(per.items()):
        print(f"     {pl:20s} {o}/{t}")
PY
echo "ALL_DONE"
