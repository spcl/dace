#!/bin/bash
#SBATCH --job-name=readable-perf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4      # phase 1 uses all 4 (one rank per Grace socket); phases 2 and 3
                                 # run a single rank inside the same allocation (each srun below
                                 # sets its own --ntasks, so this is just the upper bound).
#SBATCH --cpus-per-task=72       # one Grace CPU (72 Neoverse-V2 cores) per rank; 4 x 72 = 288
                                 # = the whole GH200 node.
#SBATCH --gpus-per-node=4        # phase 3 binds one Hopper GPU; the node's 4 are reserved so the
                                 # allocation is exclusive.  TODO: drop on a CPU-only partition.
#SBATCH --hint=nomultithread     # Grace Neoverse-V2 has no SMT (1 thread/core)
#SBATCH --time=03:00:00          # ~1.4x the observed 2h07m full run (job 4219591)
#SBATCH --partition=normal       # TODO: set to your daint.alps partition
#SBATCH --account=g34            # daint.alps account
#SBATCH --output=readable_perf_%j.out
#SBATCH --error=readable_perf_%j.err
#SBATCH --chdir=/capstor/scratch/cscs/ybudanaz/aarch64/dace/cpu_codegen_perf_jobs  # TODO: set scratch path
#
# THE readable-codegen perf job: every phase of the legacy-vs-experimental C++/CUDA codegen
# comparison, on ONE node, in ONE submission, into ONE merged TSV.
#
# Every phase runs the SAME pipeline -- dace + simplify + LoopToMap + MapFusion +
# ConvertLengthOneArraysToScalars(transient_only=True) -- and measures
# only AFTER it; the ONLY thing that varies within a phase is the code generator
# (compiler.cpu.implementation = legacy | experimental). Phases, in execution order:
#
#   1. single_core  4 ranks (one per Grace socket), S preset, OMP_NUM_THREADS=1, both codegens,
#                   all four corpora. Each rank measures a disjoint round-robin slice
#                   kernels[rank::4] (every runner self-partitions off SLURM_PROCID/SLURM_NTASKS).
#                   Run TWICE: once g++, once clang++.
#   2. multi_core   1 rank, paper preset, all 72 cores of one Grace socket (OMP_NUM_THREADS=72),
#                   both codegens, all four corpora. Run TWICE: once g++, once clang++.
#   3. gpu          1 rank + 1 GPU, paper preset, BOTH CPU-codegen options (legacy and
#                   experimental) combined with the GPU codegen. nvcc is always the device
#                   compiler; the host compiler is recorded in the `cxx` column. Runs ONCE:
#                   nvcc dictates its host compiler (see GPU_HOST_CXX below), so a g++/clang++
#                   sweep is not available here. npbench+polybench only -- the two tsvc runners
#                   are CPU-only (no --target).
#
# => 5 sweeps: (single_core, g++), (single_core, clang++), (multi_core, g++),
#              (multi_core, clang++), (gpu, nvcc + host g++-14).
#
# Every row of every sweep lands in ONE merged TSV ($OUT, default perfresults.tsv), header:
#
#   kernel corpus codegen preset threads cxx phase codegen_ms compile_ms runtime_ms speedup correctness error
#
# `phase` (single_core|multi_core|gpu), `cxx` (the host compiler) and `corpus`
# (npbench|polybench|tsvc2|tsvc2_5) are what keep the merged rows groupable.
#
# Overridable with `sbatch --export=ALL,REPS=3,...`:
#   CODEGEN       legacy | experimental | both     (default both)
#   REPS          timed reps per lane              (default 10)
#   OUT           merged TSV path                  (default perfresults.tsv)
#   CPU_CXX_LIST  host compilers for phases 1-2    (default "g++ clang++")
#   GPU_HOST_CXX  nvcc's host compiler for phase 3 (default /usr/bin/g++-14)
#   CPP_STANDARD  C++ standard, digits only        (default 20 -> consteval; 17 -> constexpr)
#   METRICS_CSV   readability metrics CSV for plot (default codegen_metrics.csv)
#
# Requires the sibling performance_regression_jobs harness next to this folder (shared
# engine / native_harness / adapters), with its submodules initialized:
#     git submodule update --init dace/external/moodycamel dace/external/cub
# The tsvc corpus harness itself is vendored here (tsvc2_perf.py, tsvc2_5_perf.py,
# tsvc_corpus.py, tsvc_2_5_corpus.py).
#
# NOTE (cmake SIGCHLD hang): srun/mpirun start every task with SIGCHLD BLOCKED in the inherited
# signal mask. cmake's configure step forks helper processes and reaps them via select(); with
# SIGCHLD blocked it is never woken, so cmake spins forever. DaCe guards this in
# dace/codegen/compiler.py::_build_subprocess_sigmask() (compiler.py:375-404, used at
# compiler.py:907-911), which unblocks SIGCHLD ONLY around the build Popen -- so every build here
# must run under a plain `srun ... python3 ...` that inherits that unblocked-around-fork behavior.
# Do NOT wrap the launch in anything that re-blocks SIGCHLD or intercepts child signals: no
# `preexec_fn`, no trap/signal handler on SIGCHLD, no `setsid`/`stdbuf`-style shims around the
# python process, and no nested `srun` inside the build. Plain srun only.

REPO_ROOT="/capstor/scratch/cscs/ybudanaz/aarch64/dace"   # TODO: set to the synced repo root
cd "$REPO_ROOT/cpu_codegen_perf_jobs"

CODEGEN="${CODEGEN:-both}"
REPS="${REPS:-10}"
OUT="${OUT:-perfresults.tsv}"
CPU_CXX_LIST="${CPU_CXX_LIST:-g++ clang++}"
GPU_HOST_CXX="${GPU_HOST_CXX:-/usr/bin/g++-14}"
CPP_STANDARD="${CPP_STANDARD:-20}"
METRICS_CSV="${METRICS_CSV:-codegen_metrics.csv}"

# Cores per rank = one Grace socket. Phase 1 gives each of its 4 ranks a whole socket but only
# ONE OpenMP thread: the ranks then never share cores or a memory controller, so a single-core
# measurement is not perturbed by its 3 neighbours.
SOCKET_CORES="${SLURM_CPUS_PER_TASK:-72}"

# DaCe finds its own local `dace` package via PYTHONPATH; the runners add the sibling
# performance_regression_jobs to sys.path themselves (relative to their own location).
export PYTHONPATH="$REPO_ROOT"

# MPI anti-hang (must match the runners' own os.environ.setdefault block) + unbuffered stdout
# so progress prints appear live under srun (not a tty).
export MPI4PY_RC_INITIALIZE="0"
export OMPI_MCA_pml="ob1"
export OMPI_MCA_btl="self,vader"
export UCX_VFS_ENABLE="n"
export PYTHONUNBUFFERED=1

# C++ standard, pinned from ONE knob so the two builds a lane measures agree:
#   - DaCe's CMake build (compiler.cpp_standard) -> the runtime/correctness binaries, AND
#   - engine.compile_sdfg_timed's DIRECT compile (reads DACE_PERF_CXX_STD) -> the codegen_ms /
#     compile_ms numbers.
# Leaving them unset would measure compile time at engine's own default (c++23) while the runtime
# binary built at DaCe's default (c++20) -- a standard mismatch inside one comparison. The readable
# generator emits `consteval` size/index helpers under C++20+ and degrades to `constexpr` under
# C++17 (dace/codegen/targets/experimental_cpu.py::size_qualifier); default 20 keeps consteval on.
# nvcc's host compilation inherits the same standard, so phase 3 stays consistent too.
export DACE_compiler_cpp_standard="$CPP_STANDARD"
export DACE_PERF_CXX_STD="c++$CPP_STANDARD"

# Python env + toolchain (mirrors performance_regression_jobs/slurm_perf.sh). TODO: adjust
# to your daint.alps module/venv setup.
export PYTHONUSERBASE=/capstor/scratch/cscs/$USER/aarch64/python
export PATH=$PYTHONUSERBASE/bin:$PATH
#python3.11 -m venv /capstor/scratch/cscs/$USER/aarch64/venvs/myenv  # one-time setup; scratch can get purged
source /capstor/scratch/cscs/$USER/aarch64/venvs/myenv/bin/activate
alias python=python3.11

spack load gcc@16.1.0
spack load llvm@22.1.5
spack load cmake
spack load openblas threads=pthreads
spack load cuda        # phase 3 only, but loading once up front keeps PATH/LD_LIBRARY_PATH
spack load cutensor    # identical across phases (a no-op for the CPU ones)

# OpenBLAS (BLAS+LAPACK+LAPACKE) discovery + runtime path: `spack load` sets PATH but not
# LD_LIBRARY_PATH, and the install sits off the ldconfig cache -- without this, MatMul/potrf
# report "OpenBLAS not installed" and expand to a naive pure loop. Also lets any LAPACK-node
# kernel link via DACE_PERF_LAPACK_LIBDIR (engine.configure_dace_process).
export OPENBLAS_DIR="$(spack location -i openblas threads=pthreads 2>/dev/null || echo "${OPENBLAS_DIR:-}")"
if [ -n "$OPENBLAS_DIR" ]; then
    for _d in "$OPENBLAS_DIR"/lib "$OPENBLAS_DIR"/lib64; do
        [ -d "$_d" ] && export LD_LIBRARY_PATH="$_d:${LD_LIBRARY_PATH:-}" LIBRARY_PATH="$_d:${LIBRARY_PATH:-}"
        [ -d "$_d" ] && export DACE_PERF_LAPACK_LIBDIR="$_d"
    done
    [ -d "$OPENBLAS_DIR/include" ] && export CPATH="$OPENBLAS_DIR/include:${CPATH:-}"
fi

# CUDA + cuTENSOR discovery (phase 3): `spack load` sets PATH but not LD_LIBRARY_PATH/CPATH;
# point DaCe's nvcc discovery (CUDA_HOME) and cuTENSOR at the spack installs.
export CUDA_HOME="$(spack location -i cuda 2>/dev/null || echo "${CUDA_HOME:-}")"
if [ -n "$CUDA_HOME" ]; then
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
    export CPATH="$CUDA_HOME/include:${CPATH:-}"
fi
CUTENSOR_ROOT="$(spack location -i cutensor 2>/dev/null || true)"
if [ -n "$CUTENSOR_ROOT" ]; then
    for _d in "$CUTENSOR_ROOT"/lib "$CUTENSOR_ROOT"/lib64; do
        [ -d "$_d" ] && export LD_LIBRARY_PATH="$_d:${LD_LIBRARY_PATH:-}" LIBRARY_PATH="$_d:${LIBRARY_PATH:-}"
    done
    [ -d "$CUTENSOR_ROOT/include" ] && export CPATH="$CUTENSOR_ROOT/include:${CPATH:-}"
fi

# OpenMP + libstdc++ runtime dirs on LD_LIBRARY_PATH so every compiled .so loads via ctypes.
_add_ldpath() { case ":${LD_LIBRARY_PATH:-}:" in *":$1:"*) ;; *) [ -d "$1" ] && export LD_LIBRARY_PATH="$1:${LD_LIBRARY_PATH:-}" ;; esac; }
_ldpath_for() { command -v "$1" >/dev/null 2>&1 || return 0; local _cc="$1"; shift; for _l in "$@"; do _p="$("$_cc" -print-file-name="$_l" 2>/dev/null)"; [ -f "$_p" ] && _add_ldpath "$(dirname "$_p")"; done; }
_ldpath_for clang++ libomp.so libgomp.so
_ldpath_for g++ libgomp.so libstdc++.so.6 libgcc_s.so.1
_ldpath_for gcc libstdc++.so.6 libgcc_s.so.1
unset -f _add_ldpath _ldpath_for

# HPTT (High-Performance Tensor Transpose) is an EXTERNAL, out-of-tree dependency (not vendored;
# see performance_regression_jobs/.gitignore). The npbench/polybench TensorTranspose nodes find its
# headers (<HPTT_ROOT>/include) and link+load its lib via DaCe's HPTT library environment (HPTT_ROOT).
# Point it at the sibling harness's local build if present; guarded, so a no-op when HPTT is not built.
export HPTT_ROOT="$REPO_ROOT/performance_regression_jobs/hptt"
[ -f "$HPTT_ROOT/lib/libhptt.so" ] && export LD_LIBRARY_PATH="$HPTT_ROOT/lib:${LD_LIBRARY_PATH:-}"

# Every per-phase TSV stem, collected for the final merge.
BASES=()

# Short filename tag for a compiler path/name ("g++" -> gcc, "/usr/bin/clang++-19" -> clang):
# `+` is awkward in globs, so the output files use this instead of the raw compiler name.
cxx_tag() { case "$1" in *clang*) echo clang ;; *) echo gcc ;; esac; }

# Drop any stale per-rank / merged TSV for these stems so a re-run never merges old rows in.
clear_stale() { local f; for f in "$@"; do rm -f "${f%.tsv}".rank*.tsv "$f"; done; }

# One CPU phase (all four corpora, both codegens) at one host compiler.
#   run_cpu_phase <phase> <preset> <ntasks> <threads> <cxx>
run_cpu_phase() {
    local phase="$1" preset="$2" ntasks="$3" threads="$4" cxx="$5"
    local tag npb t2 t25
    tag="$(cxx_tag "$cxx")"
    npb="${phase}_${tag}_npbench_polybench.tsv"
    t2="${phase}_${tag}_tsvc2.tsv"
    t25="${phase}_${tag}_tsvc2_5.tsv"
    clear_stale "$npb" "$t2" "$t25"

    # engine.configure_dace_process reads DACE_PERF_CXX -> compiler.cpu.executable; the runners
    # read it back for the `cxx` column, so the label can never disagree with the compiler used.
    export DACE_PERF_CXX="$cxx"
    export OMP_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS="$threads"
    export OMP_PROC_BIND=close   # keep each rank's thread(s) within its own socket
    export OMP_PLACES=cores

    echo "=== phase=$phase cxx=$cxx preset=$preset ranks=$ntasks OMP_NUM_THREADS=$threads codegen=$CODEGEN reps=$REPS"

    # Each rank runs the three runners in turn on its own round-robin slice and writes
    # <base>.rank<r> (a single-rank phase writes <base> itself -- the merge globs both).
    # The inner shell has no `set -e`, so one runner failing on one rank does not skip the
    # rest; the trailing `|| echo` keeps the outer `set -e` from skipping later phases.
    srun --ntasks="$ntasks" --cpus-per-task="$SOCKET_CORES" --cpu-bind=verbose,cores --distribution=block:block \
        bash -c "
            python3 run_readable_perf.py --preset $preset --target cpu --codegen $CODEGEN --corpus both \
                --reps $REPS --phase $phase --out $npb
            python3 run_readable_tsvc2_perf.py --preset $preset --codegen $CODEGEN \
                --reps $REPS --phase $phase --out $t2
            python3 run_readable_tsvc2_5_perf.py --preset $preset --codegen $CODEGEN \
                --reps $REPS --phase $phase --out $t25
        " || echo "warning: phase=$phase cxx=$cxx returned nonzero on some rank; keeping its completed rows"

    BASES+=("$npb" "$t2" "$t25")
}

# The GPU phase: 1 rank + 1 GPU, both CPU-codegen options combined with the GPU codegen.
run_gpu_phase() {
    local cxx="$GPU_HOST_CXX"
    local out="gpu_$(cxx_tag "$cxx")_npbench_polybench.tsv"
    clear_stale "$out"

    # GH200 GPU codegen: Hopper sm_90, single default stream, always-on -O3 + fast-math. nvcc's
    # host compiler must be g++<=14 (CUDA rejects the spack gcc 16 used for the CPU phases), so
    # DACE_PERF_CXX (the DaCe host-side codegen compiler, and the `cxx` column) is pinned to the
    # same g++ for consistency -- this is why phase 3 is not swept over g++/clang++.
    export DACE_compiler_cuda_cuda_arch=90
    export DACE_compiler_cuda_max_concurrent_streams=0
    export DACE_compiler_cuda_args="-Xptxas -O3 -Xcompiler -O3 -Xcompiler -march=native --use_fast_math -Xcompiler -Wno-unused-parameter"
    export CUDAHOSTCXX="$cxx"
    export DACE_PERF_CXX="$cxx"
    export OMP_NUM_THREADS="$SOCKET_CORES" OPENBLAS_NUM_THREADS="$SOCKET_CORES"   # host-side OpenMP for any map left on the host
    export OMP_PROC_BIND=close
    export OMP_PLACES=cores

    echo "=== phase=gpu device_cxx=nvcc host_cxx=$cxx preset=paper ranks=1 codegen=$CODEGEN reps=$REPS"

    # tsvc2/tsvc2_5 are CPU-only runners (no --target), so the GPU phase is npbench+polybench.
    srun --ntasks=1 --cpus-per-task="$SOCKET_CORES" --gpus-per-task=1 \
        python3 run_readable_perf.py --preset paper --target gpu --codegen "$CODEGEN" --corpus both \
            --reps "$REPS" --phase gpu --out "$out" \
        || echo "warning: phase=gpu returned nonzero; keeping its completed rows"

    BASES+=("$out")
}

rm -f "$OUT"

# ---- execution order: phase 1 (g++, clang++), phase 2 (g++, clang++), phase 3 -------------
for cxx in $CPU_CXX_LIST; do
    run_cpu_phase single_core S 4 1 "$cxx"
done
for cxx in $CPU_CXX_LIST; do
    run_cpu_phase multi_core paper 1 "$SOCKET_CORES" "$cxx"
done
run_gpu_phase

# ---- merge every phase's every rank into ONE header + rows --------------------------------
echo "merging all phases into $OUT"
python3 - "$OUT" "${BASES[@]}" <<'PYMERGE'
"""Concatenate every per-rank TSV of every phase under ONE header.

Each stem is either <stem>.rank<r>.tsv (a multi-rank phase, which self-partitions) or
<stem>.tsv (a single-rank phase, which does not add the suffix) -- glob both. Every runner
emits the identical 13-column header, so a header mismatch means a stale file from an older
run: report it loudly rather than silently writing misaligned rows.
"""
import glob
import os
import sys

out = sys.argv[1]
bases = sys.argv[2:]

parts = []
for base in bases:
    stem, ext = os.path.splitext(base)
    parts += sorted(glob.glob(f'{stem}.rank*{ext}'))
    if os.path.isfile(base):
        parts.append(base)

if not parts:
    print('no per-rank TSVs found to merge')
    sys.exit(0)

header, rows, mismatched = None, [], []
for path in parts:
    with open(path) as f:
        lines = f.read().splitlines()
    if not lines:
        continue
    if header is None:
        header = lines[0]
    elif lines[0] != header:
        mismatched.append(path)
        continue
    rows.extend(lines[1:])

with open(out, 'w') as f:
    if header:
        f.write(header + '\n')
    f.write('\n'.join(rows) + ('\n' if rows else ''))

print(f'merged {len(parts) - len(mismatched)} rank file(s), {len(rows)} rows -> {out}')
if mismatched:
    print(f'WARNING: skipped {len(mismatched)} file(s) with a different header: {", ".join(mismatched)}')
PYMERGE

# ---- readability metrics for FIGURE B's LoC + readability panels --------------------------
# The perf sweep already emitted the generated C++ frames; harvest them into a metrics CSV so
# plot_codegen_perf.py can draw the LoC / nloc / max_nesting panels without a second build.
#
# Where the sources are: engine.configure_dace_process redirects DaCe's build folder to node-local
# RAM (/dev/shm/dace_perf_jobs_<uid>_rank<r>), so the persistent frame of each variant lives at
# <root>/<sdfg_name>/src/cpu/<sdfg_name>.cpp. We take the paper (multi_core) CPU frames -- the
# writeup lane, matching the plot's default build cell -- and skip: the *_gpu_* frames (this is the
# CPU-readability score), the *_timed variants (instrumented, extra timer boilerplate) and the
# *_compilebench probe. The generated C++ is host-compiler-independent, so the single kept frame
# per (kernel, codegen) is the readable form for BOTH g++ and clang++.
# 1-node job: this batch step runs on the same node as the srun tasks, so /dev/shm is still theirs.
echo "collecting readability metrics into $METRICS_CSV"
readable_srcs=$(find "/dev/shm/dace_perf_jobs_$(id -u)_rank"* \
                    -path '*paper*/src/cpu/*.cpp' \
                    ! -path '*_gpu_*' ! -path '*_timed*' ! -path '*_compilebench*' 2>/dev/null || true)
if [ -n "$readable_srcs" ]; then
    # shellcheck disable=SC2086  # word-splitting is intentional: one CLI arg per source file.
    python3 readability_metrics.py $readable_srcs --csv "$METRICS_CSV" \
        || echo "warning: readability_metrics.py returned nonzero; $METRICS_CSV may be partial"
else
    echo "readability: no persistent paper CPU frames under /dev/shm -- FIGURE B's LoC/readability "
    echo "readability: panels will be skipped. Re-point readability_metrics.py at a source tree, or "
    echo "readability: pass --srcdir to plot_codegen_perf.py, to draw them."
fi

# ---- how to plot the two figures ----------------------------------------------------------
echo "done: $OUT"
echo "plot both figures with:"
echo "  python3 plot_codegen_perf.py --tsv $OUT --metrics-csv $METRICS_CSV --out-dir plots"
echo "  -> plots/runtime_<phase>.{png,pdf}   (FIGURE A: runtime, g++/clang++ CPU + GPU)"
echo "  -> plots/build_and_quality.{png,pdf} (FIGURE B: codegen+compile time, LoC, readability)"
