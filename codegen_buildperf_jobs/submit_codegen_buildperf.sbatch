#!/bin/bash
#SBATCH --job-name=codegen-buildperf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4      # phase 1 (single_core) uses all 4 ranks (one per Grace socket);
                                 # phase 2 (multi_core) runs a single rank inside the same
                                 # allocation (each srun below sets its own --ntasks, so 4 is
                                 # just the upper bound).
#SBATCH --cpus-per-task=72       # one Grace CPU (72 Neoverse-V2 cores) per rank; 4 x 72 = 288
                                 # = the whole node. CPU-only job: no --gpus-per-node.
#SBATCH --hint=nomultithread     # Grace Neoverse-V2 has no SMT (1 thread/core)
#SBATCH --time=06:00:00          # compile-reps heavy, no prior reference -- safe margin
#SBATCH --partition=normal       # TODO: set to your CPU-only daint.alps partition
#SBATCH --account=g34            # TODO: set to your daint.alps account
#SBATCH --output=codegen_buildperf_%j.out
#SBATCH --error=codegen_buildperf_%j.err
#SBATCH --chdir=/capstor/scratch/cscs/ybudanaz/aarch64/dace/codegen_buildperf_jobs  # TODO: scratch path
#
# THE codegen build-and-perf job: every phase of the legacy-vs-experimental readable C++
# codegen comparison, CPU-only, on ONE node, in ONE submission, into ONE merged TSV. It
# measures, for {g++, clang++} x {legacy, experimental} x {npbench, polybench, tsvc2,
# tsvc2_5}, per kernel: BUILD metrics (codegen time, C++ compile time, generated code size in
# bytes/LoC, and the readability metrics nloc/max_nesting/tokens/max_ccn -- all recorded
# INLINE in the same row), plus single-core and multi-core RUNTIME.
#
# Every phase runs the SAME pipeline -- dace + simplify + LoopToMap + MapFusion +
# ConvertLengthOneArraysToScalars(transient_only=True) -- and measures
# only AFTER it; the ONLY thing that varies is the code generator
# (compiler.cpu.implementation = legacy | experimental). Phases, in execution order:
#
#   1. single_core  4 ranks (one per Grace socket), S preset, OMP_NUM_THREADS=1, both codegens.
#                   run_buildperf_all builds ONE combined kernel list spanning all four corpora
#                   and each rank measures a disjoint round-robin slice combined[rank::4] (off
#                   SLURM_PROCID/SLURM_NTASKS) -- so the corpora are distributed across the ranks
#                   TOGETHER, not one per-corpus pass. Run TWICE: once g++, once clang++.
#   2. multi_core   1 rank, paper preset, all 72 cores of one Grace socket (OMP_NUM_THREADS=72),
#                   both codegens, the SAME single combined list. Run TWICE: once g++, once clang++.
#
# => 4 sweeps: (single_core, g++), (single_core, clang++), (multi_core, g++), (multi_core,
#    clang++). CPU only -- there is no GPU phase.
#
# Every row of every sweep lands in ONE merged TSV ($OUT, default perfresults.tsv), header:
#
#   kernel corpus codegen preset threads cxx phase codegen_ms compile_ms runtime_ms \
#     code_bytes nloc max_nesting tokens max_ccn speedup correctness error
#
# `phase` (single_core|multi_core), `cxx` (g++|clang++) and `corpus`
# (npbench|polybench|tsvc2|tsvc2_5) keep the merged rows groupable. The build/size/readability
# metrics ride inline (from readability_metrics.py, scored on the frame each correctness build
# persists) -- so both plot figures come from this one TSV, no second build or harvest pass.
#
# Overridable with `sbatch --export=ALL,REPS=3,...`:
#   CODEGEN       legacy | experimental | both     (default both)
#   REPS          timed reps per lane              (default 10)
#   OUT           merged TSV path                  (default perfresults.tsv)
#   CPU_CXX_LIST  host compilers for both phases   (default "g++ clang++")
#   CPP_STANDARD  C++ standard, digits only        (default 20 -> consteval; 17 -> constexpr)
#
# Requires the sibling performance_regression_jobs harness next to this folder (shared
# engine.py / native_harness.py / adapters.py + the npbench/polybench corpora), with its
# submodules initialized:
#     git submodule update --init dace/external/moodycamel dace/external/cub
# The tsvc corpus harness itself is vendored here (tsvc2_perf.py, tsvc2_5_perf.py,
# tsvc_corpus.py, tsvc_2_5_corpus.py), as is readability_metrics.py.
#
# NOTE (cmake SIGCHLD hang): srun/mpirun start every task with SIGCHLD BLOCKED in the inherited
# signal mask. cmake's configure step forks helper processes and reaps them via select(); with
# SIGCHLD blocked it is never woken, so cmake spins forever. DaCe guards this in
# dace/codegen/compiler.py::_build_subprocess_sigmask() (compiler.py:375-404, used at
# compiler.py:907-911), which unblocks SIGCHLD ONLY around the build Popen -- so every build
# here must run under a plain `srun ... python3 ...` that inherits that unblocked-around-fork
# behavior. Do NOT wrap the launch in anything that re-blocks SIGCHLD or intercepts child
# signals: no `preexec_fn`, no trap/signal handler on SIGCHLD, no `setsid`/`stdbuf`-style shims
# around the python process, and no nested `srun` inside the build. Plain srun only.

set -euo pipefail

REPO_ROOT="/capstor/scratch/cscs/ybudanaz/aarch64/dace"   # TODO: set to the synced repo root
cd "$REPO_ROOT/codegen_buildperf_jobs"

CODEGEN="${CODEGEN:-both}"
REPS="${REPS:-10}"
OUT="${OUT:-perfresults.tsv}"
CPU_CXX_LIST="${CPU_CXX_LIST:-g++ clang++}"
CPP_STANDARD="${CPP_STANDARD:-20}"

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
#   - DaCe's CMake build (compiler.cpp_standard) -> the runtime/correctness binaries AND the
#     readable generator's consteval/constexpr choice (experimental_cpu.size_qualifier reads
#     compiler.cpp_standard: consteval under C++20+, constexpr under C++17), AND
#   - engine.compile_sdfg_timed's DIRECT compile (reads DACE_PERF_CXX_STD) -> the codegen_ms /
#     compile_ms numbers.
# Leaving them unset would measure compile time at engine's own default (c++23) while the
# runtime binary built at DaCe's default (c++20) -- a standard mismatch inside one comparison,
# and possibly a consteval-vs-constexpr mismatch too. Default 20 keeps consteval on. clang-tidy
# also reads compiler.cpp_standard (recently changed), so it stays consistent automatically --
# nothing else hardcodes a std here.
export DACE_compiler_cpp_standard="$CPP_STANDARD"
export DACE_PERF_CXX_STD="c++$CPP_STANDARD"

# Python venv (TODO: set PYTHONUSERBASE / the venv path to your daint.alps scratch setup; the
# spack loads below are the cluster's concrete package versions, NOT placeholders).
export PYTHONUSERBASE=/capstor/scratch/cscs/$USER/aarch64/python
export PATH=$PYTHONUSERBASE/bin:$PATH
#python3.11 -m venv /capstor/scratch/cscs/$USER/aarch64/venvs/myenv  # one-time setup; scratch can get purged
source /capstor/scratch/cscs/$USER/aarch64/venvs/myenv/bin/activate
alias python=python3.11

# Toolchain via spack, the SAME concrete loads the sibling jobs use
# (performance_regression_jobs/slurm_perf.sh:53-56, cpu_codegen_perf_jobs/submit_daint_readable.sbatch:125-128):
# g++ comes from the gcc package, clang++ from the llvm package -- these are the two host
# compilers swept in CPU_CXX_LIST. CPU-only: no cuda / cutensor.
spack load gcc@16.1.0     # provides g++
spack load llvm@22.1.5    # provides clang++
spack load cmake
spack load openblas threads=pthreads

# Fail fast if either host compiler did not land on PATH (a wrong spack version pin, or the
# module not installed on the node) -- otherwise DACE_PERF_CXX=g++/clang++ fails per-kernel later.
command -v g++ >/dev/null || { echo "error: g++ not on PATH after 'spack load gcc@16.1.0'"; exit 1; }
command -v clang++ >/dev/null || { echo "error: clang++ not on PATH after 'spack load llvm@22.1.5'"; exit 1; }
echo "toolchain: g++=$(command -v g++) clang++=$(command -v clang++)"

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

# OpenMP + libstdc++ runtime dirs on LD_LIBRARY_PATH so every compiled .so loads via ctypes.
_add_ldpath() { case ":${LD_LIBRARY_PATH:-}:" in *":$1:"*) ;; *) [ -d "$1" ] && export LD_LIBRARY_PATH="$1:${LD_LIBRARY_PATH:-}" ;; esac; }
_ldpath_for() { command -v "$1" >/dev/null 2>&1 || return 0; local _cc="$1"; shift; for _l in "$@"; do _p="$("$_cc" -print-file-name="$_l" 2>/dev/null)"; [ -f "$_p" ] && _add_ldpath "$(dirname "$_p")"; done; }
_ldpath_for clang++ libomp.so libgomp.so
_ldpath_for g++ libgomp.so libstdc++.so.6 libgcc_s.so.1
_ldpath_for gcc libstdc++.so.6 libgcc_s.so.1
unset -f _add_ldpath _ldpath_for

# HPTT (High-Performance Tensor Transpose) is an EXTERNAL, out-of-tree dependency (not vendored;
# see performance_regression_jobs/.gitignore). The npbench/polybench TensorTranspose nodes find
# its headers (<HPTT_ROOT>/include) and link+load its lib via DaCe's HPTT library environment
# (HPTT_ROOT). Point it at the sibling harness's local build if present; guarded, so a no-op
# when HPTT is not built.
export HPTT_ROOT="$REPO_ROOT/performance_regression_jobs/hptt"
[ -f "$HPTT_ROOT/lib/libhptt.so" ] && export LD_LIBRARY_PATH="$HPTT_ROOT/lib:${LD_LIBRARY_PATH:-}"

# Every per-phase TSV stem, collected for the final merge.
BASES=()

# Short filename tag for a compiler path/name ("g++" -> gcc, "/usr/bin/clang++-19" -> clang):
# `+` is awkward in globs, so the output files use this instead of the raw compiler name.
cxx_tag() { case "$1" in *clang*) echo clang ;; *) echo gcc ;; esac; }

# Drop any stale per-rank / merged TSV for these stems so a re-run never merges old rows in.
clear_stale() { local f; for f in "$@"; do rm -f "${f%.tsv}".rank*.tsv "$f"; done; }

# One CPU phase (ONE combined pass over all four corpora, both codegens) at one host compiler.
#   run_cpu_phase <phase> <preset> <ntasks> <threads> <cxx>
run_cpu_phase() {
    local phase="$1" preset="$2" ntasks="$3" threads="$4" cxx="$5"
    local tag out
    tag="$(cxx_tag "$cxx")"
    out="${phase}_${tag}.tsv"
    clear_stale "$out"

    # engine.configure_dace_process reads DACE_PERF_CXX -> compiler.cpu.executable; the runners
    # read it back for the `cxx` column, so the label can never disagree with the compiler used.
    export DACE_PERF_CXX="$cxx"
    export OMP_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS="$threads"
    export OMP_PROC_BIND=close   # keep each rank's thread(s) within its own socket
    export OMP_PLACES=cores

    echo "=== phase=$phase cxx=$cxx preset=$preset ranks=$ntasks OMP_NUM_THREADS=$threads codegen=$CODEGEN reps=$REPS"

    # ONE combined pass: run_buildperf_all builds a single kernel list across all four corpora
    # and each rank measures its own round-robin slice combined[rank::ntasks] (off SLURM_PROCID/
    # SLURM_NTASKS), writing <out>.rank<r> (a single-rank phase writes <out> itself -- the merge
    # globs both). Plain `srun ... python3 ...` only -- see the cmake SIGCHLD note in the header.
    srun --ntasks="$ntasks" --cpus-per-task="$SOCKET_CORES" --cpu-bind=verbose,cores --distribution=block:block \
        python3 run_buildperf_all.py --preset "$preset" --codegen "$CODEGEN" \
            --reps "$REPS" --phase "$phase" --out "$out" \
        || echo "warning: phase=$phase cxx=$cxx returned nonzero on some rank; keeping its completed rows"

    BASES+=("$out")
}

rm -f "$OUT"

# ---- execution order: phase 1 single_core (g++, clang++), phase 2 multi_core (g++, clang++) --
for cxx in $CPU_CXX_LIST; do
    run_cpu_phase single_core S 4 1 "$cxx"
done
for cxx in $CPU_CXX_LIST; do
    run_cpu_phase multi_core paper 1 "$SOCKET_CORES" "$cxx"
done

# ---- merge every phase's every rank into ONE header + rows --------------------------------
echo "merging all phases into $OUT"
python3 - "$OUT" "${BASES[@]}" <<'PYMERGE'
"""Concatenate every per-rank TSV of every phase under ONE header.

Each stem is either <stem>.rank<r>.tsv (a multi-rank phase, which self-partitions) or
<stem>.tsv (a single-rank phase, which does not add the suffix) -- glob both. Every runner
emits the identical header, so a header mismatch means a stale file from an older run:
report it loudly rather than silently writing misaligned rows.
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

# ---- how to plot the two figures ----------------------------------------------------------
# The build/size/readability metrics are inline in $OUT (no separate metrics harvest), so both
# figures come straight from the merged TSV.
echo "done: $OUT"
echo "plot both figures with:"
echo "  python3 plot_buildperf.py --tsv $OUT --out-dir plots"
echo "  -> plots/build_and_quality.{png,pdf} (codegen+compile time, LoC, readability; g++/clang++, per corpus)"
echo "  -> plots/performance.{png,pdf}       (single-core S + multi-core paper runtime; g++/clang++, per corpus)"
