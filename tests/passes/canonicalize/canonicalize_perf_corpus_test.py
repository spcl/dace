# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end PERFORMANCE / speedup harness over the four in-repo corpora (polybench,
npbench, tsvc, tsvc_2_5): time eight comparison arms per kernel on the same machine and
report each arm's speedup against that corpus's own denominator.

Design (resumable, one result file per kernel)
----------------------------------------------
Each kernel writes a single human-readable JSON file under a results directory
(``CANON_PERF_DIR``, default ``perf_results/`` next to this test). The file holds,
for both dataset presets (``S`` small, ``paper`` larger), the dataset shape, the
per-pipeline correctness flag, ALL timed repetitions (ms) plus min/median/mean/std,
and the speedup of each pipeline vs the baseline.

The pytest harness measures ONE kernel per test (``test_speedup[suite-kernel]``):

* If the kernel's result file already holds EVERY currently selected arm it is **skipped** --
  so an interrupted sweep resumes cheaply and only missing kernels are (re)measured. A file
  written under an older arm table is not "already measured" and is re-run.
* To re-run a kernel, delete its file (or set ``CANON_PERF_FORCE=1`` to remeasure
  all). To re-run a subset, delete just those files.
* After measuring, it asserts the candidate pipelines are not grossly slower than
  the baseline (the ratio is the only assertion; absolute times are never asserted).

The eight arms (opt-in, ``--arms`` or ``CANON_PERF_ARMS=1``)
------------------------------------------------------------
An *arm* is a (SDFG pipeline, C++ compiler, extra flags) triple, expressed as an ordinary pipeline
label -- so every arm gets the same correctness gate, the same timing path, the same speedup
columns. There is exactly ONE job shape, because there is one figure::

    dace-autoopt-gcc              auto_optimize          g++      parallelized by DaCe
    dace-autoopt-llvm             auto_optimize          clang++  parallelized by DaCe
    dace-canon-gcc                canonicalize           g++      parallelized by DaCe
    dace-canon-llvm               canonicalize           clang++  parallelized by DaCe
    dace-simplify+gcc-autopar     post-simplify, SERIAL  g++      GCC autopar + Graphite, FORCED
    dace-simplify+gcc-autopar-*   post-simplify, SERIAL  g++      ... on gcc's own cost model
    dace-simplify+llvm-autopar    post-simplify, SERIAL  clang++  Polly autopar, FORCED
    dace-simplify+llvm-autopar-*  post-simplify, SERIAL  clang++  ... on Polly's own cost model

Graphite is folded INTO the gcc arm and Polly INTO the llvm arm: a polyhedral pass that
restructures a nest nobody then parallelizes is not a comparison anyone reads, so each external
arm is one column that must prove BOTH its passes ran. ``seq-cpp`` (post-simplify, serial, base
flags only) is measured alongside them but is not a comparison arm -- it is the tsvc/tsvc25
speedup denominator.

Each auto-parallelizer is measured BOTH forced and on its own cost model, because the gap is a
result: forcing takes Polly from 1.209x to 7.572x (unforced it declines flat 1-D loops and times
sequential code) while gcc barely moves, 8.058x to 8.088x.

Rules that make the numbers mean something:

* ONE base flag string for every arm (``-ffp-contract=off`` verified present); an arm adds only its
  own flags. The codegen (``compiler.cpu.implementation``) is pinned too -- the autopar arms need
  the experimental generator's ``__restrict__`` or neither auto-parallelizer can disambiguate.
* The two autopar arms are handed genuinely SEQUENTIAL C++ (``serialize``); ``test_autopar_input``
  pins that the emitted pragma set is empty. Feeding an auto-parallelizer code DaCe already
  parallelized measures 4x oversubscription, not the compiler.
* Every arm is PROBED before the sweep: a known-good loop nest is compiled with the arm's real
  flags and the compiler must report EVERY marker it is credited with (a Graphite/Polly SCoP
  report AND ``GOMP_parallel`` as an undefined symbol of the emitted object). Missing binary,
  rejected flag or silent pass raises immediately -- an arm that quietly degraded to plain ``-O3``
  would be a fabricated baseline.
* Speedup denominators are PER CORPUS: polybench and npbench divide by their timed NUMPY
  reference, tsvc/tsvc_2_5 by the ``seq-cpp`` arm (their oracles are scalar Python loops). The
  groups answer different questions, are labelled by ``kind``, and are never pooled into one
  geomean. A numpy reference that is itself a scalar python loop (polybench ``seidel_2d``) is
  labelled ``python-scalar`` and left untimed rather than divided by.

CSV export (option)
-------------------
A flat per-(suite, kernel, preset, pipeline) summary CSV can be produced from the
result files with ``--csv PATH`` (script) or ``CANON_PERF_CSV=PATH`` (env, exported
at the end of a sweep). The JSON files remain the precise source (all repetitions).

Usage::

    # measure (writes/refreshes perf_results/<suite>_<kernel>.json), skips complete ones:
    python -m tests.passes.canonicalize.canonicalize_perf_corpus_test --arms
    # re-measure everything and also export a CSV summary:
    python -m tests.passes.canonicalize.canonicalize_perf_corpus_test --arms --force --csv perf.csv
    # only kernels matching a substring:
    python -m tests.passes.canonicalize.canonicalize_perf_corpus_test --arms --only gemm
    # one corpus, every 4th kernel starting at 1 (how the batch job fans ranks out):
    python -m tests.passes.canonicalize.canonicalize_perf_corpus_test --arms --suite poly --shard 1/4
    # as a CI gate (per-kernel speedup test, ``perf`` marker):
    pytest -m perf tests/passes/canonicalize/canonicalize_perf_corpus_test.py

``CANON_PERF_ARMS=1`` in the environment is equivalent to ``--arms`` and is how the batch job
enables the table -- it needs no CLI coupling and is read at import, before pytest collection.
"""
import os

# 72 == one CSCS node's 288 cores over the 4 ranks the submit script requests, i.e. the width
# every measurement in this harness is meant to run at. Set BEFORE anything reads it, and
# before any OpenMP runtime can be loaded, so a bare run does not silently time 4 threads and
# report it under the same arm labels as a batch run.
os.environ.setdefault("OMP_NUM_THREADS", "72")
os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import contextlib
import csv
import datetime
import functools
import importlib
import json
import multiprocessing
import math
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from typing import Any, NamedTuple

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.libraries.blas.environments.openblas import (OPENBLAS_PARALLEL_NAMES, OpenBLAS, _openblas_threading_flavor,
                                                       _standalone_libopenblas)
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.cpu_specialization import SpecializeCpuTransfers
from dace.transformation.passes.insert_explicit_copies import InsertExplicitCopies
from tests.corpus import corpus_suite as CS
from tests.corpus.npbench import npbench
from tests.corpus.polybench import polybench as PB
from tests.corpus.polybench import polybench_numpy as PBN

#: The CPU set this process started with, captured before anything can load an OpenMP runtime.
#:
#: Loading the spack OpenBLAS (threads=openmp, so libgomp) under OMP_PROC_BIND=close binds the
#: calling thread and collapses the PROCESS affinity mask to a single core -- measured here: 288
#: cpus before the ctypes load, 1 cpu after. In-process that is harmless, since libgomp still binds
#: its team across the places it captured at init. A SPAWNED CHILD, however, inherits the shrunken
#: mask and its own libgomp then sees exactly one place, so all OMP_NUM_THREADS threads land on one
#: core -- measured as a 20x regression on every libgomp arm (autoopt-gcc 2.10ms -> 41.08ms) while
#: sequential and libomp arms were untouched. Children restore this set before measuring anything.
_FULL_AFFINITY = frozenset(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else frozenset()

#: ``canonicalize`` keyword bag; ``Any`` because it is a heterogeneous kwargs splat, not a mapping.
_CPU: dict[str, Any] = dict(target='cpu',
                            peel_limit=4,
                            break_anti_dependence=True,
                            interchange_carry_with_map=True,
                            scatter_to_guarded_maps=True)


def _canon(s: dace.SDFG) -> dace.SDFG:
    return finalize_for_target(canonicalize(s, validate=True, **_CPU), 'cpu')


def _autoopt(s: dace.SDFG) -> dace.SDFG:
    return auto_optimize(s, dace.DeviceType.CPU)


#: The arm every ``speedup_vs_baseline`` column divides by. It is a DaCe arm on purpose: that
#: column answers "did canonicalize regress against auto_optimize", which is the CI gate. The
#: paper figure uses ``speedup_vs_reference`` instead, whose denominator is per corpus.
BASELINE = 'dace-autoopt-gcc'
#: Pipelines to time, ``label -> transform``. Populated from the arm table below; the default is
#: the two g++ DaCe arms, which claim no external pass and so need no toolchain probe.
_PIPELINES: dict[str, Callable[[dace.SDFG], dace.SDFG]] = {}

#: A candidate pipeline is a "regression" only past this multiple of the baseline
#: (best-of-N). Generous so timing noise never flakes CI; catches gross regressions.
REGRESSION_FACTOR = 6.0
#: Below this baseline runtime (ms) the kernel is too fast to time reliably; gate it
#: very leniently rather than flake on per-call overhead.
_MIN_TIMEABLE_MS = 0.5
_REPS = int(os.environ.get('CANON_PERF_REPS', '7'))
_WARMUP = int(os.environ.get('CANON_PERF_WARMUP', '2'))
#: Rep-count floor and per-(kernel, arm) timing budget. ``_REPS`` is the CEILING now, not the flat
#: count: ``_reps_for`` spends up to ``_BUDGET_MS`` of wall clock and stops, so a cheap kernel still
#: gets the full ``_REPS`` while a 20s stencil gets ``_MIN_REPS`` instead of turning one kernel into
#: two hours. The floor keeps a median meaningful; raise the budget to trade wall clock for tighter
#: medians on the expensive tail.
#: Upper bound on a SINGLE call at the paper shape. A paper-size kernel is meant to be a realistic
#: unit of work, not a batch job: one call past this is a sizing bug in the corpus, not a slow
#: machine, and the corpus should be re-shaped rather than the sweep left to absorb it. Enforced
#: from the probe call ``_time_all_reps`` already makes, so it costs nothing extra and the kernel is
#: still REPORTED (with reps=1 and an ``over_budget`` flag) rather than dropped -- a silent hole in
#: the table would hide the very thing this is meant to surface. Three minutes leaves room for the
#: heaviest honest arm (seq-cpp on jacobi_2d measured 4.87s, so the margin is ~37x) while still
#: catching a kernel whose paper shape has run away. Lower it with CANON_PERF_MAX_CALL_MS.
_MAX_CALL_MS = float(os.environ.get('CANON_PERF_MAX_CALL_MS', '180000'))
_MIN_REPS = int(os.environ.get('CANON_PERF_MIN_REPS', '5'))
_BUDGET_MS = float(os.environ.get('CANON_PERF_BUDGET_MS', '20000'))
_PER_KERNEL_TIMEOUT = int(os.environ.get('CANON_PERF_TIMEOUT', '600'))
#: Per-ARM cap. The kernel cap alone is not enough: its alarm is spent on whichever arm is
#: running when it fires, so a single slow arm both loses itself and leaves its siblings
#: unbounded. A third of the kernel budget lets the typical nine-arm kernel finish while
#: still catching an arm that has genuinely run away.
_PER_ARM_TIMEOUT = int(os.environ.get('CANON_PERF_ARM_TIMEOUT', str(max(60, _PER_KERNEL_TIMEOUT // 3))))
#: Measure each OpenMP-runtime family in its OWN process. Load-bearing for the LLVM columns, not a
#: tidiness knob: libgomp and libomp co-resident in one process cost ~34x. MEASURED on this box with
#: a standalone microbenchmark of DaCe's emitted jacobi_1d body at OMP_NUM_THREADS=72 -- clang alone
#: 19.15ms, clang with libgomp also loaded 660.17ms, and gcc with libomp also loaded 664.43ms, so
#: the penalty is symmetric and belongs to the pairing rather than to either compiler. Both runtimes
#: build a pool of OMP_NUM_THREADS threads and spin against each other.
#:
#: The arms used to share one process, gcc first, so every clang arm ran with both runtimes loaded
#: and the LLVM columns read 50-150x slow for a reason that had nothing to do with LLVM. Grouping by
#: runtime costs one extra dataset build per kernel and buys back a comparable column.
#:
#: NOT a complete fix, and the remaining hole is worth knowing: the spack OpenBLAS is
#: threads=openmp built with gcc, so it pulls in libgomp. A clang arm on a kernel with BLAS nodes
#: therefore still meets libgomp inside its own process. numpy is clear -- it ships its own bundled
#: libscipy_openblas64 and links neither runtime -- so a stencil, which is where this was found, is
#: fully isolated. Set CANON_PERF_ISOLATE_ARMS=0 to measure everything in one process again.
#:
#: DEFAULT OFF, and this is UNRESOLVED rather than settled. Isolation does fix the llvm columns --
#: measured on jacobi_1d, autoopt-llvm 59.39ms -> 0.845ms and canon-llvm 60.17ms -> 0.830ms, a 70x
#: recovery that also makes them the fastest arms, which is what the co-residency was hiding. But it
#: breaks the gcc columns by the same mechanism in reverse: a spawned child loads OpenBLAS, libgomp
#: binds and collapses the mask to one core, and the child's libgomp has ALREADY fixed its place
#: list by the time anything can widen the mask again, so every libgomp arm runs on one core
#: (autoopt-gcc 1.94ms -> 39.6ms). Reproduced standalone at 40.76ms vs 3.62ms for a child with no
#: OpenBLAS loaded; restoring the mask helps only a FRESHLY SPAWNED process (8.07ms), never the
#: process that already initialised libgomp.
#:
#: So neither setting currently yields a wholly valid table: OFF gives honest gcc columns and
#: co-residency-inflated llvm ones, ON gives honest llvm columns and one-core gcc ones. OFF is the
#: default because it is the state every earlier measurement was taken in, and because a silent 20x
#: regression on the four gcc arms is worse than the known llvm inflation. The real fix is to stop
#: the collapse happening at all -- an OpenBLAS not linked against libgomp, or preventing libgomp
#: from binding the master thread before its place list is built -- not to widen the mask after.
_ISOLATE_ARMS = os.environ.get('CANON_PERF_ISOLATE_ARMS', '0') not in ('0', 'false', 'False')
#: Set in the spawned children so they skip the toolchain probe the parent already ran.
_IS_CHILD = os.environ.get('CANON_PERF_ARM_CHILD') == '1'
#: Bumped when the measurement METHOD changes so that results produced by the old one stop counting
#: as current. 2 == per-runtime process isolation; every record written before it has LLVM columns
#: contaminated by the co-residency above and must be re-measured rather than skipped.
MEASUREMENT_EPOCH = 2

#: Where per-kernel result files live (one ``<suite>_<kernel>.json`` each).
_RESULTS_DIR: str = os.environ.get('CANON_PERF_DIR',
                                   os.path.join(os.path.dirname(os.path.abspath(__file__)), 'perf_results'))
#: Re-measure even when a result file already exists.
_FORCE = os.environ.get('CANON_PERF_FORCE', '') not in ('', '0', 'false', 'False')
#: Dataset presets to measure, defaulting to every preset the corpus adapter defines.
#: ``CANON_PERF_PRESETS=paper`` (comma-separated) halves the compiles of a sweep that only wants
#: the performance-realistic shape -- and the paper-preset compiles are what the sweep costs.
_ONLY_PRESETS = [p for p in os.environ.get('CANON_PERF_PRESETS', '').split(',') if p]
PRESETS = tuple(p for p in CS.PRESETS if not _ONLY_PRESETS or p in _ONLY_PRESETS)
#: Also save each pipeline's SDFG BEFORE library-node expansion to ``<dir>/sdfg/``
#: (opt-in; ``--save-sdfg`` from the script sets this too).
_SAVE_SDFG = os.environ.get('CANON_PERF_SAVE_SDFG', '') not in ('', '0', 'false', 'False')

#: Structural-only builders -- the SDFG each PIPELINE produces BEFORE library-node expansion
#: (Reduce/MatMul/Einsum nodes still present). Keyed by pipeline, not by arm: the SDFG is the same
#: whichever compiler the arm goes on to use. canon stops at the ``canonicalize`` output
#: (``finalize_for_target`` would expand it); autoopt uses ``auto_optimize(..., expand=False)``.
_PRE_EXPANSION = {
    'autoopt': lambda s: auto_optimize(s, dace.DeviceType.CPU, expand=False),
    'canon': lambda s: canonicalize(s, validate=True, **_CPU),
}

# ---------------------------------------------------------------------------
# The comparison ARMS.
#
# An "arm" is a (SDFG pipeline, C++ compiler, extra flags) triple. The harness keys the per-kernel
# JSON, the speedup map, the CSV and the Markdown table by pipeline label, so an arm costs one
# ``_PIPELINES`` entry plus a toolchain override around its build -- no second axis, no schema
# change.
#
# There is exactly ONE job shape: the eight arms the figure compares, plus ``seq-cpp`` because it is
# the tsvc/tsvc25 denominator. Only the two g++ DaCe arms are timed by default; the full table is
# opt-in because four of its arms credit an external pass whose probe must be allowed to fail loudly.
# ---------------------------------------------------------------------------


class Arm(NamedTuple):
    """One measured arm: which SDFG, which C++ driver, which extra flags, and the markers that make
    that compiler *prove* EVERY pass it is being credited with actually ran (``()`` claims none)."""
    label: str
    transform: Callable[[dace.SDFG], dace.SDFG]
    executable: str
    flags: str
    probe_flags: str
    markers: tuple[str, ...]
    #: Which implementation the BLAS/LAPACK/linalg library nodes expand to for THIS arm.
    #:
    #: Per-arm rather than process-wide because the arms ask different questions. A DaCe pipeline
    #: arm is competing with numpy, and numpy calls OpenBLAS -- expanding gemv/gemm to DaCe's
    #: ``pure`` nested maps instead measures a naive triple loop against a tuned kernel and loses by
    #: 20-40x on exactly the kernels that are one BLAS call (atax, bicg, covariance). ``dace``'s
    #: shipped default for ``library.blas`` is ``pure`` while ``lapack`` and ``linalg`` already
    #: default to ``OpenBLAS``, so BLAS is the odd one out and every DaCe arm must say so explicitly.
    #:
    #: The serialize arms keep ``pure`` deliberately, and that is NOT an oversight: they exist to
    #: measure what gcc's parloops and Polly do to sequential affine loops. An OpenBLAS call is
    #: neither affine nor sequential -- it is a multithreaded library call the auto-parallelizer
    #: cannot see into -- so lowering their gemms to OpenBLAS would make all three arms report the
    #: same OpenBLAS number and measure nothing about either compiler. It would also silently
    #: destroy ``seq-cpp``, whose whole meaning is "one thread", since a threads=openmp OpenBLAS
    #: runs on OMP_NUM_THREADS regardless of the map schedule.
    blas: str = 'pure'
    #: ``compiler.cpu.implementation`` for THIS arm: which CPU code generator emits its C++.
    #:
    #: Split on purpose. ``auto_optimize`` is the established pipeline and is measured on the
    #: generator it was tuned against (``legacy``); ``canonicalize`` is the new pipeline and is
    #: measured on the new one (``experimental_readable``) -- so the canon-vs-autoopt column
    #: compares each pipeline as it is actually meant to be shipped, rather than forcing one of them
    #: through a generator it was never developed against.
    #:
    #: The serialize arms stay on ``experimental_readable`` regardless of that split: they need its
    #: ``__restrict__`` on the emitted array parameters, because neither gcc's parloops nor Polly can
    #: disambiguate memory without it, and both would silently decline to parallelize -- which is the
    #: exact failure their probes exist to catch.
    codegen: str = 'experimental_readable'


def untransformed(sdfg: dace.SDFG) -> dace.SDFG:
    """Identity 'pipeline': the POST-SIMPLIFY frontend SDFG, unchanged.

    ``corpus_suite.build`` already hands every corpus ``to_sdfg(simplify=True)``, i.e. DaCe's own
    ``SimplifyPass`` and nothing else. That is the honest floor to hand an external compiler: an
    unsimplified graph emits redundant transients and copies DaCe would never ship, so beating it
    would say nothing about the compiler.
    """
    return sdfg


def serialize(sdfg: dace.SDFG) -> dace.SDFG:
    """Strip ALL parallelism from ``sdfg``: expand library nodes, then force every map sequential.

    Load-bearing for the autopar arms. DaCe resolves a default-scheduled top-level map to
    ``CPU_Multicore``, so the plain post-simplify SDFG already emits ``#pragma omp parallel for``
    (gemm: 5 of them) -- handing that to an auto-parallelizer measures nested parallelism, not the
    compiler. Library nodes are expanded FIRST because they are expanded again at codegen time and
    bring their own multicore maps with them: expanding late leaves pragmas behind (gemm: 5 -> 2),
    expanding first leaves none.

    Bulk transfers are the same story one level later. The canonical lowering of a copy / zero is
    the parallel element map, and codegen LIFTS the implicit access-node-to-access-node edges into
    copy library nodes of its own (``InsertExplicitCopies``), so an SDFG this function already
    walked grows a fresh ``#pragma omp parallel for`` after it returns (gemm's ``C_slice -> C``).
    They are lifted HERE instead, pinned ``Sequential`` with every other scope, and handed to the
    CPU specialization band -- ``SpecializeCpuTransfers`` gives a sequential contiguous transfer
    its single ``std::memcpy`` back, which is what this arm used to measure. Codegen's own lift
    then finds nothing left to lift.
    """
    sdfg.expand_library_nodes()
    InsertExplicitCopies().apply_pass(sdfg, {})
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.MapEntry):
            node.map.schedule = dace.ScheduleType.Sequential
            # Leave ``omp_simd`` as-is: ``#pragma omp simd`` is valid on a sequential loop and the
            # baseline should measure the same vectorized body the canonicalizer targets.
        elif isinstance(node, nodes.LibraryNode):
            node.schedule = dace.ScheduleType.Sequential
    SpecializeCpuTransfers().apply_pass(sdfg, {})
    sdfg.expand_library_nodes()
    return sdfg


#: BOTH shapes are needed: unforced, gcc parallelizes the flat loop and declines the nest, Polly the
#: reverse (see commit 77676aea6). A green probe proves a pass CAN parallelize something, never that
#: it touched a measured kernel.
_PROBE_SRC = ('void probe_scop(double *__restrict__ A, double *__restrict__ B, double *__restrict__ C) {\n'
              '    for (int i = 0; i < 4096; i++)\n'
              '        for (int j = 0; j < 4096; j++)\n'
              '            C[i * 4096 + j] = A[i * 4096 + j] + B[i * 4096 + j] * 3.0;\n'
              '}\n'
              'void probe_flat(double *__restrict__ A, double *__restrict__ B, int n) {\n'
              '    for (long i = 0; i < n; i = i + 1) { const double t = B[i]; A[i] = t + 1.0; }\n'
              '}\n')

#: The ONE base flag set EVERY arm is built with; an arm adds only its own flags on top, so the
#: comparison differs in nothing else. Defaults to the string ``submit_corpus_perf.sh`` pins, and
#: defers to that pin when the job already exported it.
#:
#: NOT -ffast-math: it implies -freciprocal-math and -fassociative-math, which let each compiler
#: reassociate reductions its own way, so the gcc and clang columns of the same pipeline stop
#: computing the same expression and the table compares two different programs. The individually
#: safe relaxations are named instead -- no errno on math calls, no FP traps, no signed-zero
#: distinction -- none of which changes the association order.
#:
#: -ffp-contract is pinned EXPLICITLY, and the value matters less than the pin: gcc defaults to
#: ``fast`` while clang defaults to ``on`` for C++, so leaving it out silently fuses differently in
#: the two compiler columns. ``fast`` lets both emit FMA, which is what an -O3 -march=native
#: measurement should be showing. Correctness stays checkable because the corpus compares floats at
#: a per-precision tolerance (``polybench.outputs_match``), not bit-exactly.
_BASE_ARGS = os.environ.get(
    'DACE_compiler_cpu_args', '-std=c++20 -fPIC -Wall -Wextra -O3 -march=native -fno-math-errno '
    '-fno-trapping-math -fno-signed-zeros -ffp-contract=fast')

#: One knob per compiler column and one per external pass, all overridable so a variant (a
#: different thread count, ``-polly-vectorizer``, another GCC) is a sweep, not an edit.
_GCC = os.environ.get('CANON_PERF_GCC_CXX', 'g++')
_CLANG = os.environ.get('CANON_PERF_CLANG_CXX', 'clang++')
#: gcc's auto-parallelizer takes its thread count as a compile-time flag, and that flag IS the
#: runtime width, not a heuristic hint: parloops emits the count as a literal ``num_threads``
#: argument (``mov w2, <N>; bl GOMP_parallel``), so it does NOT widen to OMP_NUM_THREADS at run
#: time. Consequence for the table: whenever ``_AUTOPAR_HINT`` is capped below ``_THREADS``, this
#: one arm runs narrower than the other five and its speedup column is understated by roughly the
#: width ratio. That is why the cap is recorded into the evidence string rather than applied
#: silently -- a narrower arm is a legitimate measurement only if the reader can see the width.
#: Falls back to 72 -- one CSCS node's 288 cores over the 4 ranks the submit script requests --
#: so a bare run matches the width the batch job measures at instead of silently timing 4
#: threads and reporting it under the same label.
_THREADS = os.environ.get('OMP_NUM_THREADS', '72')
#: The hint is the FULL runtime width by default, so the arm asks gcc for the parallelism the job
#: actually runs with. The cap exists because of a MEASURED suppression: on g++ 15.2 (x86), with
#: Graphite in the same command line, parloops stops parallelizing a 4096x4096 affine nest somewhere
#: between 33 and 48 and emits NO GOMP_parallel at 48 or 72 -- silently turning this arm into plain
#: -O3. Without Graphite it parallelizes at 72, so the interaction is Graphite's, not the thread
#: count's, and neither -floop-parallelize-all nor --param parloops-min-per-thread overrides it.
#: That is compiler-and-target specific, so it is not defaulted around: ``probe_arm`` refuses to
#: register an autopar arm whose probe object has no GOMP_parallel undefined symbol, which turns
#: suppression into a launch failure rather than a fabricated column. Set
#: CANON_PERF_GCC_AUTOPAR_HINT_CAP to fall back to a working width if a toolchain trips it.
#:
#: MEASURED AGAIN on g++ 16.1.0 / aarch64 Neoverse V2 (2026-08-06), which tripped it and stopped a
#: sweep at OMP_NUM_THREADS=72. Bisected on the probe nest: parallelizes at every hint <= 40 and at
#: NO hint >= 41, so ``resolve_autopar_hint`` settles on 40 there. ``submit_corpus_perf.sh``
#: deliberately does NOT pin the cap: the probe discovers this width, and exporting a site number
#: would go stale the moment the compiler changes. Two refinements over the x86 note above. First, the trigger is specifically ``-floop-nest-optimize``, not Graphite in
#: general: at hint=72, ``-fgraphite-identity`` alone still emits GOMP_parallel and
#: ``-floop-nest-optimize`` alone still suppresses it. Dropping it would buy the full 72 threads at
#: the cost of the isl scheduling that makes this arm polyhedral at all, so the cap is preferred and
#: the arm keeps its full flag set. Second, ``--param parloops-min-per-thread`` does not override
#: the suppression at any value down to 1, so the numeric coincidence with the 4096-iteration probe
#: nest is not the cost model talking.
_AUTOPAR_HINT_CAP = int(os.environ.get('CANON_PERF_GCC_AUTOPAR_HINT_CAP', _THREADS))
_AUTOPAR_HINT = min(int(_THREADS), _AUTOPAR_HINT_CAP)
#: Graphite is folded into the gcc autopar arm and Polly into the llvm one -- the user's table has
#: eight columns, and a polyhedral restructuring nobody then parallelizes answers no question the
#: figure asks. Polly IS the llvm arm's polyhedral engine, so ``-polly -polly-parallel`` is both.
#:
#:
#: NO ``-floop-nest-optimize``: on g++ 16.1 / Neoverse V2 it suppresses parloops above width 40, and
#: that width is the arm's real thread count, so it would run 40 threads against everyone else's 72.
#: Read this column as "gcc autopar + Graphite codegen, no isl rescheduling".
#:
#:
#: ``-floop-parallelize-all`` forces past the profitability heuristic (never the dependence
#: analysis), pairing with ``-polly-parallel-force``; for gcc it is inside the noise, 8.088x against
#: 8.058x. Commit 77676aea6 has the measurements.
_GCC_AUTOPAR_FLAGS = os.environ.get(
    'CANON_PERF_GCC_AUTOPAR_FLAGS', f'-fopenmp -ftree-parallelize-loops={_AUTOPAR_HINT} -floop-parallelize-all '
    '-fgraphite-identity')
#: ``-polly-omp-backend=LLVM`` keeps libgomp out of a process that already links libomp (two
#: co-resident runtimes cost ~34x). ``-process-unprofitable`` widens SCoP detection and
#: ``-parallel-force`` then parallelizes what Polly proved safe -- neither alone emits any parallel
#: code, and unforced Polly declines flat 1-D loops at any trip count, so this column would otherwise
#: time sequential code (1.209x vs 7.572x). Commit 77676aea6 and 20ae06d56 have the measurements.
_LLVM_AUTOPAR_FLAGS = os.environ.get(
    'CANON_PERF_LLVM_AUTOPAR_FLAGS', '-fopenmp -mllvm -polly -mllvm -polly-parallel '
    '-mllvm -polly-omp-backend=LLVM -mllvm -polly-process-unprofitable -mllvm -polly-parallel-force')

#: What the four DaCe arms lower BLAS/LAPACK/linalg library nodes to. Overridable so a box with no
#: OpenBLAS can still run the sweep (``CANON_PERF_DACE_BLAS=pure``), but NOT defaulted to ``pure``:
#: the figure's headline comparison is against numpy, and numpy is OpenBLAS.
_DACE_BLAS = os.environ.get('CANON_PERF_DACE_BLAS', 'OpenBLAS')

#: ``GOMP_parallel`` as an UNDEFINED SYMBOL of the probe object is a stronger engagement proof than
#: any diagnostic: it means the compiler really emitted a parallel region.
_AUTOPAR_MARKER = 'GOMP_parallel'
#: Separate from ``_AUTOPAR_MARKER``, not a substring both runtimes match: a marker accepting either
#: would not catch Polly reverting to the GNU backend and re-creating the libgomp+libomp conflict.
_LLVM_AUTOPAR_MARKER = '__kmpc_fork_call'
#: The polyhedral half of each external arm, reported by its own dump/remark flag. Required
#: *alongside* the autopar marker: an arm credited with Graphite that only ran ``-ftree-parallelize``
#: is mislabelled, and mislabelled is indistinguishable from fabricated once it is a bar in a plot.
_GRAPHITE_MARKER = 'Adding SCoP'
_POLLY_MARKER = 'SCoP begins here'

#: Derived by REMOVAL so the pair always differs in exactly one flag, even under an env override.
_GCC_AUTOPAR_DEFAULT_FLAGS = _GCC_AUTOPAR_FLAGS.replace(' -floop-parallelize-all', '')
_LLVM_AUTOPAR_DEFAULT_FLAGS = _LLVM_AUTOPAR_FLAGS.replace(' -mllvm -polly-parallel-force', '')

#: The EIGHT comparison arms + ``seq-cpp``, in column order (the baseline first). An arm with EMPTY
#: ``flags`` is a plain DaCe pipeline built by the named compiler and claims no external pass; an
#: arm with flags credits external passes and therefore must prove every one of them ran. Each
#: auto-parallelizer appears twice: ``-default`` is its out-of-the-box behaviour, the unsuffixed one
#: its forced ceiling.
ARMS = (
    Arm(BASELINE, _autoopt, _GCC, '', '', (), _DACE_BLAS, 'legacy'),
    Arm('dace-autoopt-llvm', _autoopt, _CLANG, '', '', (), _DACE_BLAS, 'legacy'),
    Arm('dace-canon-gcc', _canon, _GCC, '', '', (), _DACE_BLAS, 'experimental_readable'),
    Arm('dace-canon-llvm', _canon, _CLANG, '', '', (), _DACE_BLAS, 'experimental_readable'),
    Arm('dace-simplify+gcc-autopar', serialize, _GCC, _GCC_AUTOPAR_FLAGS,
        '-fopt-info-loop -fdump-tree-graphite-details=stderr', (_GRAPHITE_MARKER, _AUTOPAR_MARKER)),
    Arm('dace-simplify+gcc-autopar-default', serialize, _GCC, _GCC_AUTOPAR_DEFAULT_FLAGS,
        '-fopt-info-loop -fdump-tree-graphite-details=stderr', (_GRAPHITE_MARKER, _AUTOPAR_MARKER)),
    Arm('dace-simplify+llvm-autopar', serialize, _CLANG, _LLVM_AUTOPAR_FLAGS, '-Rpass-analysis=polly-scops',
        (_POLLY_MARKER, _LLVM_AUTOPAR_MARKER)),
    Arm('dace-simplify+llvm-autopar-default', serialize, _CLANG, _LLVM_AUTOPAR_DEFAULT_FLAGS,
        '-Rpass-analysis=polly-scops', (_POLLY_MARKER, _LLVM_AUTOPAR_MARKER)),
    Arm('seq-cpp', serialize, _GCC, '', '', ()),
)

#: Arms timed without ``--arms``: the two g++ DaCe pipelines. They claim no external pass, so they
#: need no probe and cannot fail at import on a box without clang -- which matters because this
#: module is also collected by a plain ``pytest tests/``.
_DEFAULT_ARMS = (ARMS[0], ARMS[2])

#: Labels this harness used BEFORE the current arm table, and what each became. DOCUMENTATION only:
#: the per-kernel JSONs already on disk were measured at ``OMP_NUM_THREADS=8`` with neither the
#: shared flag string nor the codegen pinned, so they are NOT the same measurement and are
#: deliberately not renamed into the new labels. They still LOAD -- every reader keys off the labels
#: present in the file and off the record's own ``baseline`` -- and render as their own columns,
#: flagged stale in the Markdown header. ``has_current_arms`` re-measures them on the next sweep.
LEGACY_LABELS = {
    'auto-opt': BASELINE,
    'auto-opt+llvm': 'dace-autoopt-llvm',
    'canon': 'dace-canon-gcc',
    'canon+gcc': 'dace-canon-gcc',
    'canon+llvm': 'dace-canon-llvm',
    'gcc-graphite': 'dace-simplify+gcc-autopar',  # graphite folded INTO the gcc autopar arm
    'gcc-autopar': 'dace-simplify+gcc-autopar',
    'clang-polly': 'dace-simplify+llvm-autopar',  # polly folded INTO the llvm autopar arm
    'llvm-autopar': 'dace-simplify+llvm-autopar',
    'canon-serial+gcc-autopar': '',  # dropped: canonicalize-then-serialize is not one of the arms
    'canon-serial+llvm-autopar': '',
}  # ``seq-cpp`` is absent on purpose: it kept its name and its meaning, so nothing about it moved.

#: pipeline label -> arm. Always populated, so every measurement runs on a pinned toolchain.
_ARMS: dict[str, Arm] = {}
#: pipeline label -> the compiler line proving its external passes engaged, recorded into every
#: result file so a number can never be read without its provenance.
_EVIDENCE: dict[str, str] = {}


def arm_tag(label: str) -> str:
    """SDFG-name suffix for an arm: alphanumerics only, since ``+``/``-`` are not identifier
    characters and the tag ends up in a C++ symbol. Unchanged for the pre-existing labels."""
    return ''.join(ch for ch in label if ch.isalnum())


def arm_omp_runtime(arm: Arm) -> str:
    """Which OpenMP runtime this arm's compiler links: clang pulls libomp, gcc libgomp."""
    return 'omp' if arm.executable == _CLANG else 'gomp'


def blas_note(arm: Arm) -> str:
    """This arm's BLAS lowering, plus the resolved library when it is not ``pure``.

    Goes into the evidence string so a result file records WHICH libopenblas produced its numbers --
    the threading flavor in particular, since a ``threads=pthreads`` build times very differently
    from the ``threads=openmp`` one while computing the same answer."""
    if arm.blas == 'pure':
        return 'pure (naive maps, no vendor kernel)'
    # Under the same per-compiler OPENBLAS_DIR the arm's own builds see (``toolchain_env``), or the
    # evidence line records a different library than the one that produced the numbers.
    saved = os.environ.get('OPENBLAS_DIR')
    per_cc = os.environ.get('CANON_PERF_OPENBLAS_LLVM' if arm.executable == _CLANG else 'CANON_PERF_OPENBLAS_GCC')
    if per_cc:
        os.environ['OPENBLAS_DIR'] = per_cc
    try:
        lib, _ = _standalone_libopenblas()
        code, config, _ = _openblas_threading_flavor()
    finally:
        if per_cc:
            os.environ.pop('OPENBLAS_DIR', None)
            if saved is not None:
                os.environ['OPENBLAS_DIR'] = saved
    return f'{arm.blas} -> {lib or "?"} [parallel={OPENBLAS_PARALLEL_NAMES.get(code, code)}] {config or ""}'.strip()


def probe_arm(arm: Arm) -> str:
    """Compile a known SCoP with this arm's REAL flags; return the lines proving its passes engaged.

    Raises ``RuntimeError`` naming the missing binary, the rejected flag, or the silent pass. EVERY
    marker must appear: an arm labelled ``gcc-autopar`` that ran Graphite but parallelized nothing
    is as misleading as one that ran neither. Nothing falls back -- an arm that quietly degrades to
    plain ``-O3`` under a polyhedral label is a fabricated baseline, worse than having no baseline.
    """
    if _IS_CHILD:
        # The parent probed every arm before spawning anything; re-proving it in each child would
        # recompile the probe nest 9x per kernel for no extra integrity.
        return f'{os.path.basename(arm.executable)} (probed in parent)'
    exe = shutil.which(arm.executable)
    if exe is None:
        raise RuntimeError(f"arm {arm.label}: compiler {arm.executable!r} is not on PATH")
    with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR') or None) as tmp:
        src, obj = os.path.join(tmp, 'probe.cpp'), os.path.join(tmp, 'probe.o')
        with open(src, 'w') as f:
            f.write(_PROBE_SRC)
        cmd = [exe, *shlex.split(_BASE_ARGS), *shlex.split(arm.flags), *shlex.split(arm.probe_flags)]
        proc = subprocess.run(cmd + ['-c', src, '-o', obj], capture_output=True, text=True, timeout=300)
        out = proc.stdout + proc.stderr
        if proc.returncode != 0:
            raise RuntimeError(f"arm {arm.label}: {exe} rejected {arm.flags!r} "
                               f"(exit {proc.returncode}): {out.strip()[:400]}")
        if not arm.markers:  # a plain DaCe arm: compiling with the base flags is the whole claim
            return f'{os.path.basename(exe)} (base flags only, no external pass claimed); blas={blas_note(arm)}'
        # Undefined symbols of the object, so an arm can be proven by what the compiler EMITTED
        # rather than by a diagnostic it happened to print.
        syms = subprocess.run(['nm', '-u', obj], capture_output=True, text=True, timeout=300)
        out += ''.join(f'nm -u: {line}\n' for line in (syms.stdout + syms.stderr).splitlines())
    proof = []
    for marker in arm.markers:
        hit = next((ln.strip() for ln in out.splitlines() if marker in ln), '')
        if not hit:
            raise RuntimeError(f"arm {arm.label}: {exe} accepted {arm.flags!r} but never reported "
                               f"{marker!r} on a known-good loop nest -- that pass is not in this build. "
                               f"Refusing to report plain -O3 timings under this label.")
        proof.append(hit)
    return f'{os.path.basename(exe)} {arm.flags}: ' + ' | '.join(proof) + f'; blas={blas_note(arm)}'


@contextlib.contextmanager
def toolchain_env(arm: Arm) -> Iterator[None]:
    """Build+time the enclosed arm on ``arm``'s pinned toolchain; restore the caller's on exit.

    Pins every backend axis together, per arm rather than process-wide: the compiler, the flag string
    (the shared base plus only this arm's own), the CPU code generator, and how library nodes lower
    (BLAS/LAPACK/linalg via config, Reduce/ArgReduce via class attribute).

    The code generator is per arm, not global: ``auto_optimize`` is measured on ``legacy`` and
    ``canonicalize`` on ``experimental_readable``, each on the generator it is developed against. The
    serialize arms pin ``experimental_readable`` for a different reason -- they need its
    ``__restrict__``, since neither GCC's auto-parallelizer nor Polly can disambiguate memory without
    it, and a stray config file selecting ``legacy`` would silently halve those two columns.

    Goes through ``DACE_*`` environment rather than ``Config.set`` deliberately: ``Config.get``
    consults the environment FIRST, so a config write would be silently ignored under the submit
    script, which exports ``DACE_compiler_cpu_args``.
    """
    keys = ('DACE_compiler_cpu_executable', 'DACE_compiler_cpu_args', 'DACE_compiler_cpu_implementation',
            'DACE_library_blas_default_implementation', 'DACE_library_lapack_default_implementation',
            'DACE_library_linalg_default_implementation', 'OPENBLAS_DIR', 'DACE_compiler_emit_tree_reductions')
    saved = {k: os.environ.get(k) for k in keys}
    os.environ['DACE_compiler_cpu_executable'] = arm.executable
    os.environ['DACE_compiler_cpu_args'] = f'{_BASE_ARGS} {arm.flags}'.strip()
    os.environ['DACE_compiler_cpu_implementation'] = arm.codegen
    # Rides the codegen axis by request: auto_optimize is measured as it ships (legacy codegen, no
    # tree reductions), canonicalize with tree reductions on the experimental generator.
    os.environ['DACE_compiler_emit_tree_reductions'] = '0' if arm.codegen == 'legacy' else '1'
    # Per-COMPILER OpenBLAS, when the job provides both: a clang arm gets an OpenBLAS linked against
    # libomp and a gcc arm one against libgomp, so an arm's own BLAS never drags the OTHER OpenMP
    # runtime into the process. The build links the full .so path, so cmake bakes the dir into the
    # kernel's RUNPATH -- which only decides if no libopenblas is on LD_LIBRARY_PATH (it overrides
    # RUNPATH); the submit script therefore keeps both dirs OFF it.
    per_cc = os.environ.get('CANON_PERF_OPENBLAS_LLVM' if arm.executable == _CLANG else 'CANON_PERF_OPENBLAS_GCC')
    if per_cc:
        os.environ['OPENBLAS_DIR'] = per_cc
    # Spans the TRANSFORM, not just the build: ``CS.build`` runs the arm's pipeline inside this
    # context, and ``serialize`` calls ``expand_library_nodes()`` there -- so a library node is
    # lowered under whichever implementation is live at that moment, and setting this any later
    # would expand every gemv to ``pure`` before the setting was ever read.
    for lib in ('blas', 'lapack', 'linalg'):
        os.environ[f'DACE_library_{lib}_default_implementation'] = arm.blas
    # ``dace.libraries.standard`` has no ``library.*`` config key, so Reduce/ArgReduce are pinned on
    # the node classes. Saved and restored like the env vars: these are process-global class
    # attributes, so leaking one would silently re-lower the NEXT arm's reductions.
    try:
        yield
    finally:
        for k, old in saved.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


def resolve_autopar_hint(arm: Arm) -> Arm:
    """Return ``arm`` with the largest ``-ftree-parallelize-loops`` width that actually parallelizes.

    MEASURED on g++ 15.2 (x86) and again on g++ 16.1 (aarch64): with Graphite in the same command
    line, parloops stops parallelizing a known-good affine nest above roughly 33 threads and emits
    NO ``GOMP_parallel`` at all, silently turning this arm into plain ``-O3``. Without Graphite it
    parallelizes at 72, so the interaction is Graphite's rather than the thread count's, and neither
    ``-floop-parallelize-all`` nor ``--param parloops-min-per-thread`` overrides it. The cliff moves
    with the compiler build, so it cannot be defaulted around with a fixed cap.

    The width is NOT a free knob, and an earlier revision of this docstring had it backwards: it is
    the arm's REAL runtime width. parloops emits the number as a literal ``num_threads`` argument
    (``mov w2, <N>; bl GOMP_parallel`` on aarch64, verified by disassembling the probe object), so
    the region does NOT widen to ``OMP_NUM_THREADS``. Every step down therefore costs real
    parallelism and understates this one column against every other one. That is why the search walks
    a fine ladder near the top and why the chosen width is printed and recorded: a narrower arm is a
    legitimate measurement only when the reader can see how narrow.

    The ladder is dense above 32 because the cliff is sharp, not gradual: on g++ 16.1 / Neoverse V2
    the largest width that still parallelizes is exactly 40 (41 emits nothing), so a coarse
    72 -> 48 -> 32 descent silently gives up 8 of the 40 threads it could have had.

    Probes descending widths and takes the first that emits a parallel region; raising is left to the
    caller when none does.
    """
    hint = _AUTOPAR_HINT
    for candidate in [w for w in (hint, 64, 56, 48, 44, 40, 36, 32, 24, 16, 8, 4) if w <= hint]:
        probe = arm._replace(
            flags=arm.flags.replace(f'-ftree-parallelize-loops={hint}', f'-ftree-parallelize-loops={candidate}'))
        try:
            evidence = probe_arm(probe)
        except RuntimeError:
            continue
        if candidate != hint:
            print(f'NOTE: {arm.label} parallelizes at -ftree-parallelize-loops={candidate}, not '
                  f'{hint}: this g++ build suppresses parloops above that width when Graphite is on. '
                  f'That number is baked into the object as the region\'s num_threads, so THIS arm '
                  f'runs on {candidate} threads while every other arm runs on {_THREADS} -- its '
                  f'speedup column is understated by roughly {int(_THREADS) / candidate:.2f}x.')
        _EVIDENCE[probe.label] = evidence
        return probe
    return arm


def register_arms(arms: tuple[Arm, ...]) -> dict[str, str]:
    """Make ``arms`` the whole measured table; return ``label -> engagement evidence``.

    Probes EVERY arm before registering any, so an unusable toolchain stops the sweep up front
    instead of after hours of compiling.
    """
    # The VALUE is a choice; the explicit pin is not. gcc defaults -ffp-contract to fast and clang to
    # on, so an override that drops the flag makes the two compiler columns of one pipeline fuse
    # differently and the table stops comparing like with like.
    if '-ffp-contract=' not in _BASE_ARGS:
        raise RuntimeError('every arm needs an explicit -ffp-contract= in DACE_compiler_cpu_args; the '
                           'gcc and clang defaults differ, so omitting it makes the compiler columns '
                           f'incomparable. Got: {_BASE_ARGS!r}')
    if '-ffast-math' in _BASE_ARGS:
        raise RuntimeError('-ffast-math must not be in DACE_compiler_cpu_args: it enables associative '
                           'math, so each compiler reassociates reductions its own way and the columns '
                           f'no longer evaluate the same expression. Got: {_BASE_ARGS!r}')
    # The gcc autopar width is resolved against THIS compiler before anything is probed for real:
    # the Graphite suppression cliff moves between g++ builds, and a width past it turns the arm
    # into plain -O3.
    arms = tuple(
        resolve_autopar_hint(a) if a.executable == _GCC and '-ftree-parallelize-loops=' in a.flags else a for a in arms)
    # An arm that ASKS for OpenBLAS and silently gets ``pure`` is the same class of fabrication the
    # autopar probe exists to prevent: the label credits a tuned kernel and the number is a naive
    # triple loop. Detection is the same code the build uses, so this cannot pass here and fail there.
    wants_blas = sorted({arm.blas for arm in arms} - {'pure'})
    if wants_blas and not OpenBLAS.is_installed():
        raise RuntimeError(f'arms {[a.label for a in arms if a.blas != "pure"]} ask for '
                           f'{wants_blas} but no OpenBLAS resolves. DaCe would expand every gemv/gemm '
                           'to its naive `pure` maps and the DaCe columns would lose to numpy by '
                           '20-40x for that reason alone. Point OPENBLAS_DIR at the install (or put '
                           'libopenblas.so on LD_LIBRARY_PATH); set CANON_PERF_DACE_BLAS=pure to '
                           'measure the naive expansion on purpose.')
    evidence = {arm.label: probe_arm(arm) for arm in arms}
    _PIPELINES.clear()  # a registration is the WHOLE table: every arm's toolchain is pinned, none implicit
    _ARMS.clear()
    for arm in arms:
        _PIPELINES[arm.label] = arm.transform
        _ARMS[arm.label] = arm
    _EVIDENCE.update(evidence)
    return evidence


def arms_requested(value: str) -> bool:
    """Whether ``value`` (a CLI flag or ``CANON_PERF_ARMS``) asks for the full nine-arm table.

    Any non-empty, non-false value counts, including the job-shape names this harness used to take
    -- a batch job that still exports ``CANON_PERF_ARMS=pipelines`` gets the full table rather than a
    crash halfway through a node allocation.
    """
    return value not in ('', '0', 'false', 'False')


#: Two g++ DaCe pipelines by default; the full table only on request. Enabling here raises on an
#: unusable toolchain: the intended loud failure, since a silently-disabled arm is a fabricated
#: number. Registration also pins the toolchain of the default pair, so a label means the same
#: measurement whether or not the external arms were asked for.
register_arms(ARMS if arms_requested(os.environ.get('CANON_PERF_ARMS', '')) else _DEFAULT_ARMS)

_HOST = socket.gethostname()
#: Appended-to only. Every pre-existing column keeps its exact meaning, so an old CSV consumer
#: still reads a new CSV; ``reference_*`` / ``speedup_vs_reference`` are blank in old-shape data.
_CSV_FIELDS = [
    'suite', 'kernel', 'preset', 'pipeline', 'correct', 'min_ms', 'median_ms', 'mean_ms', 'std_ms',
    'speedup_vs_baseline', 'reps', 'omp', 'host', 'timestamp', 'error', 'reference_kind', 'reference_min_ms',
    'speedup_vs_reference'
]


class _Timeout(Exception):
    pass


def _now():
    return datetime.datetime.now().isoformat(timespec='seconds')


def _result_path(suite, name):
    return os.path.join(_RESULTS_DIR, f'{suite}_{name}.json')


def _shape_of(ctx):
    """The dataset shape (symbol -> value) for the kernel/preset, for the record."""
    if ctx['suite'] == 'poly':
        return {str(k): int(v) for k, v in ctx['psize'].items()}
    return {k: (v if isinstance(v, float) else int(v)) for k, v in ctx['params'].items()}


def _aggregate(times_ms):
    a = np.asarray(times_ms, dtype=float)
    return dict(
        min_ms=round(float(a.min()), 6),
        median_ms=round(float(np.median(a)), 6),
        mean_ms=round(float(a.mean()), 6),
        std_ms=round(float(a.std(ddof=1)) if a.size > 1 else 0.0, 6),
    )


def _save_pre_expansion(ctx, suite, name, preset):
    """Save each pipeline's pre-expansion SDFG (opt-in) to ``<results>/sdfg/`` for
    inspection. Best-effort: a pipeline that fails to build is reported by the
    timing path, so its snapshot is simply skipped here."""
    out_dir = os.path.join(_RESULTS_DIR, 'sdfg')
    os.makedirs(out_dir, exist_ok=True)
    for label, fn in _PRE_EXPANSION.items():
        try:
            s = CS.build(ctx, fn, f'{label}_pre')
            s.save(os.path.join(out_dir, f'{suite}_{name}_{preset}_{label}.sdfg'))
        except Exception:
            continue


#: cupy's default allocator calls ``cudaMalloc`` per array, so a GPU pipeline would time driver
#: allocation instead of the kernel. Set once, and only if a pipeline already pulled cupy in --
#: importing it here would cost a CUDA init on the CPU-only sweeps this harness usually runs.
_CUPY_POOLED = False


def _pool_cupy():
    """Route cupy device + pinned allocations through pools. No-op without cupy or a device."""
    global _CUPY_POOLED
    cupy = sys.modules.get('cupy')
    if _CUPY_POOLED or cupy is None:
        return
    _CUPY_POOLED = True
    try:
        cupy.cuda.set_allocator(cupy.cuda.MemoryPool().malloc)
        cupy.cuda.set_pinned_memory_allocator(cupy.cuda.PinnedMemoryPool().malloc)
    except Exception:  # no device, driver mismatch -- pooling is an optimization, never fatal
        pass


def _reps_for(est_ms: float) -> int:
    """How many timed reps a kernel costing ``est_ms`` per call should get.

    A FLAT rep count does not survive this corpus: it spans kernels from ~50us to ~20s, and the
    expensive tail is where a flat count explodes. jacobi_2d at its paper shape (tsteps=1000,
    N=2800) is ~20s per call, so 50 reps x 7 arms is ~2 wall-clock HOURS for one kernel -- measured,
    not estimated: it and heat_3d together took 6.4h of a 10h sweep while the other 79 kernels
    shared the rest.

    Reps buy a tighter median, and what they buy falls off fast: run-to-run spread on a 20s kernel
    is a fraction of a percent (it is one long steady-state loop nest, not a 50us kernel where timer
    granularity and cache state dominate), so rep 6 through rep 50 re-confirm a number rep 5 already
    had. Spending the budget on the cheap kernels, where the noise actually is, is strictly better
    measurement per CPU-hour. The floor is never crossed, so even the most expensive kernel keeps
    enough samples for a median to mean something, and the per-entry ``reps`` is recorded next to
    every timing so a reader can see which regime produced it.
    """
    if est_ms <= 0:  # timer granularity: too fast to size, so it is cheap -- give it the full count
        return _REPS
    return max(_MIN_REPS, min(_REPS, int(_BUDGET_MS / est_ms)))


def _time_all_reps(ctx, sdfg):
    """All timed repetitions for one compiled SDFG, in milliseconds.

    The call arguments are built ONCE per pipeline and reused across every repetition -- there is
    no per-rep allocation to pool away, and re-initializing them per rep would both change what is
    measured and put the memcpy inside the timed region.
    """
    _pool_cupy()
    cs, kw = CS.compiled_call(ctx, sdfg)
    # One untimed call to size the rep count. It doubles as an extra warmup -- it faults in the
    # output pages and warms the caches -- so it costs nothing that ``warmup`` was not already
    # paying, and its own duration is discarded rather than pooled into the reported statistics.
    t0 = time.perf_counter()
    cs(**kw)
    est_ms = (time.perf_counter() - t0) * 1000.0
    if est_ms > _MAX_CALL_MS:
        # One call already blew the budget, so there is nothing to learn from four more of them.
        # Report the single sample and say so: the fix belongs in the kernel's paper shape.
        print(
            f'      OVER BUDGET: one call took {est_ms / 1000:.1f}s > {_MAX_CALL_MS / 1000:.0f}s '
            f'-- reporting reps=1; this kernel\'s paper size needs lowering',
            flush=True)
        return np.asarray([est_ms], dtype=float), True
    reps = _reps_for(est_ms)
    # Warmup is FULL calls, so on the expensive tail it is priced like reps: two of them on a 20s
    # stencil cost 40s, as much as the entire rep floor. The call above has already warmed the
    # caches and faulted the pages, which is what warmup is for, so once a kernel is expensive
    # enough to be sized down the second warmup buys nothing but wall clock.
    warmup = _WARMUP if reps > _MIN_REPS else min(_WARMUP, 1)
    with dace.profile(repetitions=reps, warmup=warmup, print_results=False) as prof:
        cs(**kw)
    _report, times = prof.times[-1]
    # dace.profile reports per-call wall time in ms
    return np.asarray(times, dtype=float), False


#: What each corpus's timed reference ACTUALLY is. They are NOT interchangeable denominators:
#:  * ``np`` and ``poly`` are real numpy -- the framework-independent baseline npbench itself
#:    reports against, and (since ``polybench_numpy``) the same kind of number for polybench, so the
#:    two corpora finally divide by the same THING rather than one of them by compiled C++.
#:  * ``tsvc`` / ``tsvc25`` oracles are SCALAR PYTHON loops. Timing those measures CPython, would
#:    read as speedups in the hundreds, and would silently invalidate the figure.
REFERENCE_KIND = {'np': 'numpy', 'poly': 'numpy', 'tsvc': 'python-scalar', 'tsvc25': 'python-scalar'}

#: Speedup denominator per corpus. polybench/npbench divide by the timed corpus reference;
#: tsvc/tsvc_2_5 divide by the ``seq-cpp`` arm -- their oracles are scalar Python, so sequential
#: C++ from the same post-simplify SDFG is the only defensible in-repo baseline. The two groups
#: therefore answer DIFFERENT questions and must never share one undifferentiated plot series.
DENOMINATOR = {'poly': 'reference', 'np': 'reference', 'tsvc': 'seq-cpp', 'tsvc25': 'seq-cpp'}


def reference_kind(suite: str, name: str) -> str:
    """What THIS kernel's reference is, which is not always what its corpus's is.

    polybench's numpy references are not uniformly vectorized: ``polybench_numpy.VECTORIZATION``
    records one as a genuine scalar python loop (``seidel_2d``: an in-row Gauss-Seidel recurrence
    with no array form). That one is reported as ``python-scalar`` -- the same label the tsvc oracles
    carry -- so it is never timed and never divided by, instead of quietly contributing a speedup in
    the hundreds to the polybench geomean.
    """
    if suite == 'poly' and PBN.VECTORIZATION.get(name) == 'scalar':
        return 'python-scalar'
    return REFERENCE_KIND.get(suite, 'unknown')


def numpy_call(ctx: dict) -> tuple[Callable[..., Any], dict[str, Any]]:
    """``(fn, kwargs)`` for this kernel's numpy reference, resolved ONCE outside the timed region.

    Both corpora resolve the reference's parameters by name against the input arrays and the dataset
    symbols -- the corpus adapters' own rule, so the timed call is the call the correctness gate
    compared against.
    """
    if ctx['suite'] == 'poly':
        return PB.numpy_call(ctx['k'], ctx['arrays'], ctx['psize'])
    ref = ctx['c']['reference']
    work = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ctx['arrays'].items()}
    return ref, npbench._map_call(ref, work, ctx['params'])


def time_reference(ctx: dict) -> np.ndarray:
    """All timed repetitions of this kernel's NUMPY reference, in milliseconds.

    Treated EXACTLY like an arm: same warmup, same repetition count, call arguments resolved once
    outside the timed region. Both corpora write their input arrays in place, so the inputs are
    restored BETWEEN repetitions -- also outside the timed region: without it repetition ``k+1``
    computes from repetition ``k``'s output, which is different data and therefore a different time;
    inside it, the memcpy lands in the denominator and inflates every speedup on the figure.

    BLAS threads are deliberately NOT pinned down here: the agreed baseline is *parallel* numpy.
    """
    suite = ctx['suite']
    if DENOMINATOR.get(suite) != 'reference':
        raise RuntimeError(f"{suite} has no timeable reference; its denominator is {DENOMINATOR[suite]!r}")
    fn, call = numpy_call(ctx)
    for _ in range(_WARMUP):
        PB.restore_inputs(call, ctx['arrays'])
        fn(**call)
    times = []
    for _ in range(_REPS):
        PB.restore_inputs(call, ctx['arrays'])
        t0 = time.perf_counter()
        fn(**call)
        times.append((time.perf_counter() - t0) * 1000.0)
    return np.asarray(times, dtype=float)


def _denominator(suite: str, pres: dict) -> dict:
    """The speedup denominator for one (corpus, preset): timed reference, or the ``seq-cpp`` arm.

    Recorded explicitly -- ``kind`` names WHAT the number is divided by -- so no plot can pool a
    "3x over numpy" with a "3x over optimized sequential C++" and present them as one claim.
    """
    source = DENOMINATOR.get(suite, 'reference')
    if source == 'reference':
        ref = pres.get('reference', {})
        kind = ref.get('kind', 'unknown')
        # A scalar-python reference is reported but never divided by, whatever it timed at.
        min_ms = None if kind in ('python-scalar', 'unknown') else ref.get('min_ms')
        return dict(source='reference', kind=kind, min_ms=min_ms)
    entry = pres['pipelines'].get(source, {})
    min_ms = entry.get('min_ms') if entry.get('correct') else None
    return dict(source=source, kind='sequential-c++', min_ms=min_ms)


def usable_ratios(ratios: list[float]) -> list[float]:
    """The entries of ``ratios`` a geomean may be taken over: finite and strictly positive.

    A zero or non-finite ratio is a kernel too fast to time or an arm that never produced a number;
    ``log(0)`` would take the whole geomean to zero and a substituted ``1.0`` would drag it toward
    parity, so both are dropped and ``n`` shrinks to say so.
    """
    return [r for r in ratios if isinstance(r, (int, float)) and math.isfinite(r) and r > 0]


def geomean(ratios: list[float]) -> float:
    """Geometric mean of RAW ratios. Never fed the signed display scale, which spans zero and
    negatives: a geomean there is undefined where ``s <= 0`` and meaningless elsewhere."""
    good = usable_ratios(ratios)
    return math.exp(sum(map(math.log, good)) / len(good)) if good else float('nan')


def signed_speedup(ratio: float) -> float:
    """Ratio -> the signed symmetric scale the plot uses: parity 0, 2x faster +1, 2x slower -1."""
    if not (isinstance(ratio, (int, float)) and math.isfinite(ratio) and ratio > 0):
        return float('nan')
    return ratio - 1.0 if ratio >= 1.0 else -(1.0 / ratio - 1.0)


def geomeans(records: dict[tuple, dict], preset: str, key: str = 'speedup_vs_reference') -> dict[tuple, tuple]:
    """``(suite, arm) -> (geomean ratio, n)`` for one preset.

    Kernels that errored, miscompiled or were never measured contribute NOTHING -- no substituted
    1.0, which would silently drag the geomean toward parity -- so ``n`` is the honest count of
    kernels actually behind the number. Kept per corpus because the denominator is per corpus.
    """
    per: dict[tuple, list[float]] = {}
    for (suite, _name), r in records.items():
        pres = r.get('presets', {}).get(preset) or {}
        for label, ratio in (pres.get(key) or {}).items():
            per.setdefault((suite, label), []).append(ratio)
    return {k: (geomean(v), len(usable_ratios(v))) for k, v in per.items()}


def restore_affinity() -> None:
    """Put the process back on every CPU it started with.

    Loading an OpenMP-threaded OpenBLAS under OMP_PROC_BIND=close binds the calling thread and
    shrinks the PROCESS mask to one core (measured: 288 cpus before the load, 1 after). Any DaCe
    build or reference computation can trigger that, so this runs immediately before each arm is
    built and timed rather than once at startup -- a restore that happens before the next OpenBLAS
    load is simply undone by it.
    """
    if _FULL_AFFINITY and hasattr(os, 'sched_setaffinity'):
        try:
            if frozenset(os.sched_getaffinity(0)) != _FULL_AFFINITY:
                os.sched_setaffinity(0, set(_FULL_AFFINITY))
        except OSError:  # a cgroup may forbid widening; measuring narrow beats not measuring
            pass


def run_arm(ctx, label: str, deadline: float) -> dict:
    """Build, correctness-gate and time ONE arm; return its result entry.

    Never raises: a failure is the arm's own ``error`` field, so one bad arm cannot lose the eight
    that worked. Bounded by its own alarm, clamped to the kernel deadline the caller set.
    """
    entry: dict[str, Any] = {}
    restore_affinity()
    arm_t0 = time.perf_counter()
    try:
        # Bound EVERY arm, not just the kernel: the kernel alarm is a single shot and the handler
        # below absorbs it, so without re-arming, every arm after the first timeout ran unbounded.
        signal.alarm(max(1, int(min(_PER_ARM_TIMEOUT, deadline - time.monotonic()))))
        # The compiler override has to span the whole arm: ``run_matches`` compiles too, and
        # correctness must gate the very binary that gets timed.
        with toolchain_env(_ARMS[label]):
            sdfg = CS.build(ctx, _PIPELINES[label], arm_tag(label))
            entry['correct'] = bool(CS.run_matches(ctx, sdfg))
            if entry['correct']:
                times, over_budget = _time_all_reps(ctx, sdfg)
                entry.update(_aggregate(times))
                if over_budget:
                    # Travels with the number so a reader never takes a one-sample timing for a
                    # best-of-N, and so the offending kernel is findable in the result files.
                    entry['over_budget'] = True
                    entry['budget_ms'] = _MAX_CALL_MS
                # Per-entry, because the count is adaptive: the record-level ``reps`` is a ceiling.
                entry['reps'] = int(times.size)
                entry['times_ms'] = [round(float(x), 6) for x in times]
        print(
            f'    {label}: {time.perf_counter() - arm_t0:.1f}s '
            f'(reps={entry.get("reps", 0)}, min={entry.get("min_ms", "-")}ms)',
            flush=True)
    except Exception as e:
        entry['error'] = f'{type(e).__name__}: {str(e)[:120]}'
        print(f'    {label}: FAILED {entry["error"]}', flush=True)
    return entry


def arms_worker(suite: str, name: str, preset: str, labels: list[str], budget: float, affinity: frozenset) -> dict:
    """Child entry point: rebuild the dataset and measure ``labels``. Runs in a SPAWNED process.

    Spawned rather than forked on purpose -- a fork inherits the parent's already-mapped OpenMP
    runtime, which is the very thing being kept apart. A fresh interpreter loads only the runtime
    its own arms pull in.

    The dataset is rebuilt here because a ctx holds live numpy arrays that would have to cross the
    process boundary; re-deriving it costs one allocation and one reference computation per child,
    which is far below the co-residency penalty it avoids.
    """
    global _FULL_AFFINITY
    if affinity:
        # Remember it for restore_affinity(); the restore itself happens per arm, NOT here. Doing it
        # only at child start is too early: CS.make below loads OpenBLAS, which re-collapses the mask
        # straight after -- measured as the gcc arms staying 20x slow while the llvm arms recovered.
        _FULL_AFFINITY = frozenset(affinity)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(_Timeout()))
    ctx = CS.make(suite, name, preset)
    deadline = time.monotonic() + budget
    out = {label: run_arm(ctx, label, deadline) for label in labels}
    signal.alarm(0)
    return out


def measure_arms(ctx, preset: str, labels: list[str], deadline: float) -> dict:
    """Measure ``labels`` for the kernel in ``ctx``, one process per OpenMP runtime family.

    The grouping is the whole point: libgomp and libomp in one process cost ~34x (see
    ``_ISOLATE_ARMS``), so the gcc arms and the clang arms must never share an interpreter. Arms are
    returned in ``labels`` order regardless of which child produced them, so the record's column
    order does not depend on the grouping.

    Falls back to measuring in-process when isolation is off, when this IS the child, or when a
    child dies outright -- a lost child would otherwise silently drop a whole family of columns.
    """
    if not _ISOLATE_ARMS or _IS_CHILD:
        return {label: run_arm(ctx, label, deadline) for label in labels}
    families: dict[str, list[str]] = {}
    for label in labels:
        families.setdefault(arm_omp_runtime(_ARMS[label]), []).append(label)
    budget = max(1.0, deadline - time.monotonic())
    env = dict(os.environ, CANON_PERF_ARM_CHILD='1')
    out: dict[str, dict] = {}
    mp = multiprocessing.get_context('spawn')
    for family, group in families.items():
        try:
            with _child_env(env):
                with mp.Pool(1) as pool:
                    out.update(
                        pool.apply(arms_worker, (ctx['suite'], ctx['name'], preset, group, budget, _FULL_AFFINITY)))
        except Exception as e:
            # Never lose a family to a crashed child: fall back to measuring it here. The numbers
            # are then co-resident-contaminated, which is why it says so rather than staying quiet.
            print(
                f'    [{family}] child failed ({type(e).__name__}: {str(e)[:80]}); '
                f'measuring in-process -- these columns are NOT runtime-isolated',
                flush=True)
            out.update({label: run_arm(ctx, label, deadline) for label in group})
    return {label: out[label] for label in labels if label in out}


@contextlib.contextmanager
def _child_env(env: dict) -> Iterator[None]:
    """Apply ``env`` for the duration of a child spawn: ``spawn`` copies os.environ at fork time."""
    saved = dict(os.environ)
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def _measure_kernel(suite, name):
    """Measure every preset x pipeline for one kernel; return the result record.

    Each pipeline is correctness-gated against the suite reference; a pipeline that
    errors or miscompiles is recorded (``correct=false`` / ``error``) and excluded
    from speedups instead of failing the whole kernel. Raises ``pytest.skip`` if
    nothing at all was measurable.
    """
    # Mirrors the alarm the caller arms for this kernel, so the per-arm bound below can tighten
    # that deadline but never extend past it.
    deadline = time.monotonic() + _PER_KERNEL_TIMEOUT
    result: dict[str, Any] = dict(
        suite=suite,
        kernel=name,
        host=_HOST,
        timestamp=_now(),
        omp_num_threads=os.environ.get('OMP_NUM_THREADS', ''),
        reps=_REPS,
        warmup=_WARMUP,
        baseline=BASELINE,
        # Which measurement METHOD produced these numbers; a record
        # from an older epoch is re-measured, never skipped.
        epoch=MEASUREMENT_EPOCH,
        presets={})
    # Provenance travels with every number: which SDFG, which compiler, which flags, and the line
    # proving the credited passes engaged. A speedup that cannot be read back to its toolchain is
    # not a measurement.
    result['arms'] = {
        label:
        dict(sdfg=arm.transform.__name__,
             cxx=arm.executable,
             args=f'{_BASE_ARGS} {arm.flags}'.strip(),
             engaged=_EVIDENCE.get(label, ''))
        for label, arm in _ARMS.items()
    }
    measured_any = False
    for preset in PRESETS:
        pres: dict[str, Any] = dict(shape=None, pipelines={}, speedup_vs_baseline={})
        try:
            ctx = CS.make(suite, name, preset)
        except Exception as e:  # input/reference build failure -> record and move on
            pres['error'] = f'{type(e).__name__}: {str(e)[:120]}'
            result['presets'][preset] = pres
            continue
        pres['shape'] = _shape_of(ctx)
        # Time the corpus reference only where it is actually the denominator, and only where it is
        # not itself a scalar python loop: timing those costs minutes to produce a number we must
        # not use. ``vectorization`` travels with the record because polybench's numpy references
        # are not uniformly array-level -- a "vs numpy" bar needs to be readable back to what it
        # divided by.
        ref: dict[str, Any] = dict(kind=reference_kind(suite, name))
        if suite == 'poly':
            ref['vectorization'] = PBN.VECTORIZATION.get(name, 'unknown')
        if DENOMINATOR.get(suite) == 'reference' and ref['kind'] != 'python-scalar':
            try:
                rtimes = time_reference(ctx)
                ref.update(_aggregate(rtimes))
                ref['times_ms'] = [round(float(x), 6) for x in rtimes]
            except Exception as e:
                ref['error'] = f'{type(e).__name__}: {str(e)[:120]}'
        pres['reference'] = ref
        if _SAVE_SDFG:
            _save_pre_expansion(ctx, suite, name, preset)
        for label, entry in measure_arms(ctx, preset, list(_PIPELINES), deadline).items():
            measured_any = measured_any or ('min_ms' in entry)
            pres['pipelines'][label] = entry
        # Back to the KERNEL bound: the last arm left its own alarm armed, and it would
        # otherwise fire while the speedups below are being computed and the record written.
        signal.alarm(max(1, int(deadline - time.monotonic())))
        # Speedups vs baseline (best-of-N min): >1 means the candidate is faster.
        base = pres['pipelines'].get(BASELINE, {})
        b_min = base.get('min_ms') if base.get('correct') else None
        for label, entry in pres['pipelines'].items():
            if label == BASELINE:
                continue
            if b_min and entry.get('correct') and entry.get('min_ms'):
                pres['speedup_vs_baseline'][label] = round(b_min / entry['min_ms'], 4)
        pres['denominator'] = _denominator(suite, pres)
        d_min = pres['denominator'].get('min_ms')
        pres['speedup_vs_reference'] = {
            label: round(d_min / entry['min_ms'], 4)
            for label, entry in pres['pipelines'].items() if d_min and entry.get('correct') and entry.get('min_ms')
        }
        result['presets'][preset] = pres
    if not measured_any:
        pytest.skip(f"{suite}:{name} not measurable (every pipeline errored or miscompiled)")
    return result


def _write_result(path, result):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    os.replace(tmp, path)  # atomic: never leave a half-written file for skip-existing


def _regressions(result):
    """List of human-readable regression strings (candidate grossly slower than baseline).

    Exempt: every arm that is not a DaCe-PARALLEL pipeline -- the two autopar arms and the
    deliberately sequential ``seq-cpp``. Being slower than ``auto_optimize`` is their expected
    outcome and the interesting result, not a DaCe regression. Every arm running ``auto_optimize``
    or ``canonicalize`` -- including the clang++ columns -- stays gated.
    """
    exempt = {label for label, arm in _ARMS.items() if arm.flags or arm.transform not in (_autoopt, _canon)}
    out = []
    for preset, pres in result['presets'].items():
        base = pres['pipelines'].get(result['baseline'], {})
        if not (base.get('correct') and base.get('min_ms')):
            continue
        b = base['min_ms']
        factor = REGRESSION_FACTOR if b >= _MIN_TIMEABLE_MS else 50.0
        for label, entry in pres['pipelines'].items():
            if label == result['baseline'] or label in exempt:
                continue
            if not entry.get('correct') or not entry.get('min_ms'):
                continue
            if entry['min_ms'] > factor * b:
                out.append(f"{label}[{preset}] {entry['min_ms']:.3f}ms > {factor:g}x "
                           f"{result['baseline']} {b:.3f}ms")
    return out


@functools.lru_cache(maxsize=None, typed=True)
def expected_paper_shape(suite: str, name: str, preset: str) -> tuple[tuple[str, int], ...] | None:
    """The dataset shape ``(suite, name)`` SHOULD have at ``preset``, or ``None`` if not checkable.

    Cheap on purpose -- it reads the kernel module's ``paper_sizes`` constant and allocates nothing
    -- because it runs inside the resume check, once per kernel, before anything is built.

    Only polybench's ``paper`` preset is answerable this way; the other suites derive their shapes
    inside ``make_inputs``, which allocates the arrays and computes the reference, and calling that
    just to decide whether to skip would cost more than the measurement it is trying to avoid.
    ``None`` therefore means "cannot tell", and the caller leaves such a record alone.
    """
    if suite != 'poly' or preset != 'paper':
        return None
    kernels = PB.collect(name)
    if not kernels:
        return None
    row = PB.paper_size_row(importlib.import_module(kernels[0].modpath))
    return tuple(sorted((str(k), int(v)) for k, v in row.items()))


def has_current_arms(path: str, suite: str = '', name: str = '') -> bool:
    """True when ``path`` already holds a CURRENT measurement of every registered arm.

    Plain file existence is NOT enough. The results directory carries files written before the arm
    table existed -- two pipelines, a different thread count, unpinned flags -- and skipping those
    would have the sweep silently report stale two-column data under a nine-arm heading. A preset
    whose dataset failed to build carries no pipelines and is left alone: re-running it would only
    fail the same way, every sweep.

    Neither is the arm set alone enough, which is the failure this second check exists to stop: a
    record can hold all nine arms and still be stale because the KERNEL changed under it. Editing a
    ``paper_sizes`` row leaves every arm present and every number measured at the old shape, so the
    sweep would skip it forever and the table would mix shapes without saying so -- exactly what
    happened when the stencil ``tsteps`` were levelled. Comparing the recorded shape against the one
    the corpus declares today makes a size edit invalidate its own results, with no ``--force`` to
    remember and no stale row surviving into the figure.
    """
    try:
        with open(path) as f:
            record = json.load(f)
    except (OSError, ValueError):
        return False
    # A record from an older measurement method is stale no matter how complete it looks. Epoch 2
    # is per-runtime process isolation: everything before it timed the clang arms in a process that
    # also held libgomp, so their numbers carry the ~34x co-residency penalty.
    if int(record.get('epoch', 1)) < MEASUREMENT_EPOCH:
        return False
    want = set(_PIPELINES)
    for preset in PRESETS:
        pres = (record.get('presets') or {}).get(preset)
        if pres is None:
            return False
        if pres.get('error'):
            continue
        pipelines = pres.get('pipelines') or {}
        if not want <= set(pipelines):
            return False
        # An arm that TIMED OUT is present but unmeasured, and presence is what the check above
        # tests -- so without this a transient timeout is baked in permanently: the record looks
        # complete, every later sweep skips it, and the column stays empty in the figure with
        # nothing saying why. A timeout is a property of the budget rather than of the kernel (it
        # fires on whichever arm happens to be running when the per-kernel alarm expires, and
        # heat_3d legitimately needs more than the default 600s), so it is worth retrying. Other
        # arm errors are NOT retried: a kernel that fails to build or miscompares under an arm
        # fails the same way every sweep, and retrying it would burn the budget forever.
        if any(_Timeout.__name__ in str(e.get('error', '')) for e in pipelines.values()):
            return False
        expect = expected_paper_shape(suite, name, preset) if suite and name else None
        if expect is not None:
            got = pres.get('shape') or {}
            if tuple(sorted((str(k), int(v)) for k, v in got.items())) != expect:
                return False
    return True


# ---------------------------------------------------------------------------
# pytest harness: one (resumable) speedup test per kernel.
# ---------------------------------------------------------------------------
#: Every pragma that would introduce a *second* level of parallelism under an auto-parallelizer.
#: ``omp simd`` and ``omp atomic`` are deliberately absent: they are vectorization and a reduction
#: guard, not a thread team.
_OMP_PARALLEL_PRAGMA = re.compile(r'#pragma\s+omp\s+(?:parallel|sections|task|target|teams)')


def emitted_pragmas(ctx: dict, transform: Callable[[dace.SDFG], dace.SDFG], tag: str) -> list[str]:
    """The thread-team OpenMP pragmas in the C++ ``transform`` makes DaCe emit. Codegen only."""
    sdfg = CS.build(ctx, transform, tag)
    return _OMP_PARALLEL_PRAGMA.findall('\n'.join(c.clean_code for c in sdfg.generate_code()))


@pytest.mark.parametrize("suite,name", [('poly', 'gemm'), ('np', 'arc_distance')])
def test_autopar_input_is_sequential(suite, name):
    """The two autopar arms must be handed genuinely SEQUENTIAL C++.

    DaCe resolves a default-scheduled top-level map to ``CPU_Multicore``, so the post-simplify SDFG
    already emits ``#pragma omp parallel`` (gemm: 5). Handing that to GCC's auto-parallelizer or to
    Polly measures nested parallelism at ``OMP_NUM_THREADS`` squared oversubscription, not the
    compiler, and the resulting column is unattributable. ``serialize`` therefore has to leave the
    emitted pragma set EMPTY, and this pins it. ``gemm`` carries a MatMul library node on purpose:
    library nodes are expanded again at codegen time and bring their own multicore maps, so
    ``serialize`` has to expand FIRST -- expanding late leaves gemm at 2 pragmas, not 0.
    """
    ctx = CS.make(suite, name, 'S')
    assert emitted_pragmas(ctx, untransformed, 'par'), \
        'precondition: the post-simplify SDFG is expected to be OpenMP-parallel'
    found = emitted_pragmas(ctx, serialize, 'ser')
    assert not found, f'{suite}:{name} still emits {len(found)} thread-team pragma(s) for the autopar arms: {found}'


@pytest.mark.perf
@pytest.mark.parametrize("suite,name", CS.kernels())
def test_speedup(suite, name):
    path = _result_path(suite, name)
    if not _FORCE and os.path.exists(path) and has_current_arms(path, suite, name):
        pytest.skip(f"already measured: {os.path.relpath(path)} "
                    f"(delete it or set CANON_PERF_FORCE=1 to re-run)")

    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(_Timeout()))
    signal.alarm(_PER_KERNEL_TIMEOUT)
    try:
        result = _measure_kernel(suite, name)  # may pytest.skip (build error / not measurable)
    except _Timeout:
        pytest.skip(f"{suite}:{name} exceeded {_PER_KERNEL_TIMEOUT}s (not measured; will retry)")
    finally:
        signal.alarm(0)

    _write_result(path, result)  # written only on success -> skip-existing is meaningful
    problems = _regressions(result)
    assert not problems, f"perf regression on {suite}:{name}: " + "; ".join(problems)


# ---------------------------------------------------------------------------
# CSV export: flat per-(suite, kernel, preset, pipeline) summary from result files.
# ---------------------------------------------------------------------------
def export_csv(csv_path, results_dir=None):
    """Aggregate every result JSON in ``results_dir`` into one summary CSV. Returns row count."""
    results_dir = results_dir or _RESULTS_DIR
    rows = []
    for fn in sorted(os.listdir(results_dir)) if os.path.isdir(results_dir) else []:
        if not fn.endswith('.json'):
            continue
        with open(os.path.join(results_dir, fn)) as f:
            r = json.load(f)
        baseline = r.get('baseline', BASELINE)
        for preset, pres in r.get('presets', {}).items():
            den = pres.get('denominator') or pres.get('reference', {})
            for label, entry in pres.get('pipelines', {}).items():
                speedup = 1.0 if label == baseline else pres.get('speedup_vs_baseline', {}).get(label)
                rows.append(
                    dict(
                        suite=r.get('suite'),
                        kernel=r.get('kernel'),
                        preset=preset,
                        pipeline=label,
                        correct=entry.get('correct'),
                        min_ms=entry.get('min_ms'),
                        median_ms=entry.get('median_ms'),
                        mean_ms=entry.get('mean_ms'),
                        std_ms=entry.get('std_ms'),
                        speedup_vs_baseline=speedup,
                        # Prefer the arm's OWN count: with adaptive reps the record-level value is
                        # just the ceiling, so reporting it would overstate the expensive kernels.
                        reps=entry.get('reps', r.get('reps')),
                        omp=r.get('omp_num_threads'),
                        host=r.get('host'),
                        timestamp=r.get('timestamp'),
                        error=entry.get('error', ''),
                        reference_kind=den.get('kind', ''),
                        reference_min_ms=den.get('min_ms'),
                        speedup_vs_reference=pres.get('speedup_vs_reference', {}).get(label)))
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or '.', exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


# ---------------------------------------------------------------------------
# Markdown table: the headline "speedup numbers in a table" deliverable.
# ---------------------------------------------------------------------------
def _md_correct(entry):
    """Correctness glyph for a pipeline entry: numerically verified / wrong / not run."""
    if entry.get('correct') is True:
        return '✓'
    if entry.get('correct') is False:
        return '✗'
    return '·'


def _md_time(entry):
    if entry.get('min_ms') is not None:
        return f"{entry['min_ms']:.3f}"
    if entry.get('correct') is False:
        return 'WRONG'
    if entry.get('error'):
        return 'ERR'
    return '—'


def _md_speed(speedups, label):
    v = speedups.get(label)
    return f"{v:.2f}×" if v else '—'


def _candidate_labels(records: dict[tuple, dict]) -> list[str]:
    """Non-baseline pipeline labels to give a column, in ``ARMS`` order.

    The aggregation step re-exports result files it did not produce, so an arm is picked up from
    the DATA as well as from ``_PIPELINES`` -- otherwise ``--no-run``, which deliberately registers
    no toolchain, would drop every column it did not itself register. That is also how a superseded
    label (``canon``, ``gcc-autopar``, ... -- see ``LEGACY_LABELS``) keeps a column of its own
    instead of being silently folded into the arm that replaced it under different flags. Ordering
    is the arm table's, not discovery order, so the columns are the same whether a process
    registered the arms or only read their JSON -- a plot may key on column position.
    """
    order = {arm.label: i for i, arm in enumerate(ARMS)}
    seen = set(_PIPELINES)
    seen.update(label for r in records.values() for pres in r.get('presets', {}).values()
                for label in pres.get('pipelines', {}))
    return sorted((label for label in seen if label != BASELINE),
                  key=lambda label: (order.get(label, len(order)), label))


def _md_geomean_block(records: dict[tuple, dict], preset: str) -> list[str]:
    """Per-corpus, per-arm geomean against that corpus's OWN denominator.

    Separate from the vs-baseline line above on purpose: ``poly``/``np`` divide by the corpus
    reference and ``tsvc``/``tsvc25`` by sequential C++, so one pooled row would mix two different
    questions. There is deliberately no combined row.
    """
    gm = geomeans(records, preset)
    if not gm:
        return []
    kinds = {}
    for (suite, _n), r in records.items():
        kind = ((r.get('presets', {}).get(preset) or {}).get('denominator') or {}).get('kind')
        if kind:
            kinds[suite] = kind
    out = [
        "", f"### preset `{preset}` \u2014 geomean vs each corpus's own denominator", "",
        "Geomean of the RAW ratio `denominator_min / arm_min`; the signed column is that one geomean "
        "converted for display (parity 0, `+1` = 2x faster, `-1` = 2x slower). Kernels that errored or "
        "miscompiled are excluded, so `n` is the count actually behind the number.", "",
        "| suite | denominator | arm | geomean | signed | n |", "|:--|:--|:--|--:|--:|--:|"
    ]
    for (suite, label), (ratio, n) in sorted(gm.items()):
        signed = signed_speedup(ratio)
        signed_s = f"{signed:+.3f}" if math.isfinite(signed) else "\u2014"
        ratio_s = f"{ratio:.3f}\u00d7" if math.isfinite(ratio) else "\u2014"
        out.append(f"| {suite} | {kinds.get(suite, '?')} | {label} | {ratio_s} | {signed_s} | {n} |")
    return out


def _load_records(results_dir: str | None = None) -> dict[tuple, dict]:
    """``(suite, kernel) -> result record`` for every JSON in the results directory."""
    results_dir = results_dir or _RESULTS_DIR
    records = {}
    for fn in sorted(os.listdir(results_dir)) if os.path.isdir(results_dir) else []:
        if not fn.endswith('.json'):
            continue
        with open(os.path.join(results_dir, fn)) as f:
            r = json.load(f)
        records[(r.get('suite'), r.get('kernel'))] = r
    return records


def export_markdown(md_path, results_dir=None):
    """Render every result JSON in ``results_dir`` into one GitHub-flavored Markdown
    file: per preset a kernel x pipeline table (best-of-N min ms, correctness glyph,
    speedup vs baseline) plus a geomean + correctness summary line. Returns the path."""
    records = _load_records(results_dir)
    # A record written under a superseded arm table keeps its own baseline label, so its rows are
    # rendered against THAT baseline and counted here. Silently mixing it into the current columns
    # would compare measurements taken at a different thread count and different compiler flags.
    stale = sorted(k for k, r in records.items() if r.get('baseline') != BASELINE)

    out = [
        f"# Corpus speedup — {len(ARMS) - 1} comparison arms, baseline `{BASELINE}`",
        "",
        f"host `{_HOST}` · OMP `{os.environ.get('OMP_NUM_THREADS', '')}` · "
        f"best-of `{_REPS}` (warmup {_WARMUP}) · kernels measured `{len(records)}` · generated `{_now()}`",
        "",
        f"Speedup = `{BASELINE}_min / arm_min` (best-of-N); **>1× means faster than the baseline**. "
        "Only numerically-correct arms are timed — ✓ verified, ✗ miscompile, · not measured.",
    ]
    labels = _candidate_labels(records)
    if stale:
        moved = ', '.join(f"`{old}` -> `{LEGACY_LABELS[old] or 'dropped'}`" for old in labels if old in LEGACY_LABELS)
        out += [
            "", f"> ⚠ `{len(stale)}` record(s) predate the current arm table (their `baseline` is not "
            f"`{BASELINE}`). Their superseded labels get their own columns rather than being folded "
            f"into the arms that replaced them ({moved}); re-run those kernels to refresh."
        ]
    for preset in PRESETS:
        head, sep = "| suite | kernel | shape | baseline ms |", "|:--|:--|:--|--:|"
        for label in labels:
            head, sep = head + f" {label} ms | ✓ | {label}× |", sep + "--:|:-:|--:|"
        out += ["", f"## preset `{preset}`", "", head, sep]
        speeds = {label: [] for label in labels}
        stats = {label: dict(ok=0, wrong=0, na=0) for label in labels}
        for suite, name in CS.kernels():
            r = records.get((suite, name)) or {}
            pres = r.get('presets', {}).get(preset)
            if not pres:
                continue
            pp = pres.get('pipelines', {})
            sp = pres.get('speedup_vs_baseline', {})
            shape = pres.get('shape') or {}
            shape_s = ' '.join(f"{k}={v}" for k, v in shape.items()) if isinstance(shape, dict) else str(shape)
            # A stale record's baseline column shows ITS baseline, not the current one, so the
            # speedups beside it stay divisible by the number in the same row.
            row = f"| {suite} | {name} | {shape_s} | {_md_time(pp.get(r.get('baseline', BASELINE), {}))} |"
            for label in labels:
                e = pp.get(label, {})
                row += f" {_md_time(e)} | {_md_correct(e)} | {_md_speed(sp, label)} |"
                stats[label]['ok' if e.get('correct') is True else 'wrong' if e.get('correct') is False else 'na'] += 1
                if sp.get(label):
                    speeds[label].append(sp[label])
            out.append(row)
        geo = ' · '.join(f"{label} `{geomean(speeds[label]):.3f}×` (n={len(usable_ratios(speeds[label]))})"
                         for label in labels)
        corr = ' · '.join(f"{label} {stats[label]['ok']}✓ {stats[label]['wrong']}✗ {stats[label]['na']}·"
                          for label in labels)
        out += ["", f"**geomean speedup** — {geo} · **correctness** — {corr}"]
        out += _md_geomean_block(records, preset)
    out.append("")
    os.makedirs(os.path.dirname(os.path.abspath(md_path)) or '.', exist_ok=True)
    with open(md_path, 'w') as f:
        f.write('\n'.join(out))
    return md_path


# ---------------------------------------------------------------------------
# Script entry point: run the (resumable) sweep, print a table, optional CSV.
# ---------------------------------------------------------------------------
def print_summary() -> None:
    """Per-preset, per-corpus, per-arm geomean to stdout.

    Only the geomeans: the per-kernel table is the Markdown export, and the sweep already prints a
    line per kernel as it goes, so rendering the whole grid a third time was pure duplication.
    """
    records = _load_records()
    for preset in PRESETS:
        print(
            f"\n### preset={preset}  (best-of-{_REPS}, warmup={_WARMUP}, "
            f"OMP={os.environ.get('OMP_NUM_THREADS')}, kernels={len(records)})",
            flush=True)
        for (suite, label), (ratio, n) in sorted(geomeans(records, preset).items()):
            print(f"  [{suite:6}] {label:28} geomean {ratio:8.3f}x  signed {signed_speedup(ratio):+8.3f}  n={n}",
                  flush=True)


def _run_sweep(only: str = '', force: bool = False, suite: str = '', shard: tuple = (0, 1), limit: int = 0) -> None:
    """Measure every selected kernel, skipping those already measured with the CURRENT arm table.

    ``suite`` restricts to a COMMA-SEPARATED set of corpus tags (``poly,np``) and ``shard=(i, n)``
    keeps every ``n``-th kernel starting at ``i`` -- the same fan-out primitive
    ``tests.corpus.measure_parallelization`` takes, so a batch job shards both facets the same way.
    ``limit`` keeps the first N of EACH corpus BEFORE sharding, for smoke runs.

    A SET rather than one tag because the corpora have very different cost profiles: polybench and
    npbench are a few hundred big kernels that divide by numpy, while tsvc/tsvc25 are ~200 small
    ones that divide by ``seq-cpp``. Splitting them into separate jobs lets each get a time limit
    and a results directory that suit it, and keeps a long polybench sweep from eating the wall
    clock the tsvc kernels needed. One tag still works, so every existing caller is unaffected.
    """
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    want = {t.strip() for t in suite.split(',') if t.strip()}
    kernels = [(s, n) for s, n in CS.kernels() if (not want or s in want) and (not only or only in n)]
    if limit:
        # Per corpus, not off the pooled head: the pool is ordered poly, np, tsvc, tsvc25, so a
        # pooled truncation is 30 polybench kernels before it reaches another corpus, and a smoke
        # run would exercise one corpus while reporting itself as a sweep. Matches the parallelism
        # facet, so both facets select the same kernels.
        seen: dict[str, int] = {}
        kept = []
        for s, n in kernels:
            if seen.get(s, 0) < limit:
                seen[s] = seen.get(s, 0) + 1
                kept.append((s, n))
        kernels = kept
    kernels = kernels[shard[0]::shard[1]]
    for i, (suite, name) in enumerate(kernels, 1):
        path = _result_path(suite, name)
        if not force and os.path.exists(path) and has_current_arms(path, suite, name):
            print(f"[{i}/{len(kernels)}] skip {suite}:{name} (all {len(_PIPELINES)} arms already measured)", flush=True)
            continue
        print(f"[{i}/{len(kernels)}] measure {suite}:{name} ...", flush=True)
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(_Timeout()))
        signal.alarm(_PER_KERNEL_TIMEOUT)
        try:
            result = _measure_kernel(suite, name)
        except (_Timeout, BaseException) as e:  # incl. pytest.skip -> just don't write a file
            print(f"      not measured: {type(e).__name__}: {str(e)[:80]}", flush=True)
            continue
        finally:
            signal.alarm(0)
        _write_result(path, result)
        probs = _regressions(result)
        if probs:
            print(f"      REGRESSION: {'; '.join(probs)}", flush=True)


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description=(__doc__ or '').splitlines()[0])
    ap.add_argument('--csv', metavar='PATH', help='export a flat summary CSV from the result dir')
    ap.add_argument('--markdown',
                    metavar='PATH',
                    help='write a Markdown speedup table (default <dir>/speedup_table.md)')
    ap.add_argument('--force', action='store_true', help='re-measure even if a result file exists')
    ap.add_argument('--only', metavar='SUBSTR', help='only kernels whose name contains SUBSTR')
    ap.add_argument('--suite',
                    metavar='TAGS',
                    default='',
                    help='only these corpora, comma-separated (poly, np, tsvc, tsvc25); default all')
    ap.add_argument('--shard', metavar='I/N', default='0/1', help='only every N-th selected kernel starting at I')
    ap.add_argument('--limit', metavar='N', type=int, default=0, help='only the first N kernels of each corpus')
    ap.add_argument('--dir', metavar='PATH', help=f'results directory (default {_RESULTS_DIR})')
    ap.add_argument('--no-run', action='store_true', help='skip measuring; only (re)export CSV / tables')
    ap.add_argument('--save-sdfg',
                    action='store_true',
                    help='also save each pipeline SDFG before libnode expansion to <dir>/sdfg/')
    ap.add_argument('--arms',
                    action='store_true',
                    help=f'time all {len(ARMS)} arms (the eight compared + seq-cpp) instead of only the '
                    'two g++ DaCe pipelines; aborts if a toolchain cannot prove its passes ran')
    args = ap.parse_args()
    if args.dir:
        _RESULTS_DIR = args.dir
    if args.save_sdfg:
        _SAVE_SDFG = True
    if args.arms:
        for _label, _line in register_arms(ARMS).items():
            print(f"arm {_label} engaged: {_line}", flush=True)
    if not args.no_run:
        index, total = (int(p) for p in args.shard.split('/'))
        _run_sweep(only=args.only or '', force=args.force, suite=args.suite, shard=(index, total), limit=args.limit)
    print_summary()
    md_out = args.markdown or os.path.join(_RESULTS_DIR, 'speedup_table.md')
    export_markdown(md_out)
    print(f"\nwrote speedup table to {md_out}", flush=True)
    csv_out = args.csv or os.environ.get('CANON_PERF_CSV')
    if csv_out:
        n = export_csv(csv_out)
        print(f"wrote {n} rows to {csv_out}", flush=True)
