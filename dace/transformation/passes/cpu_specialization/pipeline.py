# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The CPU specialization stage: the one place a canonical parallel scope is made sequential.

Canonicalization and specialization are two stages, in that order, and this module is the second
one. The first produces the maximally parallel, device-neutral form: wherever the choice between
parallel and sequential is open, canonicalization takes parallel, because that form is the one a
GPU, a vectorizer and a CPU can all still specialize from. Only here, with the target known and the
map shapes final, is parallelism given back -- for a scope whose work cannot pay for a
``#pragma omp parallel``, or one that would fork a team per outer iteration.

The stage runs at the very end, AFTER cleanup and parallelization, for a reason the passes below
depend on: the verdict is read off the FINAL map shapes, and fusion and collapse change the work
per region by orders of magnitude. A schedule set earlier also blocks the map fusion stages, which
only fuse maps of equal schedule.

Order within the stage, and why:

1. The fork/join thresholds are read from the config as they stand: the schema defaults unless pinned.
   :class:`~dace.transformation.passes.cpu_specialization.calibrate_thresholds.CalibrateCpuThresholds`
   derives them from the host when called explicitly; running it here would change the process-wide
   config for every SDFG compiled after the first.
2. :class:`~dace.transformation.passes.cpu_specialization.chunk_anti_dependence.ChunkAntiDependence`
   -- trades the device-neutral anti-dependence snapshot for per-chunk seam buffers;
   sequential-within-chunk is a CPU scheduling decision (the matcher refuses GPU maps). Before the
   cost models so they read the chunked shape: the prologue and the seam-iteration map it emits are
   a 1-iteration and an ``nchunks``-iteration region, and sequentializing those is exactly the
   verdict the fork/join model exists to give.
3. :class:`~dace.transformation.passes.cpu_specialization.recompute_oversized_intermediates.RecomputeIntermediates`
   -- recomputes a producer map inside every consumer map and drops the intermediate, wherever legal.
5. :class:`~dace.transformation.passes.cpu_specialization.sequentialize_reduction_axes.SequentializeReductionAxes`
   -- a parallel map that reduces over some of its axes shares out the others and walks the reduced
   ones in an inner sequential map, so its writes need no atomic. Before the cost model, which then
   rules on the split shape.
6. :class:`~dace.transformation.passes.cpu_specialization.sequentialize_small_axes.SequentializeSmallAxes`
   -- the small facts decide: nested parallel scopes, small maps and data segments under a long loop run
   sequentially, and a map's leading ``SMALL`` axes are shared out with the first larger one. After the reduction
   split, which it then rules on.
7. :class:`~dace.transformation.passes.cpu_specialization.specialize_cpu_transfers.SpecializeCpuTransfers`
   -- gives the transfers step 6 just sequentialized their single ``memcpy`` / ``memset`` back, so
   the parallel-by-default copy expansion costs nothing when the cost model refuses it.
8. :class:`~dace.transformation.passes.cpu_specialization.range_privatize_scatter_reductions.RangePrivatizeScatterReductions`
   -- after the schedules are final: a parallel map scattering into an array through data-dependent
   indices gives each thread a private copy of only the rows its chunk writes, not of the whole array.
9. :class:`~dace.transformation.passes.specialize_small_gemms.SpecializeSmallGemms` -- a small GEMM a map or
   loop repeats runs as the row-wise loop nest in the iteration's thread, not as a BLAS call per iteration.
10. :class:`~dace.transformation.passes.cpu_specialization.choose_omp_schedule.ChooseOmpSchedule` -- once the parallel
    maps are final: one whose iterations do data-dependent work (a loop over a CSR row) is scheduled ``guided``.

The two terminal hygiene passes run again at the end because this stage builds states and indices
of its own: ``ChunkAntiDependence`` emits prologue and seam blocks, which orphan the entry guard
``AssumeSymbolConstraints`` prepended during canonicalization, and any pass may leave an index
holding python's ``//`` on a sympy expression -- sympy ``floor()``, which prints WITHOUT the floor
and truncates term by term. Both are re-runnable: the guard dedups and re-anchors itself.
"""

from dace import SDFG, dtypes
from dace.sdfg import infer_types
from dace.transformation.interstate.move_map_invariant_if_up import hoist_invariant_guards
from dace.transformation.passes.canonicalize.assume_symbols_nonnegative import AssumeSymbolConstraints
from dace.transformation.passes.canonicalize.normalize_floor_division import NormalizeFloorDivision
from dace.transformation.passes.cpu_specialization.choose_omp_schedule import ChooseOmpSchedule
from dace.transformation.passes.cpu_specialization.chunk_anti_dependence import ChunkAntiDependence
from dace.transformation.passes.cpu_specialization.if_convert_map_bodies import IfConvertParallelMapBodies
from dace.transformation.passes.cpu_specialization.range_privatize_scatter_reductions import (
    RangePrivatizeScatterReductions,
)
from dace.transformation.passes.cpu_specialization.recompute_oversized_intermediates import (
    RecomputeIntermediates,
)
from dace.transformation.passes.cpu_specialization.sequentialize_reduction_axes import SequentializeReductionAxes
from dace.transformation.passes.cpu_specialization.sequentialize_small_axes import SequentializeSmallAxes
from dace.transformation.passes.cpu_specialization.specialize_cpu_transfers import SpecializeCpuTransfers
from dace.transformation.passes.specialize_small_gemms import SpecializeSmallGemms


def cpu_specialize(sdfg: SDFG, validate: bool = False) -> SDFG:
    """Specialize a canonicalized ``sdfg`` for the CPU, in place.

    Run this AFTER :func:`~dace.transformation.passes.canonicalize.pipeline.canonicalize`, which
    leaves every open parallel/sequential choice on parallel. Idempotent: a second call re-confirms
    the same verdicts.

    :param sdfg: A canonicalized SDFG.
    :param validate: Validate the SDFG once at the end.
    :returns: The same ``sdfg`` instance, specialized.
    """
    # Canonicalization leaves schedules unset (``Default``), which is not a decision but the absence
    # of one: codegen reads a top-level ``Default`` as ``CPU_Multicore`` and a nested one as
    # ``Sequential``. Resolve them first so the cost model below rules on the schedules that will
    # actually be emitted, and so this stage's output states its verdicts outright instead of
    # leaving half of them implicit in a codegen rule.
    infer_types.set_default_schedule_and_storage_types(sdfg, None)

    hoist_invariant_guards(sdfg, require_full_hoist=False)
    ChunkAntiDependence().apply_pass(sdfg, {})
    RecomputeIntermediates().apply_pass(sdfg, {})
    SequentializeReductionAxes().apply_pass(sdfg, {})
    SequentializeSmallAxes().apply_pass(sdfg, {})
    # After the schedules are final: only a body that stays a parallel map is if-converted.
    IfConvertParallelMapBodies().apply_pass(sdfg, {})
    RangePrivatizeScatterReductions().apply_pass(sdfg, {})
    SpecializeCpuTransfers().apply_pass(sdfg, {})
    SpecializeSmallGemms(dtypes.DeviceType.CPU).apply_pass(sdfg, {})
    ChooseOmpSchedule().apply_pass(sdfg, {})

    AssumeSymbolConstraints().apply_pass(sdfg, {})
    NormalizeFloorDivision().apply_pass(sdfg, {})

    if validate:
        sdfg.validate()
    return sdfg
