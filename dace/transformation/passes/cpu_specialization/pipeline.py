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

1. :class:`~dace.transformation.passes.cpu_specialization.calibrate_thresholds.CalibrateCpuThresholds`
   -- first, because both cost-model passes read fork/join thresholds out of the config and those
   shipped as constants measured on one box. Calibrating to the host's own core count before anyone
   reads them stops a 72-core machine inheriting an 8-core machine's break-even. A user-set value
   is left alone.
2. :class:`~dace.transformation.passes.cpu_specialization.chunk_anti_dependence.ChunkAntiDependence`
   -- trades the device-neutral anti-dependence snapshot for per-chunk seam buffers;
   sequential-within-chunk is a CPU scheduling decision (the matcher refuses GPU maps). Before the
   cost models so they read the chunked shape: the prologue and the seam-iteration map it emits are
   a 1-iteration and an ``nchunks``-iteration region, and sequentializing those is exactly the
   verdict the fork/join model exists to give.
3. :class:`~dace.transformation.passes.cpu_specialization.recompute_oversized_intermediates.RecomputeOversizedIntermediates`
   -- collapses a producer map into its single consumer and drops the intermediate, but only when
   that intermediate provably does not fit in the host's last-level cache. The CPU keeps a
   cache-resident intermediate materialized (reading it back beats recomputing it) and pays ALU only
   for the ones that would cost a full DRAM round-trip.
4. :class:`~dace.transformation.passes.cpu_specialization.sequentialize_unprofitable_parallel_scopes.SequentializeUnprofitableParallelScopes`
   -- the fork/join cost model itself.
5. :class:`~dace.transformation.passes.cpu_specialization.specialize_cpu_transfers.SpecializeCpuTransfers`
   -- gives the transfers step 4 just sequentialized their single ``memcpy`` / ``memset`` back, so
   the parallel-by-default copy expansion costs nothing when the cost model refuses it.

The two terminal hygiene passes run again at the end because this stage builds states and indices
of its own: ``ChunkAntiDependence`` emits prologue and seam blocks, which orphan the entry guard
``AssumeSymbolConstraints`` prepended during canonicalization, and any pass may leave an index
holding python's ``//`` on a sympy expression -- sympy ``floor()``, which prints WITHOUT the floor
and truncates term by term. Both are re-runnable: the guard dedups and re-anchors itself.
"""

from dace import SDFG
from dace.sdfg import infer_types
from dace.transformation.passes.canonicalize.assume_symbols_nonnegative import AssumeSymbolConstraints
from dace.transformation.passes.canonicalize.normalize_floor_division import NormalizeFloorDivision
from dace.transformation.passes.cpu_specialization.calibrate_thresholds import CalibrateCpuThresholds
from dace.transformation.passes.cpu_specialization.chunk_anti_dependence import ChunkAntiDependence
from dace.transformation.passes.cpu_specialization.recompute_oversized_intermediates import (
    RecomputeOversizedIntermediates)
from dace.transformation.passes.cpu_specialization.sequentialize_unprofitable_parallel_scopes import (
    SequentializeUnprofitableParallelScopes)
from dace.transformation.passes.cpu_specialization.specialize_cpu_transfers import SpecializeCpuTransfers


def cpu_specialize(sdfg: SDFG, break_anti_dependence: bool = True, validate: bool = True) -> SDFG:
    """Specialize a canonicalized ``sdfg`` for the CPU, in place.

    Run this AFTER :func:`~dace.transformation.passes.canonicalize.pipeline.canonicalize`, which
    leaves every open parallel/sequential choice on parallel. Idempotent: a second call re-confirms
    the same verdicts.

    :param sdfg: A canonicalized SDFG.
    :param break_anti_dependence: Chunk the anti-dependence snapshots into per-chunk seam buffers.
                                  Pass ``False`` when canonicalization did not break anti
                                  dependences, so there are no snapshots to chunk.
    :param validate: Validate the SDFG once at the end.
    :returns: The same ``sdfg`` instance, specialized.
    """
    # Canonicalization leaves schedules unset (``Default``), which is not a decision but the absence
    # of one: codegen reads a top-level ``Default`` as ``CPU_Multicore`` and a nested one as
    # ``Sequential``. Resolve them first so the cost model below rules on the schedules that will
    # actually be emitted, and so this stage's output states its verdicts outright instead of
    # leaving half of them implicit in a codegen rule.
    infer_types.set_default_schedule_and_storage_types(sdfg, None)

    CalibrateCpuThresholds().apply_pass(sdfg, {})
    if break_anti_dependence:
        ChunkAntiDependence().apply_pass(sdfg, {})
    RecomputeOversizedIntermediates().apply_pass(sdfg, {})
    SequentializeUnprofitableParallelScopes().apply_pass(sdfg, {})
    SpecializeCpuTransfers().apply_pass(sdfg, {})

    AssumeSymbolConstraints().apply_pass(sdfg, {})
    NormalizeFloorDivision().apply_pass(sdfg, {})

    if validate:
        sdfg.validate()
    return sdfg
