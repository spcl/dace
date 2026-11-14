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
4. :class:`~dace.transformation.passes.cpu_specialization.decide_cpu_parallel_axes.DecideCpuParallelAxes`
   -- the one decision on parallel axes, top-down, while the schedules are still unset: a one- or few-trip map
   hands the team to its body maps, a reducing map shares out the axes its outputs are indexed by and walks the
   reduced ones inside, small maps and data segments under a long loop run sequentially, and a map's leading
   ``SMALL`` axes are shared out with the first larger one. Schedule inference then resolves what it left unset.
   A copy or fill it sequentializes picks its single ``memcpy`` / ``memset`` itself, at implementation selection.
5. :class:`~dace.transformation.passes.cpu_specialization.bound_masked_maps.BoundMaskedMaps` -- on the final shapes:
   a map over padded slots stops at the count that masks them, and ``ChooseOmpSchedule`` then sees the uneven work.
6. :class:`~dace.transformation.passes.privatize_scatter_reduction.PrivatizeScatterReduction` then
   :class:`~dace.transformation.passes.cpu_specialization.range_privatize_scatter_reductions.RangePrivatizeScatterReductions`
   -- after the schedules are final: a parallel map scattering into an array through data-dependent
   indices reduces into a private copy (an OpenMP array-section ``reduction``), of only the rows its chunk writes
   where that is smaller than the whole array.
7. :class:`~dace.transformation.passes.specialize_small_gemms.SpecializeSmallGemms` -- a small GEMM a map or
   loop repeats runs as the row-wise loop nest in the iteration's thread, not as a BLAS call per iteration.
8. :class:`~dace.transformation.passes.cpu_specialization.choose_omp_schedule.ChooseOmpSchedule` -- once the parallel
   maps are final: one whose iterations do data-dependent work (a loop over a CSR row) is scheduled ``guided``.
"""

from dace import SDFG, dtypes
from dace.sdfg import infer_types
from dace.transformation.interstate.move_map_invariant_if_up import hoist_invariant_guards
from dace.transformation.passes.cpu_specialization.bound_masked_maps import BoundMaskedMaps
from dace.transformation.passes.cpu_specialization.choose_omp_schedule import ChooseOmpSchedule
from dace.transformation.passes.cpu_specialization.chunk_anti_dependence import ChunkAntiDependence
from dace.transformation.passes.cpu_specialization.decide_cpu_parallel_axes import DecideCpuParallelAxes
from dace.transformation.passes.cpu_specialization.if_convert_map_bodies import IfConvertParallelMapBodies
from dace.transformation.passes.cpu_specialization.range_privatize_scatter_reductions import (
    RangePrivatizeScatterReductions,
)
from dace.transformation.passes.cpu_specialization.recompute_oversized_intermediates import (
    RecomputeIntermediates,
)
from dace.transformation.passes.cpu_specialization.split_invariant_map_nests import SplitInvariantMapNests
from dace.transformation.passes.privatize_scatter_reduction import PrivatizeScatterReduction
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
    hoist_invariant_guards(sdfg, require_full_hoist=False)
    # The canonical form keeps map nests whole; here an outer map with enough trips feeds the team alone.
    SplitInvariantMapNests().apply_pass(sdfg, {})
    ChunkAntiDependence().apply_pass(sdfg, {})
    RecomputeIntermediates().apply_pass(sdfg, {})
    # Canonicalization leaves schedules unset (``Default``), which is not a decision but the absence
    # of one. The CPU maps' schedules are decided first, before inference would pin every nested map
    # ``Sequential``; inference then resolves the rest (library nodes, storage), so this stage's
    # output states its verdicts outright instead of leaving half of them implicit in a codegen rule.
    DecideCpuParallelAxes().apply_pass(sdfg, {})
    infer_types.set_default_schedule_and_storage_types(sdfg, None)
    # After the schedules are final: only a body that stays a parallel map is if-converted.
    IfConvertParallelMapBodies().apply_pass(sdfg, {})
    BoundMaskedMaps().apply_pass(sdfg, {})
    PrivatizeScatterReduction().apply_pass(sdfg, {})
    RangePrivatizeScatterReductions().apply_pass(sdfg, {})
    SpecializeSmallGemms(dtypes.DeviceType.CPU).apply_pass(sdfg, {})
    ChooseOmpSchedule().apply_pass(sdfg, {})

    if validate:
        sdfg.validate()
    return sdfg
