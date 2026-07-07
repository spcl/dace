# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``VectorizeGPU`` — the NVIDIA CUDA half2 (FP16x2) vectorization entry.

The GPU path is the K=1 multi-dim tile pipeline (:class:`VectorizeCPUMultiDim`)
with three fixed GPU choices, so it is a thin subclass rather than a parallel
implementation:

* ``target_isa = "CUDA"`` — the innermost tile op lowers to ``dace/tile_ops/cuda.h``,
  whose fp16 elementwise binops/unop use the native ``__hadd2`` / ``__hmul2`` / ...
  half2 intrinsics (2 lanes per instruction). fp8 computes through ``float``.
* ``widths = (2,)`` — a half2 packs exactly two fp16 lanes, so the innermost tile
  is always width 2 (``_validate_knobs`` already requires ``widths[-1] % 2 == 0``
  for CUDA).
* ``assume_even = True`` — a GPU kernel emits NO remainder loop: the map extent is
  assumed an exact multiple of 2, so the map is a single ``0:N:2`` strided map with
  no masked tail (which would otherwise split into two GPU_Device maps of different
  thread-block sizes). The caller guarantees the even extent (optionally enforced by
  a runtime check).

The tile ops must run inside a GPU kernel for the ``__device__`` half2 intrinsics to
apply, so the target map must be ``GPU_Device``-scheduled. If the input SDFG has no
GPU-scheduled map yet, :meth:`apply_pass` runs ``apply_gpu_transformations`` first
(the caller's ``simplify -> LoopToMap -> map-fusion`` has already produced the maps).
"""
from typing import Optional, Tuple

import dace
from dace.dtypes import ScheduleType
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim


def _has_gpu_device_map(sdfg: dace.SDFG) -> bool:
    """True iff any map in the SDFG (recursively) is ``GPU_Device``-scheduled."""
    return any(
        isinstance(n, dace.nodes.MapEntry) and n.map.schedule == ScheduleType.GPU_Device
        for n, _ in sdfg.all_nodes_recursive())


class VectorizeGPU(VectorizeCPUMultiDim):
    """Drive the CUDA half2 (FP16x2) tile pipeline.

    Fixes the GPU knob row (``target_isa='CUDA'``, ``widths=(2,)``, ``assume_even=True``)
    on top of the shared multi-dim tile pipeline and GPU-schedules the SDFG first when
    the caller has not. Every other knob (branch mode, expand, ...) is forwarded to
    :class:`VectorizeCPUMultiDim` unchanged.
    """

    def __init__(self, widths: Tuple[int, ...] = (2, ), gpu_schedule_if_needed: bool = True, **kwargs):
        """Build the GPU orchestrator.

        :param widths: Per-dim tile widths, innermost-last; the innermost must be even
            (a half2 is 2 fp16 lanes). Defaults to ``(2,)``.
        :param gpu_schedule_if_needed: When ``True`` (default), run
            ``apply_gpu_transformations`` in :meth:`apply_pass` if the SDFG has no
            ``GPU_Device`` map yet. Set ``False`` when the caller has already scheduled
            (or deliberately keeps a custom GPU schedule).
        :param kwargs: Forwarded to :class:`VectorizeCPUMultiDim` (``branch_mode``,
            ``expand_tile_nodes``, ``validate_all``, ...). ``target_isa`` and
            ``assume_even`` are pinned to the GPU values and must not be overridden.
            ``expand_tile_nodes`` defaults to ``False`` here: the pipeline returns
            with the ``cuda``-stamped tile lib nodes in place (the vectorized form),
            and the caller lowers them with ``sdfg.expand_library_nodes()`` (or just
            ``sdfg.compile()``) when ready.
        """
        kwargs.setdefault("target_isa", "CUDA")
        kwargs.setdefault("assume_even", True)
        kwargs.setdefault("expand_tile_nodes", False)
        if kwargs["target_isa"] != "CUDA":
            raise NotImplementedError(f"VectorizeGPU only targets CUDA; got target_isa={kwargs['target_isa']!r}")
        if not kwargs["assume_even"]:
            raise NotImplementedError("VectorizeGPU emits no remainder loop; assume_even must stay True")
        super().__init__(widths=widths, **kwargs)
        self._gpu_schedule_if_needed = gpu_schedule_if_needed

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results) -> Optional[int]:
        """GPU-schedule the SDFG (if needed), then run the half2 tile pipeline.

        :param sdfg: SDFG to transform in place.
        :param pipeline_results: Carry-in from any enclosing pipeline.
        :returns: Whatever the tile pipeline returned.
        """
        if self._gpu_schedule_if_needed and not _has_gpu_device_map(sdfg):
            sdfg.apply_gpu_transformations()
            sdfg.simplify()
        return super().apply_pass(sdfg, pipeline_results)
