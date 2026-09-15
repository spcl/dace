# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Backwards-compatible entry point for the CUDA half2 (FP16x2) vectorizer.

GPU orchestrator = :class:`VectorizeGPUMultiDim`: thin ``device=GPU`` wrapper on
:class:`VectorizeMultiDim` (see
:mod:`dace.transformation.passes.vectorization.vectorize_multi_dim`). Fixes GPU knob row
(``target_isa='CUDA'``, ``widths=(2,)``, ``assume_even=True``). It assumes an ALREADY
GPU-offloaded SDFG and vectorizes the resident ``GPU_Device`` maps; it never offloads /
schedules the SDFG itself (the canonicalize-GPU pipeline offloads first).
:class:`VectorizeGPU` = alias for the stable import path.
"""
from dace.transformation.passes.vectorization.vectorize_multi_dim import VectorizeGPUMultiDim, _has_gpu_device_map


class VectorizeGPU(VectorizeGPUMultiDim):
    """Alias of :class:`VectorizeGPUMultiDim` — the CUDA half2 (FP16x2) tile pipeline."""


__all__ = ["VectorizeGPU", "VectorizeGPUMultiDim", "_has_gpu_device_map"]
