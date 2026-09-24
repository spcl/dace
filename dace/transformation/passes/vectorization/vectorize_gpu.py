# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Stable entry point for the CUDA half2 (FP16x2) vectorizer.

:class:`VectorizeGPUMultiDim` is a thin ``device=GPU`` wrapper on
:class:`VectorizeMultiDim` (fixed ``target_isa='CUDA'``, ``widths=(2,)``,
``assume_even=True``). It vectorizes the resident ``GPU_Device`` maps of an
already-offloaded SDFG; it never offloads/schedules the SDFG itself.
"""
from dace.transformation.passes.vectorization.vectorize_multi_dim import VectorizeGPUMultiDim, _has_gpu_device_map


class VectorizeGPU(VectorizeGPUMultiDim):
    """Alias of :class:`VectorizeGPUMultiDim` -- the CUDA half2 (FP16x2) tile pipeline."""


__all__ = ["VectorizeGPU", "VectorizeGPUMultiDim", "_has_gpu_device_map"]
