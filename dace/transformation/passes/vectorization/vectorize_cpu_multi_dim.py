# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Stable entry point for the CPU K-dim tile-op vectorizer.

Re-exports :class:`VectorizeMultiDim`, its ``device=CPU`` wrapper
:class:`VectorizeCPUMultiDim`, and the helpers the corpus harness/tests import.
"""
from dace.transformation.passes.vectorization.vectorize_multi_dim import (
    VectorizeMultiDim,
    VectorizeCPUMultiDim,
    VectorizeGPUMultiDim,
    normalize_loop_nests,
    _validate_knobs,
    _VALID_ISAS,
    _TILE_NODE_TYPES,
)

__all__ = [
    "VectorizeMultiDim",
    "VectorizeCPUMultiDim",
    "VectorizeGPUMultiDim",
    "normalize_loop_nests",
    "_validate_knobs",
    "_VALID_ISAS",
    "_TILE_NODE_TYPES",
]
