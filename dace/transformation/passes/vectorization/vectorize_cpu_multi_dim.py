# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Stable entry point for the CPU K-dim tile-op vectorizer.

Re-exports :class:`VectorizeMultiDim`, its ``device=CPU`` wrapper
:class:`VectorizeCPUMultiDim`, and the helpers the corpus harness/tests import.
"""

from dace.transformation.passes.vectorization.vectorize_multi_dim import (
    TILE_NODE_TYPES,
    VectorizeCPUMultiDim,
    VectorizeGPUMultiDim,
    VectorizeMultiDim,
    _validate_knobs,
    normalize_loop_nests,
)

__all__ = [
    "VectorizeMultiDim",
    "VectorizeCPUMultiDim",
    "VectorizeGPUMultiDim",
    "normalize_loop_nests",
    "_validate_knobs",
    "TILE_NODE_TYPES",
]
