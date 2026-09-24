# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Public vectorization passes: the multi-dim CPU/GPU tile-op vectorizer.

Pipeline entry points (``VectorizeMultiDim`` / ``VectorizeCPUMultiDim`` /
``VectorizeGPUMultiDim``) are exported LAZILY via :pep:`562` ``__getattr__``:
eager import would close an ``interstate -> canonicalize -> vectorization``
cycle, since ``vectorize_multi_dim`` imports ``interstate`` at module load.
"""
# Registers the "vectorized" impl on the standard Reduce library node. Cycle-safe.
from dace.transformation.passes.vectorization import reduce_expansion  # noqa: F401

_PIPELINE_EXPORTS = frozenset({"VectorizeMultiDim", "VectorizeCPUMultiDim", "VectorizeGPUMultiDim"})


def __getattr__(name: str) -> object:
    """Lazily resolve the pipeline entry points (breaks the interstate import cycle)."""
    if name in _PIPELINE_EXPORTS:
        from dace.transformation.passes.vectorization import vectorize_multi_dim
        if name == "VectorizeMultiDim":
            return vectorize_multi_dim.VectorizeMultiDim
        if name == "VectorizeCPUMultiDim":
            return vectorize_multi_dim.VectorizeCPUMultiDim
        return vectorize_multi_dim.VectorizeGPUMultiDim
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals()) + list(_PIPELINE_EXPORTS))
