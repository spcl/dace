# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Public vectorization passes: the multi-dim CPU/GPU tile-op vectorizer.

The pipeline entry points (``VectorizeMultiDim`` / ``VectorizeCPUMultiDim`` /
``VectorizeGPUMultiDim``) are exported LAZILY via :pep:`562` ``__getattr__``.
``vectorize_multi_dim`` imports ``dace.transformation.interstate`` at module load,
and ``interstate`` in turn (through ``passes -> canonicalize``) imports back into
this package -- eager-importing the pipeline here would close that cycle. Deferring
it means importing a vectorization *submodule* does not drag in the whole pipeline,
while ``from ...vectorization import VectorizeCPUMultiDim`` still works on demand.
"""
# Importing this module registers the ``"vectorized"`` implementation on the
# standard ``Reduce`` library node (schedule-aware dispatcher). Cycle-safe.
from . import reduce_expansion  # noqa: F401

_PIPELINE_EXPORTS = frozenset({"VectorizeMultiDim", "VectorizeCPUMultiDim", "VectorizeGPUMultiDim"})


def __getattr__(name):
    """Lazily resolve the pipeline entry points (breaks the interstate import cycle)."""
    if name in _PIPELINE_EXPORTS:
        from dace.transformation.passes.vectorization import vectorize_multi_dim
        if name == "VectorizeMultiDim":
            return vectorize_multi_dim.VectorizeMultiDim
        if name == "VectorizeCPUMultiDim":
            return vectorize_multi_dim.VectorizeCPUMultiDim
        return vectorize_multi_dim.VectorizeGPUMultiDim
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_PIPELINE_EXPORTS))
