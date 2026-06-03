"""Public vectorization passes: CPU and GPU map/tasklet vectorizers."""
from .vectorize import Vectorize
from .vectorize_cpu import VectorizeCPU
# Importing this module registers the ``"vectorized"`` implementation on
# the standard ``Reduce`` library node (schedule-aware dispatcher).
from . import reduce_expansion  # noqa: F401
