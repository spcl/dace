"""Data-class shapes shared by every intrinsic sub-registry.

Each registry file (``elementwise.py``, ``reductions/...``, ``linalg/...``,
``direct/...``) just populates a dict whose values are one of these
dataclasses, so the public helpers in ``__init__.py`` can be family-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ElementwiseIntrinsic:
    """A Fortran intrinsic whose lowered form is a per-element scalar call
    inside an ``hlfir.elemental`` body.  The name is used verbatim in the
    DaCe tasklet code string; DaCe's codegen resolves it through
    ``_ALLOWED_MODULES`` in ``dace/dtypes.py`` to the runtime wrappers in
    ``dace/runtime/include/dace/math.h``."""

    name: str
    arity: int


@dataclass(frozen=True)
class ReductionIntrinsic:
    """Whole-array reduction that becomes a ``standard.Reduce`` library
    node via ``state.add_reduce(wcr, axes, identity)``.  Not used yet  --
    Phase 2."""

    name: str
    wcr: str
    identity: str


@dataclass(frozen=True)
class LibNodeIntrinsic:
    """Intrinsic that becomes a direct DaCe library-node emission
    (``blas.Matmul``, ``standard.Transpose``, ``blas.Dot``, ``fft.FFT``).
    Not used yet  --  Phase 3."""

    name: str
    module: str  # e.g. "blas", "standard", "fft"
    node_cls: str  # e.g. "Matmul", "Transpose", "Dot"
