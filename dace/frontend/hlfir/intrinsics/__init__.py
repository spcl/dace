"""Fortran intrinsic -> DaCe lowering registry for the HLFIR frontend.

Public surface:
    is_elementwise(name)
    is_reduction(name)
    is_libnode(name)
    is_intrinsic(name)
    render_call(name, args)
    reduction_spec(name)   # -> ReductionIntrinsic | None
    libnode_spec(name)     # -> LibNodeIntrinsic | None

Emitter code should only talk to this module, never import the per-family
registries directly.  Adding a new intrinsic means editing one file under
this package and nothing else.  The families live in flat siblings:

    elementwise.py     --  sin, cos, exp, sqrt, abs, min, max, ...
    reduction.py       --  sum, product, minval, maxval
    linalg.py          --  matmul, transpose, dot_product
    direct.py          --  SIZE / LBOUND / UBOUND / ... (Phase 4 stub)
"""
from __future__ import annotations

from dace.frontend.hlfir.intrinsics.elementwise import ELEMENTWISE_INTRINSICS
from dace.frontend.hlfir.intrinsics.reduction import REDUCTIONS
from dace.frontend.hlfir.intrinsics.linalg import LINALG, STANDARD
from dace.frontend.hlfir.intrinsics.direct import DIRECT_INTRINSICS


def is_elementwise(name: str) -> bool:
    return name in ELEMENTWISE_INTRINSICS


def is_reduction(name: str) -> bool:
    return name in REDUCTIONS


def is_libnode(name: str) -> bool:
    return name in LINALG or name in STANDARD


def is_intrinsic(name: str) -> bool:
    """True if ``name`` is a known Fortran intrinsic in any family."""
    return (is_elementwise(name) or is_reduction(name) or is_libnode(name) or name in DIRECT_INTRINSICS)


def render_call(name: str, args: list[str]) -> str:
    """Return ``name(arg0, arg1, ...)`` verbatim.

    Only validates elementwise arity today; reduction / libnode callers
    will gain their own render helpers as those phases come online.
    """
    spec = ELEMENTWISE_INTRINSICS.get(name)
    if spec is not None:
        assert len(args) == spec.arity, (f"{name} expects {spec.arity} arg(s), got {len(args)}")
    return f"{name}({', '.join(args)})"


def reduction_spec(name: str):
    """Return the ``ReductionIntrinsic`` for ``name`` or ``None``."""
    return REDUCTIONS.get(name)


def libnode_spec(name: str):
    """Return the ``LibNodeIntrinsic`` for ``name`` or ``None``.  Looks up
    across both the linalg registry (matmul / transpose / dot_product) and
    the generic standard-library registry (count / merge / ...)."""
    spec = LINALG.get(name)
    if spec is not None:
        return spec
    return STANDARD.get(name)
