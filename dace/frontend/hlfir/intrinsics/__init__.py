"""Fortran intrinsic → DaCe lowering registry for the HLFIR frontend.

Public surface:
    is_elementwise(name)
    is_reduction(name)   # Phase 2 — today always False
    is_libnode(name)     # Phase 3 — today always False
    render_call(name, args)

Emitter code should only talk to this module, never import the per-family
registries directly.  Adding a new intrinsic means editing one file under
this package and nothing else.
"""
from __future__ import annotations

from dace.frontend.hlfir.intrinsics.elementwise import ELEMENTWISE_INTRINSICS

# Phase 2 / 3 / 4 registries live in sub-packages and will be merged in as
# they come online.  For now they're empty dictionaries so callers can ask
# ``is_reduction('sum')`` today without an ImportError later.
from dace.frontend.hlfir.intrinsics.reductions import REDUCTION_INTRINSICS
from dace.frontend.hlfir.intrinsics.linalg import LIBNODE_INTRINSICS
from dace.frontend.hlfir.intrinsics.direct import DIRECT_INTRINSICS


def is_elementwise(name: str) -> bool:
    return name in ELEMENTWISE_INTRINSICS


def is_reduction(name: str) -> bool:
    return name in REDUCTION_INTRINSICS


def is_libnode(name: str) -> bool:
    return name in LIBNODE_INTRINSICS


def is_intrinsic(name: str) -> bool:
    """True if ``name`` is a known Fortran intrinsic in any family."""
    return (is_elementwise(name) or is_reduction(name) or is_libnode(name) or name in DIRECT_INTRINSICS)


def render_call(name: str, args: list[str]) -> str:
    """Return ``name(arg0, arg1, …)`` verbatim.

    Only validates elementwise arity today; reduction / libnode callers
    will gain their own render helpers as those phases come online.
    """
    spec = ELEMENTWISE_INTRINSICS.get(name)
    if spec is not None:
        assert len(args) == spec.arity, (f"{name} expects {spec.arity} arg(s), got {len(args)}")
    return f"{name}({', '.join(args)})"


def reduction_spec(name: str):
    """Return the ``ReductionIntrinsic`` for ``name`` or ``None``."""
    return REDUCTION_INTRINSICS.get(name)
