"""Elementwise Fortran intrinsics.

Every entry lowers to a bare-name scalar call inside a Python tasklet
body (e.g. ``_out = sin(_in_a)``).  DaCe's codegen maps the bare name
through ``_ALLOWED_MODULES`` (``dace/dtypes.py``) to ``dace::math::...``
from ``dace/runtime/include/dace/math.h``, so we do not prefix with
``math.`` or switch the tasklet language.

When Flang lowers ``sin(a)`` on an array, the result is an
``hlfir.elemental`` whose body ends in ``math.sin``.  The bridge's
``buildExpr`` emits the bare ``sin(...)`` form; the SDFG emitter keeps
it bare by consulting ``is_elementwise`` so the name isn't rewritten to
an ``_in_sin`` tasklet connector.
"""
from __future__ import annotations

from dace.frontend.hlfir.intrinsics.base import ElementwiseIntrinsic


def _one(name: str, arity: int = 1) -> tuple[str, ElementwiseIntrinsic]:
    return name, ElementwiseIntrinsic(name=name, arity=arity)


ELEMENTWISE_INTRINSICS: dict[str, ElementwiseIntrinsic] = dict([
    # Transcendentals
    _one('sin'),
    _one('cos'),
    _one('tan'),
    _one('asin'),
    _one('acos'),
    _one('atan'),
    _one('sinh'),
    _one('cosh'),
    _one('tanh'),
    _one('exp'),
    _one('log'),
    _one('log10'),
    _one('sqrt'),
    # Rounding / sign
    _one('abs'),
    _one('floor'),
    _one('ceil'),
    # Special functions
    _one('erf'),
    _one('erfc'),
    # Two-arg
    _one('min', arity=2),
    _one('max', arity=2),
    _one('pow', arity=2),
    _one('atan2', arity=2),
])
