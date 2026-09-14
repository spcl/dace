# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""First-grade numeric typecast functions (``int32`` / ``int64`` /
``float32`` / ``float64``) in symbolic expressions.

A Fortran kind coercion lands as ``int32(x)`` / ``float64(x)`` in BOTH
tasklet bodies and symbolic expressions (interstate edges, memlet
subsets, conditions).  These are registered ``sympy.Function``s so they
parse cleanly, carry integer/real-ness for downstream symbolic
reasoning, and print to the matching ``dace::<type>(x)`` C++ cast
(truncating for int) -- so the SAME bare spelling round-trips through the
sympy printer and cppunparse to identical code.
"""
import pytest

import dace
from dace.symbolic import pystr_to_symbolic
from dace.codegen.targets.cpp import sym2cpp


@pytest.mark.parametrize("expr,cpp", [
    ("int32(qm) + 1", "(dace::int32(qm) + 1)"),
    ("int64(x)", "(dace::int64(x))"),
    ("float32(i) * 2", "(2*dace::float32(i))"),
    ("float64(i) - r", "(-r + dace::float64(i))"),
])
def test_typecast_prints_to_dace_cast(expr, cpp):
    assert sym2cpp(pystr_to_symbolic(expr)) == cpp


def test_typecast_roundtrips():
    """The bare spelling survives parse -> str (no ``dace.`` prefix, not
    folded to an attribute)."""
    for expr in ("int32(qm)", "float64(i)", "int64(a + b)"):
        s = pystr_to_symbolic(expr)
        assert str(s).replace(" ", "") == expr.replace(" ", "")


def test_int_typecast_is_integer():
    """``int32``/``int64`` report as integers so they can index a memlet
    subset / drive a loop bound."""
    qm = dace.symbol("qm")
    assert pystr_to_symbolic("int32(qm)").is_integer is True
    assert pystr_to_symbolic("int64(qm)").is_integer is True


def test_dace_prefixed_cast_is_accepted():
    """Safety net: a stray ``dace.int32(x)`` (the attribute-call spelling)
    still parses to the bare typecast rather than raising
    ``'Attr' object is not callable``."""
    assert sym2cpp(pystr_to_symbolic("dace.int32(qm) + 1")) == "(dace::int32(qm) + 1)"
