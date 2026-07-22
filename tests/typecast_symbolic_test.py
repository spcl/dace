# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""First-grade numeric typecast functions (every DaCe scalar width --
``int8``..``int64`` / ``uint8``..``uint64`` / ``float16``..``float64`` /
``complex*``, built from ``dtypes.TYPECLASS_TO_STRING``) in symbolic expressions.

A Fortran kind coercion lands as ``int32(x)`` / ``float64(x)`` in BOTH
tasklet bodies and symbolic expressions (interstate edges, memlet
subsets, conditions).  These are registered ``sympy.Function``s so they
parse cleanly, carry integer/real-ness for downstream symbolic
reasoning, and print to the matching ``dace::<type>(x)`` C++ cast
(truncating for int) -- so the SAME bare spelling round-trips through the
sympy printer and cppunparse to identical code.
"""
import pytest
import sympy

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


# The cast set is EVERY DaCe scalar width, not a hardcoded {int32,int64,float32,float64}. A width outside
# that old subset -- e.g. ``dace.uint16`` (a CRC step ``dace.uint16(byte & 1)``) -- used to fall through
# ``visit_Call`` to ``Attr(dace, uint16)(x)`` -> ``'Attr' object is not callable`` at parse. The map is now
# built from ``dtypes.TYPECLASS_TO_STRING`` so it tracks the dtype list.
@pytest.mark.parametrize("name", ["int8", "int16", "uint8", "uint16", "uint32", "uint64", "float16"])
def test_all_width_casts_parse_and_print(name):
    # both the bare and the ``dace.``-prefixed spelling reach ``dace::<name>(x)`` -- neither raises.
    assert sym2cpp(pystr_to_symbolic(f"{name}(x)")) == f"(dace::{name}(x))"
    assert sym2cpp(pystr_to_symbolic(f"dace.{name}(x)")) == f"(dace::{name}(x))"


@pytest.mark.parametrize("expr,expect", [
    ("math.sin(y)", "sin(y)"),
    ("numpy.sqrt(x)", "sqrt(x)"),
    ("np.exp(x) + 1", "exp(x) + 1"),
    ("dace.math.sin(y)", "sin(y)"),
])
def test_math_module_qualifier_is_stripped(expr, expect):
    # A library-function module qualifier (math/numpy/np/dace.math) is noise in a symbolic expression: it is
    # stripped to the bare sympy function, not folded to ``Attr(math, sin)(y)`` -> 'Attr' object is not
    # callable. Same visit_Call gap the dace-cast branch closes, for the function modules.
    assert str(pystr_to_symbolic(expr)).replace(" ", "") == expect.replace(" ", "")


def test_all_int_width_casts_are_integer():
    # every int/uint width carries the integer assumption so it can index a subset / drive a Min without
    # mixing kinds (the azimint_hist Min(int64_index, bins-1) mixed-type class of bug).
    for name in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"):
        assert pystr_to_symbolic(f"{name}(x)").is_integer is True
    for name in ("float16", "float32", "float64"):
        assert pystr_to_symbolic(f"{name}(x)").is_real is True
