# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the ``ipow`` symbolic Function and the ``RelaxIntegerPowers`` pass."""
import numpy as np
import sympy

import dace
from dace import symbolic
from dace.symbolic import ipow, pystr_to_symbolic, symstr
from dace.transformation.passes.relax_integer_powers import (RelaxIntegerPowers, _proven_nonnegative, _relaxed_exponent)


# --------------------------------------------------------------------------- #
# ipow symbolic Function
# --------------------------------------------------------------------------- #
def test_ipow_lowers_to_cpp_ipow():
    R, K = (symbolic.symbol(s, positive=True, integer=True) for s in ('R', 'K'))
    assert 'dace::math::ipow(R, K)' in symstr(ipow(R, K), cpp_mode=True)


def test_ipow_roundtrips_through_serialization():
    R, K, i = (symbolic.symbol(s) for s in ('R', 'K', 'i'))
    e = R * ipow(R, K - i - 1) + ipow(R, K - i - 1)
    back = pystr_to_symbolic(str(e))
    # Structural (assumption-free) identity: the C++ rendering must match.
    assert symstr(e, cpp_mode=True) == symstr(back, cpp_mode=True)
    assert any(type(a).__name__ == 'ipow' for a in sympy.preorder_traversal(back))


def test_ipow_is_integer_and_positive():
    R, K = (symbolic.symbol(s, positive=True, integer=True) for s in ('R', 'K'))
    assert ipow(R, K).is_integer is True
    assert ipow(R, K).is_positive is True


def test_ipow_folds_constant_power():
    assert ipow(sympy.Integer(2), sympy.Integer(10)) == 1024


# --------------------------------------------------------------------------- #
# Interval analysis (the soundness crux)
# --------------------------------------------------------------------------- #
def test_interval_proves_radix_decomposition():
    K = symbolic.symbol('K', positive=True, integer=True)
    i = symbolic.symbol('i', integer=True)
    ranges = {'i': (0, K - 1)}  # loop ``for i in range(K)``
    # K - i - 1 bottoms out at 0 when i = K-1; i bottoms out at 0.
    assert _proven_nonnegative(K - i - 1, ranges) is True
    assert _proven_nonnegative(i, ranges) is True
    assert _proven_nonnegative(K, ranges) is True  # positive symbol, no iterator


def test_interval_refuses_unbounded_iterator():
    K = symbolic.symbol('K', positive=True, integer=True)
    i = symbolic.symbol('i', integer=True)
    # Without a range for i, K - i - 1 could be negative -> must not relax.
    assert _proven_nonnegative(K - i - 1, {}) is False


def test_interval_refuses_provably_negative():
    i = symbolic.symbol('i', positive=True, integer=True)
    assert _proven_nonnegative(-i, {}) is False


def test_interval_canonicalizes_mismatched_assumptions():
    # The exponent's K is positive but the range bound's K (as if parsed from a
    # loop condition) is assumption-less; they must still cancel to 0.
    K_pos = symbolic.symbol('K', positive=True, integer=True)
    K_bare = sympy.Symbol('K')
    i = symbolic.symbol('i', integer=True)
    assert _proven_nonnegative(K_pos - i - 1, {'i': (0, K_bare - 1)}) is True


# --------------------------------------------------------------------------- #
# Exponent classification
# --------------------------------------------------------------------------- #
def test_relaxes_symbolic_and_constant_exponents():
    K = symbolic.symbol('K', positive=True, integer=True)
    i = symbolic.symbol('i', integer=True)
    ranges, auth = {'i': (0, K - 1)}, {'K': K, 'i': i}
    assert _relaxed_exponent(K - i - 1, ranges, auth) == K - i - 1  # symbolic, proven
    assert _relaxed_exponent(sympy.Integer(3), ranges, auth) == 3  # integer constant
    assert _relaxed_exponent(sympy.Float(2.0), ranges, auth) == 2  # integer-valued float


def test_refuses_fractional_and_negative_exponents():
    assert _relaxed_exponent(sympy.Rational(1, 2), {}, {}) is None  # sqrt -> keep pow
    assert _relaxed_exponent(sympy.Integer(-2), {}, {}) is None  # reciprocal -> keep pow


# --------------------------------------------------------------------------- #
# End-to-end: stockham-like complex array with a symbolic-power shape
# --------------------------------------------------------------------------- #
def test_end_to_end_complex_power_shape_compiles():
    """A ``complex128`` array whose shape ``R**K`` is a symbolic power -- the
    stockham-fft shape.

    Both the array size and the ``0:R**K`` map bound are that symbolic power.
    ``dace::math::pow`` would render them as ``double`` (an illegal C++ array
    size / loop predicate), so the pass must relax them to ``ipow`` for the
    kernel to compile at all -- and it must stay bit-exact with NumPy.
    """
    R = dace.symbol('R', positive=True, integer=True)
    K = dace.symbol('K', positive=True, integer=True)

    sdfg = dace.SDFG('power_shape')
    sdfg.add_array('x', [R**K], dace.complex128)
    state = sdfg.add_state()
    state.add_mapped_tasklet('scale',
                             dict(j=f'0:{R**K}'), {'a': dace.Memlet('x[j]')},
                             'b = a * 2.0', {'b': dace.Memlet('x[j]')},
                             external_edges=True)

    assert RelaxIntegerPowers().apply_pass(sdfg, {}) is not None
    # Both the descriptor size and the map bound became ``ipow`` (not double pow).
    assert any(
        isinstance(e, sympy.Basic) and e.atoms(ipow) for d in sdfg.arrays.values()
        for e in (*d.shape, d.total_size)), "expected a relaxed ipow array size"

    Rv, Kv = 2, 4  # N = R**K = 16
    x = (np.arange(1, Rv**Kv + 1) + 1j * np.arange(1, Rv**Kv + 1)).astype(np.complex128)
    ref = x * 2.0
    sdfg(x=x, R=Rv, K=Kv)
    assert np.allclose(x, ref)
