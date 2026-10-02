# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the ``ipow`` symbolic Function and the ``RelaxIntegerPowers`` pass."""

import numpy as np
import pytest
import sympy

import dace
from dace import data, symbolic
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.symbolic import ipow, pystr_to_symbolic, symstr
from dace.transformation.passes.relax_integer_powers import (RelaxIntegerPowers, SignFacts, loop_range, ordered_range,
                                                             proven_nonnegative, relaxed_exponent)


def ipow_count(sdfg: dace.SDFG) -> int:
    """Count ``ipow`` occurrences across descriptors, map ranges and memlet subsets."""

    def atoms(expr):
        return len(expr.atoms(ipow)) if isinstance(expr, sympy.Basic) else 0

    total = 0
    for sd in sdfg.all_sdfgs_recursive():
        for desc in sd.arrays.values():
            if isinstance(desc, data.Array):
                total += sum(atoms(e) for e in (*desc.shape, *desc.strides, desc.total_size))
        for state in sd.all_states():
            for node in state.nodes():
                if isinstance(node, nodes.MapEntry):
                    total += sum(atoms(x) for rng in node.map.range.ranges for x in rng)
            for edge in state.edges():
                if edge.data is not None and isinstance(edge.data.subset, dace.subsets.Range):
                    total += sum(atoms(x) for rng in edge.data.subset.ranges for x in rng)
    return total


def test_ipow_lowers_to_cpp_ipow():
    R, K = (symbolic.symbol(s, positive=True, integer=True) for s in ('R', 'K'))
    assert 'dace::math::ipow(R, K)' in symstr(ipow(R, K), cpp_mode=True)


def test_ipow_roundtrips_through_serialization():
    R, K, i = (symbolic.symbol(s) for s in ('R', 'K', 'i'))
    e = R * ipow(R, K - i - 1) + ipow(R, K - i - 1)
    back = pystr_to_symbolic(str(e))
    assert symstr(e, cpp_mode=True) == symstr(back, cpp_mode=True)
    assert any(type(a) is ipow for a in sympy.preorder_traversal(back))


def test_ipow_survives_property_json_roundtrip_and_folds():
    P = symbolic.symbol('P')
    e = 64 * ipow(P, 2)
    back = symbolic.deserialize_symbolic(symbolic.serialize_symbolic(e))
    assert any(type(a) is ipow for a in sympy.preorder_traversal(back))
    assert int(symbolic.evaluate(back, {P: 4})) == 64 * 16


def test_ipow_is_integer_and_positive():
    R, K = (symbolic.symbol(s, positive=True, integer=True) for s in ('R', 'K'))
    assert ipow(R, K).is_integer is True
    assert ipow(R, K).is_positive is True


def test_ipow_folds_constant_power():
    assert ipow(sympy.Integer(2), sympy.Integer(10)) == 1024


def test_interval_proves_radix_decomposition():
    K = symbolic.symbol('K', positive=True, integer=True)
    i = symbolic.symbol('i', integer=True)

    def prove(exp, ranges):
        return proven_nonnegative(exp, ranges, SignFacts())

    assert prove(K - i - 1, {'i': (0, K - 1)}) is True  # bottoms out at 0 when i = K-1
    assert prove(i, {'i': (0, K - 1)}) is True
    assert prove(K, {}) is True  # positive size symbol


def test_interval_refuses_unbounded_iterator():
    K = symbolic.symbol('K', positive=True, integer=True)
    i = symbolic.symbol('i', integer=True)
    # Without a range for i, K - i - 1 could be negative -> must not relax.
    assert proven_nonnegative(K - i - 1, {}, SignFacts()) is False


def test_relaxes_pow_inside_loop():
    """A radix subscript ``R**(K-i-1)`` in a memlet under ``for i in range(K)``
    relaxes via the iterator range the loop binds during the descent."""
    R = dace.symbol('R', positive=True, integer=True)
    K = dace.symbol('K', positive=True, integer=True)

    sdfg = dace.SDFG('loop')
    sdfg.add_array('x', [R**K], dace.float64)
    loop = LoopRegion(label='loop',
                      condition_expr='i < K',
                      loop_var='i',
                      initialize_expr='i = 0',
                      update_expr='i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state('body', is_start_block=True)
    tasklet = body.add_tasklet('set', {}, {'o'}, 'o = 1.0')
    body.add_edge(tasklet, 'o', body.add_write('x'), None, dace.Memlet('x[R**(K - i - 1) - 1]'))

    assert RelaxIntegerPowers().apply_pass(sdfg, {}) is not None
    assert ipow_count(sdfg) > 0  # R**(K-i-1) proven >= 0 from i in [0, K-1]


def test_relaxes_pow_inside_map_and_nested_sdfg():
    """An indirect access ``a[b[i]]`` inside a map lowers to a nested SDFG; a radix
    power there must relax with the map iterator carried through the symbol map."""
    R = dace.symbol('R', positive=True, integer=True)
    M = dace.symbol('M', positive=True, integer=True)

    @dace.program
    def prog(a: dace.float64[R**M], b: dace.int64[M]):
        for i in dace.map[0:M]:
            a[b[i] + R**(M - i - 1) - 1] = 2.0

    sdfg = prog.to_sdfg(simplify=False)
    assert any(isinstance(n, nodes.NestedSDFG) for n, _ in sdfg.all_nodes_recursive())
    assert RelaxIntegerPowers().apply_pass(sdfg, {}) is not None
    assert ipow_count(sdfg) > 0


def test_relaxes_under_dynamic_map_symbol():
    """A map whose range is a dynamic input (a data-dependent symbol) is handled; a
    power over a provably-positive symbol still relaxes."""
    R = dace.symbol('R', positive=True, integer=True)
    N = dace.symbol('N', positive=True, integer=True)

    sdfg = dace.SDFG('dyn')
    sdfg.add_array('x', [R**N], dace.float64)
    sdfg.add_array('bound', [1], dace.int64)
    state = sdfg.add_state()
    me, mx = state.add_map('m', dict(j='0:s'))  # dynamic upper bound symbol s
    me.add_in_connector('s')
    state.add_edge(state.add_read('bound'), None, me, 's', dace.Memlet('bound[0]'))
    t = state.add_tasklet('w', {}, {'o'}, 'o = 1.0')
    state.add_edge(me, None, t, None, dace.Memlet())
    w = state.add_write('x')
    state.add_memlet_path(t, mx, w, src_conn='o', memlet=dace.Memlet('x[R**N - 1]'))

    # Must not crash on the dynamic-range symbol, and must relax the R**N size/index.
    assert RelaxIntegerPowers().apply_pass(sdfg, {}) is not None
    assert ipow_count(sdfg) > 0


def test_refuses_unprovable_and_negative_exponents():
    """A sign-unknown exponent and a provably-negative one stay on the ``pow`` path."""
    R = dace.symbol('R', positive=True, integer=True)
    K = dace.symbol('K', positive=True, integer=True)
    M = dace.symbol('M', integer=True)  # no sign assumption

    sdfg = dace.SDFG('unprovable')
    sdfg.add_array('u', [R**M], dace.float64)
    sdfg.add_state().add_access('u')
    RelaxIntegerPowers().apply_pass(sdfg, {})
    assert ipow_count(sdfg) == 0
    assert sdfg.arrays['u'].shape[0].has(sympy.Pow)  # R**M unchanged

    assert relaxed_exponent(-K, {}, SignFacts()) is None
    assert relaxed_exponent(sympy.Rational(1, 2), {}, SignFacts()) is None


def test_end_to_end_complex_power_shape_compiles():
    """A ``complex128`` array whose shape ``R**K`` is a symbolic power (the
    stockham-fft shape) must relax to ``ipow`` -- ``pow`` would be an illegal
    ``double`` array size / map bound -- and stay bit-exact with NumPy."""
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
    assert ipow_count(sdfg) > 0

    Rv, Kv = 2, 4
    x = (np.arange(1, Rv**Kv + 1) + 1j * np.arange(1, Rv**Kv + 1)).astype(np.complex128)
    ref = x * 2.0
    sdfg(x=x, R=Rv, K=Kv)
    assert np.allclose(x, ref)


@pytest.mark.parametrize('bad', [sympy.Integer(-1), sympy.Integer(-7), sympy.Rational(-1, 2), sympy.Float(-0.5)])
def test_ipow_rejects_a_negative_constant_exponent_integral_or_not(bad):
    """A negative constant is a reciprocal, so it is a ``Pow``; the sign test does not ask for an integer."""
    with pytest.raises(ValueError):
        ipow(symbolic.symbol('R', positive=True, integer=True), bad)


def test_ipow_leaves_an_exponent_of_unknown_sign_to_the_caller():
    base = symbolic.symbol('R', positive=True, integer=True)
    unknown = symbolic.symbol('M', integer=True)
    assert isinstance(ipow(base, unknown), ipow)
    with pytest.raises(ValueError):
        ipow(base, unknown).subs(unknown, -1)


def test_a_symbolic_exponent_of_unknown_sign_is_not_relaxed_by_the_pass():
    R = dace.symbol('R', positive=True, integer=True)
    M = dace.symbol('M', integer=True)
    sdfg = dace.SDFG('unknown_sign_exponent')
    sdfg.add_array('u', [R**M], dace.float64)
    sdfg.add_state().add_access('u')
    RelaxIntegerPowers().apply_pass(sdfg, {})
    assert ipow_count(sdfg) == 0


def test_an_unsigned_exponent_is_relaxed_by_the_pass():
    R = dace.symbol('R', positive=True, integer=True)
    M = dace.symbol('M', dace.uint32)
    sdfg = dace.SDFG('unsigned_exponent')
    sdfg.add_symbol('M', dace.uint32)
    sdfg.add_array('u', [R**M], dace.float64)
    sdfg.add_state().add_access('u')
    RelaxIntegerPowers().apply_pass(sdfg, {})
    assert ipow_count(sdfg) > 0


def test_loop_range_direction_from_stride_sign():
    """``loop_range`` orders (low, high) by the stride sign, and refuses when the sign is
    unknown -- a wrong direction guess would relax a negative exponent."""
    sstep = dace.symbol('sstep', integer=True)  # unknown sign

    def rng(cond, init, update):
        return loop_range(LoopRegion('L', condition_expr=cond, loop_var='i', initialize_expr=init, update_expr=update))

    assert str(rng('i < K', 'i = 0', 'i = i + 1')) == '(0, K - 1)'  # ascending
    assert str(rng('i > 0', 'i = K', 'i = i - 1')) == '(1, K)'  # descending
    assert rng('i > 0', 'i = K', 'i = i + sstep') is None  # unknown-sign stride -> no trusted range


def test_ordered_range_accepts_raw_int_step():
    """A range step can be a raw Python ``int``, which has no ``is_positive``."""
    assert ordered_range(0, 10, 1) == (0, 10)
    assert ordered_range(10, 0, -1) == (0, 10)


def test_refuses_pow_under_unknown_sign_stride():
    """The stride sign is the only thing that changes between these two loops: ``K - p`` is
    provably >= 0 over an ascending sweep, so it relaxes; under an unknown-sign stride there is
    no trusted range, so it must stay ``pow`` (if the stride were negative it would go negative)."""
    R = dace.symbol('R', positive=True, integer=True)
    K = dace.symbol('K', positive=True, integer=True)
    p = dace.symbol('p', integer=True)
    sstep = dace.symbol('sstep', integer=True)  # unknown sign

    def count(name, update):
        sdfg = dace.SDFG(name)
        sdfg.add_array('x', [100], dace.float64)  # constant shape -> only the subscript carries a power
        loop = LoopRegion('L', condition_expr='p < K', loop_var='p', initialize_expr='p = 0', update_expr=update)
        sdfg.add_node(loop, is_start_block=True)
        body = loop.add_state('body', is_start_block=True)
        t = body.add_tasklet('set', {}, {'o'}, 'o = 1.0')
        body.add_edge(t, 'o', body.add_write('x'), None, dace.Memlet('x[R**(K - p)]'))
        RelaxIntegerPowers().apply_pass(sdfg, {})
        return ipow_count(sdfg)

    assert count('ascending', 'p = p + 1') > 0  # p in [0, K-1] -> K-p in [1, K] -> relax
    assert count('unknown', 'p = p + sstep') == 0  # unknown-sign stride -> stays pow


def test_descending_loop_still_relaxes():
    """A radix subscript proven non-negative over a descending loop's range relaxes (direction
    is handled, not just refused)."""
    R = dace.symbol('R', positive=True, integer=True)
    K = dace.symbol('K', positive=True, integer=True)

    sdfg = dace.SDFG('descending')
    sdfg.add_array('x', [100], dace.float64)
    loop = LoopRegion('L', condition_expr='i > 0', loop_var='i', initialize_expr='i = K', update_expr='i = i - 1')
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state('body', is_start_block=True)
    t = body.add_tasklet('set', {}, {'o'}, 'o = 1.0')
    body.add_edge(t, 'o', body.add_write('x'), None, dace.Memlet('x[R**(K - i)]'))  # i in [1, K] -> K-i in [0, K-1]

    RelaxIntegerPowers().apply_pass(sdfg, {})
    assert ipow_count(sdfg) > 0


def test_loop_condition_off_by_one_not_relaxed():
    """The loop condition is evaluated one past the body range (``i = end + step``), so an
    exponent that uses the iterator must not relax there; a constant-exponent bound still does."""
    R = dace.symbol('R', positive=True, integer=True)
    K = dace.symbol('K', positive=True, integer=True)

    def build(cond):
        sdfg = dace.SDFG('cond')
        sdfg.add_array('x', [R**K], dace.float64)  # shape carries K -> pass knows K is a positive int
        loop = LoopRegion('L', condition_expr=cond, loop_var='i', initialize_expr='i = 0', update_expr='i = i + 1')
        sdfg.add_node(loop, is_start_block=True)
        loop.add_state('body', is_start_block=True)
        RelaxIntegerPowers().apply_pass(sdfg, {})
        return loop.loop_condition.as_string

    assert 'ipow' not in build('i < R**(K - i - 1)')  # exponent uses i, negative at i = K
    assert 'ipow' in build('i < R**K')  # constant exponent -> safe to relax


def test_int64_power_compiles_as_an_integer():
    """A power of int64 symbols the pass proves non-negative lowers to ``ipow`` and stays integral: as a
    double it is neither a legal OpenMP controlling predicate nor a pointer offset."""
    R = dace.symbol('R', dtype=dace.int64)
    K = dace.symbol('K', dtype=dace.int64, nonnegative=True)

    @dace.program
    def tester(A: dace.float64[R**K]):
        for i in dace.map[0:R**K]:
            A[i] = 1.0

    a = np.zeros(8)
    tester(a, R=2, K=3)
    assert np.allclose(a, 1.0)


if __name__ == '__main__':
    test_ipow_lowers_to_cpp_ipow()
    test_ipow_roundtrips_through_serialization()
    test_ipow_survives_property_json_roundtrip_and_folds()
    test_ipow_is_integer_and_positive()
    test_ipow_folds_constant_power()
    test_interval_proves_radix_decomposition()
    test_interval_refuses_unbounded_iterator()
    test_relaxes_pow_inside_loop()
    test_relaxes_pow_inside_map_and_nested_sdfg()
    test_relaxes_under_dynamic_map_symbol()
    test_refuses_unprovable_and_negative_exponents()
    test_end_to_end_complex_power_shape_compiles()
    for bad in (sympy.Integer(-1), sympy.Integer(-7), sympy.Rational(-1, 2), sympy.Float(-0.5)):
        test_ipow_rejects_a_negative_constant_exponent_integral_or_not(bad)
    test_ipow_leaves_an_exponent_of_unknown_sign_to_the_caller()
    test_a_symbolic_exponent_of_unknown_sign_is_not_relaxed_by_the_pass()
    test_an_unsigned_exponent_is_relaxed_by_the_pass()
    test_loop_range_direction_from_stride_sign()
    test_ordered_range_accepts_raw_int_step()
    test_refuses_pow_under_unknown_sign_stride()
    test_descending_loop_still_relaxes()
    test_loop_condition_off_by_one_not_relaxed()
    test_int64_power_compiles_as_an_integer()
