# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the ``ipow`` symbolic Function and the ``RelaxIntegerPowers`` pass."""
import numpy as np
import sympy

import dace
from dace import data, symbolic
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.symbolic import ipow, pystr_to_symbolic, symstr
from dace.transformation.passes.relax_integer_powers import RelaxIntegerPowers


def _ipow_count(sdfg: dace.SDFG) -> int:
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


def _prover() -> RelaxIntegerPowers:
    p = RelaxIntegerPowers()
    p._pos = p._nonneg = p._int = frozenset()
    return p


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
    prove = _prover()._proven_nonnegative
    assert prove(K - i - 1, {'i': (0, K - 1)}) is True  # bottoms out at 0 when i = K-1
    assert prove(i, {'i': (0, K - 1)}) is True
    assert prove(K, {}) is True  # positive size symbol


def test_interval_refuses_unbounded_iterator():
    K = symbolic.symbol('K', positive=True, integer=True)
    i = symbolic.symbol('i', integer=True)
    # Without a range for i, K - i - 1 could be negative -> must not relax.
    assert _prover()._proven_nonnegative(K - i - 1, {}) is False


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
    assert _ipow_count(sdfg) > 0  # R**(K-i-1) proven >= 0 from i in [0, K-1]


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
    assert _ipow_count(sdfg) > 0


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
    assert _ipow_count(sdfg) > 0


def test_refuses_unprovable_and_negative_exponents():
    """A sign-unknown exponent and a provably-negative one stay on the ``pow`` path."""
    R = dace.symbol('R', positive=True, integer=True)
    K = dace.symbol('K', positive=True, integer=True)
    M = dace.symbol('M', integer=True)  # no sign assumption

    sdfg = dace.SDFG('unprovable')
    sdfg.add_array('u', [R**M], dace.float64)
    sdfg.add_state().add_access('u')
    RelaxIntegerPowers().apply_pass(sdfg, {})
    assert _ipow_count(sdfg) == 0
    assert sdfg.arrays['u'].shape[0].has(sympy.Pow)  # R**M unchanged

    classify = _prover()._relaxed_exponent
    assert classify(-K, {}) is None
    assert classify(sympy.Rational(1, 2), {}) is None


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
    assert _ipow_count(sdfg) > 0

    Rv, Kv = 2, 4
    x = (np.arange(1, Rv**Kv + 1) + 1j * np.arange(1, Rv**Kv + 1)).astype(np.complex128)
    ref = x * 2.0
    sdfg(x=x, R=Rv, K=Kv)
    assert np.allclose(x, ref)
