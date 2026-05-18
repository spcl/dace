# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for the ``MinimizeStridePermutation`` canonicalization pass.

    The pass permutes perfectly-nested, single-parameter map dimensions so the
    innermost parameter indexes the smallest-stride (contiguous) array axis. It
    only emits ``MapInterchange`` applications, so the result is unchanged.

    Reordering happens only when every stride is a concrete (symbol-free)
    number. With symbolic shapes the relative magnitudes are undecidable
    (``N*M`` versus ``M``), so the nest is intentionally left untouched -- a
    safe, idempotent no-op rather than a guess. Each scenario therefore has an
    integer-dimension variant (reorder expected) and a symbolic one (no-op).

    Kernels use the dace Python frontend only. A single multi-dimensional
    ``dace.map`` is split into a clean nested single-parameter nest with the
    standard ``MapExpansion`` transformation.
"""
import copy

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.dataflow import MapExpansion
from dace.transformation.passes.minimize_stride_permutation import MinimizeStridePermutation

K, N, M = (dace.symbol(s) for s in ('K', 'N', 'M'))


@dace.program
def transposed_2d_int(A: dace.float64[7, 11], B: dace.float64[7, 11]):
    for i, j in dace.map[0:11, 0:7]:
        B[j, i] = A[j, i] * 2.0


@dace.program
def transposed_2d_sym(A: dace.float64[N, M], B: dace.float64[N, M]):
    for i, j in dace.map[0:M, 0:N]:
        B[j, i] = A[j, i] * 2.0


@dace.program
def canonical_2d_int(A: dace.float64[11, 7], B: dace.float64[11, 7]):
    for i, j in dace.map[0:11, 0:7]:
        B[i, j] = A[i, j] * 2.0


@dace.program
def transposed_3d_int(A: dace.float64[5, 7, 11], B: dace.float64[5, 7, 11]):
    for i, j, k in dace.map[0:11, 0:7, 0:5]:
        B[k, j, i] = A[k, j, i] + 1.0


@dace.program
def transposed_3d_sym(A: dace.float64[K, N, M], B: dace.float64[K, N, M]):
    for i, j, k in dace.map[0:M, 0:N, 0:K]:
        B[k, j, i] = A[k, j, i] + 1.0


@dace.program
def transposed_2d_mixed(A: dace.float64[7, M], B: dace.float64[7, M]):
    # Strides (M, 1): one symbolic, one concrete -> order undecidable.
    for i, j in dace.map[0:M, 0:7]:
        B[j, i] = A[j, i] * 2.0


@dace.program
def transposed_3d_mixed(A: dace.float64[5, N, 11], B: dace.float64[5, N, 11]):
    # Strides (N*11, 11, 1): outermost symbolic, inner two concrete.
    for i, j, k in dace.map[0:11, 0:N, 0:5]:
        B[k, j, i] = A[k, j, i] + 1.0


@dace.program
def transposed_3d_mixed_inner_symbolic(A: dace.float64[7, 5, M], B: dace.float64[7, 5, M]):
    # Strides (5*M, M, 1): two symbolic, one concrete.
    for i, j, k in dace.map[0:M, 0:5, 0:7]:
        B[k, j, i] = A[k, j, i] - 1.0


def _expanded(program) -> dace.SDFG:
    """Build the program's SDFG and split its multi-dimensional map into a
    clean nested single-parameter map nest.

    :param program: The ``@dace.program`` to compile.
    :return: The SDFG with the map expanded into a perfect nest.
    """
    sdfg = program.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(MapExpansion)
    return sdfg


def _nest_param_order(sdfg: dace.SDFG):
    """Return the single-parameter map params, outermost to innermost, of the
    first perfect map nest found in ``sdfg``.

    :param sdfg: The SDFG to inspect.
    :return: The list of parameter names from outer to inner.
    """
    for state in sdfg.all_states():
        children = state.scope_children()
        for node in children[None]:
            if not isinstance(node, nodes.MapEntry) or state.entry_node(node) is not None:
                continue
            order, current = [], node
            while current is not None:
                if len(current.map.params) != 1:
                    break
                order.append(current.map.params[0])
                exit_node = state.exit_node(current)
                body = [c for c in children.get(current, []) if c is not exit_node]
                nxt = [c for c in body if isinstance(c, nodes.MapEntry)]
                current = nxt[0] if len(body) == 1 and len(nxt) == 1 else None
            if len(order) >= 2:
                return order
    return []


def test_integer_dims_two_level_reordered():
    sdfg = _expanded(transposed_2d_int)
    assert _nest_param_order(sdfg) == ['i', 'j']

    a = np.random.rand(7, 11)
    ref = np.zeros((7, 11))
    copy.deepcopy(sdfg)(A=a.copy(), B=ref)

    assert MinimizeStridePermutation().apply_pass(sdfg, {}) is not None
    # Strides (11, 1): ``i`` (unit) must become innermost.
    assert _nest_param_order(sdfg) == ['j', 'i']

    out = np.zeros((7, 11))
    sdfg(A=a.copy(), B=out)
    assert np.allclose(out, ref) and np.allclose(out, a * 2.0)


def test_integer_dims_already_canonical_is_noop():
    sdfg = _expanded(canonical_2d_int)
    assert _nest_param_order(sdfg) == ['i', 'j']
    assert MinimizeStridePermutation().apply_pass(sdfg, {}) is None
    assert _nest_param_order(sdfg) == ['i', 'j']


def test_integer_dims_three_level_reordered():
    sdfg = _expanded(transposed_3d_int)
    assert _nest_param_order(sdfg) == ['i', 'j', 'k']

    a = np.random.rand(5, 7, 11)
    ref = np.zeros((5, 7, 11))
    copy.deepcopy(sdfg)(A=a.copy(), B=ref)

    MinimizeStridePermutation().apply_pass(sdfg, {})
    # Strides (77, 11, 1): ``i`` innermost, ``k`` outermost.
    assert _nest_param_order(sdfg) == ['k', 'j', 'i']

    out = np.zeros((5, 7, 11))
    sdfg(A=a.copy(), B=out)
    assert np.allclose(out, ref) and np.allclose(out, a + 1.0)


def test_symbolic_dims_two_level_is_safe_noop():
    """ Symbolic strides are incomparable: the pass must leave the nest
        unchanged (intended), stay idempotent, and preserve the result.
    """
    n, m = 12, 17
    sdfg = _expanded(transposed_2d_sym)
    assert _nest_param_order(sdfg) == ['i', 'j']

    a = np.random.rand(n, m)
    ref = np.zeros((n, m))
    copy.deepcopy(sdfg)(A=a.copy(), B=ref, N=n, M=m)

    assert MinimizeStridePermutation().apply_pass(sdfg, {}) is None
    assert _nest_param_order(sdfg) == ['i', 'j']

    out = np.zeros((n, m))
    sdfg(A=a.copy(), B=out, N=n, M=m)
    assert np.allclose(out, ref) and np.allclose(out, a * 2.0)


def test_symbolic_dims_three_level_is_safe_noop():
    k, n, m = 5, 7, 11
    sdfg = _expanded(transposed_3d_sym)
    assert _nest_param_order(sdfg) == ['i', 'j', 'k']

    a = np.random.rand(k, n, m)
    ref = np.zeros((k, n, m))
    copy.deepcopy(sdfg)(A=a.copy(), B=ref, K=k, N=n, M=m)

    assert MinimizeStridePermutation().apply_pass(sdfg, {}) is None
    assert _nest_param_order(sdfg) == ['i', 'j', 'k']

    out = np.zeros((k, n, m))
    sdfg(A=a.copy(), B=out, K=k, N=n, M=m)
    assert np.allclose(out, ref) and np.allclose(out, a + 1.0)


def _assert_mixed_safe_noop(program, shape, symbols, order_before):
    """A nest mixing static-integer and symbolic dimensions has an
    undeducible stride order, so the pass must do nothing -- twice
    (idempotent) -- and preserve the numerical result.

    :param program: The ``@dace.program``.
    :param shape: Concrete array shape for the run.
    :param symbols: Symbol kwargs (e.g. ``dict(M=17)``).
    :param order_before: Expected parameter order, unchanged by the pass.
    """
    sdfg = _expanded(program)
    assert _nest_param_order(sdfg) == order_before

    a = np.random.rand(*shape)
    ref = np.zeros(shape)
    copy.deepcopy(sdfg)(A=a.copy(), B=ref, **symbols)

    # No permutation, and a second run is still a no-op (idempotent).
    assert MinimizeStridePermutation().apply_pass(sdfg, {}) is None
    assert MinimizeStridePermutation().apply_pass(sdfg, {}) is None
    assert _nest_param_order(sdfg) == order_before

    out = np.zeros(shape)
    sdfg(A=a.copy(), B=out, **symbols)
    assert np.allclose(out, ref)


def test_mixed_dims_2d_one_symbolic_is_safe_noop():
    """Static dim 7 + symbolic dim M: M vs 1 is undecidable (M could be
    0/1), so the pass must not permute."""
    _assert_mixed_safe_noop(transposed_2d_mixed, (7, 17), dict(M=17), ['i', 'j'])


def test_mixed_dims_3d_outer_symbolic_is_safe_noop():
    """Strides (N*11, 11, 1): the symbolic outer stride makes the order
    undeducible even though two strides are concrete -> no-op."""
    _assert_mixed_safe_noop(transposed_3d_mixed, (5, 7, 11), dict(N=7), ['i', 'j', 'k'])


def test_mixed_dims_3d_two_symbolic_is_safe_noop():
    """Strides (5*M, M, 1): two symbolic, one concrete -> no-op."""
    _assert_mixed_safe_noop(transposed_3d_mixed_inner_symbolic, (7, 5, 13), dict(M=13),
                            ['i', 'j', 'k'])


def test_all_concrete_mixed_magnitudes_still_reorders():
    """Sanity: when every dimension is a concrete integer (even with very
    different magnitudes) the order IS deducible, so the pass still
    permutes -- the symbolic guard does not over-suppress."""
    sdfg = _expanded(transposed_3d_int)
    assert _nest_param_order(sdfg) == ['i', 'j', 'k']
    a = np.random.rand(5, 7, 11)
    ref = np.zeros((5, 7, 11))
    copy.deepcopy(sdfg)(A=a.copy(), B=ref)
    assert MinimizeStridePermutation().apply_pass(sdfg, {}) is not None
    assert _nest_param_order(sdfg) == ['k', 'j', 'i']
    out = np.zeros((5, 7, 11))
    sdfg(A=a.copy(), B=out)
    assert np.allclose(out, ref) and np.allclose(out, a + 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
