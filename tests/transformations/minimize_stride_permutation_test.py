# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the ``MinimizeStridePermutation`` canonicalization pass.

The pass permutes perfectly-nested, single-parameter map dimensions so the
innermost parameter indexes the smallest-stride (contiguous) array axis. It
only emits ``MapInterchange`` applications, so the result is unchanged.

Reordering is decided symbolically. An array's strides are products of its
shape extents and an extent is at least one wherever the array exists, so
``1`` versus ``M`` and ``11`` versus ``11*N`` both follow from the shape
contract and are permuted exactly like concrete strides, and so are ``M``
versus ``N*M``. A pair the pass does not manage to decide (``N*M`` versus
``M + K``) leaves the nest untouched instead -- a safe, idempotent no-op
rather than a guess. Each scenario therefore has an
integer-dimension variant and a symbolic one, and both reorder unless the
pass declines the comparison.

Kernels use the dace Python frontend only. A single multi-dimensional
``dace.map`` is split into a clean nested single-parameter nest with the
standard ``MapExpansion`` transformation.
"""

import copy

import numpy as np

import dace
from dace.sdfg import nodes
from dace.transformation.dataflow import MapExpansion
from dace.transformation.passes.minimize_stride_permutation import MinimizeStridePermutation

K, N, M = (dace.symbol(s) for s in ("K", "N", "M"))


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
    # Strides (M, 1): 1 <= M for every extent, so the order follows.
    for i, j in dace.map[0:M, 0:7]:
        B[j, i] = A[j, i] * 2.0


@dace.program
def transposed_3d_mixed(A: dace.float64[5, N, 11], B: dace.float64[5, N, 11]):
    # Strides (N*11, 11, 1): outermost symbolic, inner two concrete.
    # 11 <= 11*N follows from N >= 1, so the order follows.
    for i, j, k in dace.map[0:11, 0:N, 0:5]:
        B[k, j, i] = A[k, j, i] + 1.0


@dace.program
def transposed_3d_mixed_inner_symbolic(A: dace.float64[7, 5, M], B: dace.float64[7, 5, M]):
    # Strides (5*M, M, 1): two symbolic, one concrete; M <= 5*M for every M.
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
    for state in sdfg.states():
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
    rng = np.random.default_rng(42)
    sdfg = _expanded(transposed_2d_int)
    assert _nest_param_order(sdfg) == ["i", "j"]

    a = rng.random((7, 11))
    ref = np.zeros((7, 11))
    copy.deepcopy(sdfg)(A=a.copy(), B=ref)

    assert MinimizeStridePermutation().apply_pass(sdfg, {}) is not None
    # Strides (11, 1): ``i`` (unit) must become innermost.
    assert _nest_param_order(sdfg) == ["j", "i"]

    out = np.zeros((7, 11))
    sdfg(A=a.copy(), B=out)
    assert np.allclose(out, ref) and np.allclose(out, a * 2.0)


def test_integer_dims_already_canonical_is_noop():
    sdfg = _expanded(canonical_2d_int)
    assert _nest_param_order(sdfg) == ["i", "j"]
    assert MinimizeStridePermutation().apply_pass(sdfg, {}) is None
    assert _nest_param_order(sdfg) == ["i", "j"]


def test_integer_dims_three_level_reordered():
    rng = np.random.default_rng(42)
    sdfg = _expanded(transposed_3d_int)
    assert _nest_param_order(sdfg) == ["i", "j", "k"]

    a = rng.random((5, 7, 11))
    ref = np.zeros((5, 7, 11))
    copy.deepcopy(sdfg)(A=a.copy(), B=ref)

    MinimizeStridePermutation().apply_pass(sdfg, {})
    # Strides (77, 11, 1): ``i`` innermost, ``k`` outermost.
    assert _nest_param_order(sdfg) == ["k", "j", "i"]

    out = np.zeros((5, 7, 11))
    sdfg(A=a.copy(), B=out)
    assert np.allclose(out, ref) and np.allclose(out, a + 1.0)


def test_symbolic_dims_two_level_reordered():
    """Strides ``(M, 1)``: ``M >= 1`` wherever the array exists, so ``1 <= M``
    follows from the shape contract and ``i`` (the unit-stride parameter)
    belongs innermost -- the same order the concrete ``(11, 1)`` nest gets.
    """
    rng = np.random.default_rng(42)
    n, m = 12, 17
    sdfg = _expanded(transposed_2d_sym)
    assert _nest_param_order(sdfg) == ["i", "j"]

    a = rng.random((n, m))
    ref = np.zeros((n, m))
    copy.deepcopy(sdfg)(A=a.copy(), B=ref, N=n, M=m)

    # One adjacent interchange realizes the two-level reversal.
    applied = MinimizeStridePermutation().apply_pass(sdfg, {})
    assert applied is not None and sum(applied.values()) == 1
    assert _nest_param_order(sdfg) == ["j", "i"]
    # The permuted nest is canonical: a second run finds nothing to swap.
    assert MinimizeStridePermutation().apply_pass(sdfg, {}) is None
    assert _nest_param_order(sdfg) == ["j", "i"]

    out = np.zeros((n, m))
    sdfg(A=a.copy(), B=out, N=n, M=m)
    assert np.allclose(out, ref) and np.allclose(out, a * 2.0)


def test_symbolic_dims_three_level_reordered():
    """Strides ``(N*M, M, 1)``: ``N*M - M`` factors to ``M*(N - 1)``, nonnegative for extents of at least one, so
    the fully symbolic nest reaches the ``k, j, i`` order its concrete twin does."""
    _assert_mixed_reordered(transposed_3d_sym, (5, 7, 11), dict(K=5, N=7, M=11), ["i", "j", "k"], ["k", "j", "i"], 3)


def _assert_mixed_reordered(program, shape, symbols, order_before, order_after, interchanges):
    """A nest mixing static-integer and symbolic dimensions still has a
    deducible stride order, because every stride is a product of extents that
    are each at least one. The pass must reach ``order_after``, be idempotent
    there, and preserve the numerical result.

    :param program: The ``@dace.program``.
    :param shape: Concrete array shape for the run.
    :param symbols: Symbol kwargs (e.g. ``dict(M=17)``).
    :param order_before: Expected parameter order before the pass.
    :param order_after: Expected parameter order after the pass.
    :param interchanges: Expected number of adjacent ``MapInterchange``
                         applications (1 for a two-level reversal, 3 for a
                         three-level one).
    """
    rng = np.random.default_rng(42)
    sdfg = _expanded(program)
    assert _nest_param_order(sdfg) == order_before

    a = rng.random(shape)
    ref = np.zeros(shape)
    copy.deepcopy(sdfg)(A=a.copy(), B=ref, **symbols)

    applied = MinimizeStridePermutation().apply_pass(sdfg, {})
    assert applied is not None and sum(applied.values()) == interchanges
    assert _nest_param_order(sdfg) == order_after
    # The permuted nest is canonical, so a second run is a no-op (idempotent).
    assert MinimizeStridePermutation().apply_pass(sdfg, {}) is None
    assert _nest_param_order(sdfg) == order_after

    out = np.zeros(shape)
    sdfg(A=a.copy(), B=out, **symbols)
    assert np.allclose(out, ref)


def test_mixed_dims_2d_one_symbolic_reordered():
    """Static dim 7 + symbolic dim M, strides (M, 1): ``1 <= M`` holds for
    every extent the array exists at, so ``i`` moves innermost. At the
    degenerate ``M == 1`` the two strides tie and either order is equally
    contiguous; at ``M == 0`` the map is empty."""
    _assert_mixed_reordered(transposed_2d_mixed, (7, 17), dict(M=17), ["i", "j"], ["j", "i"], 1)


def test_mixed_dims_3d_outer_symbolic_reordered():
    """Strides (N*11, 11, 1): one symbolic stride does not block the order,
    because ``11 <= 11*N`` follows from ``N >= 1``. The nest reaches the same
    ``k, j, i`` order as its all-concrete (77, 11, 1) twin."""
    _assert_mixed_reordered(transposed_3d_mixed, (5, 7, 11), dict(N=7), ["i", "j", "k"], ["k", "j", "i"], 3)


def test_mixed_dims_3d_two_symbolic_reordered():
    """Strides (5*M, M, 1): two symbolic strides still order, because they
    share the symbolic factor and ``M <= 5*M`` for every M."""
    _assert_mixed_reordered(
        transposed_3d_mixed_inner_symbolic, (7, 5, 13), dict(M=13), ["i", "j", "k"], ["k", "j", "i"], 3
    )


def test_all_concrete_mixed_magnitudes_still_reorders():
    """Sanity: when every dimension is a concrete integer (even with very
    different magnitudes) the order IS deducible, so the pass still
    permutes -- the symbolic guard does not over-suppress."""
    rng = np.random.default_rng(42)
    sdfg = _expanded(transposed_3d_int)
    assert _nest_param_order(sdfg) == ["i", "j", "k"]
    a = rng.random((5, 7, 11))
    ref = np.zeros((5, 7, 11))
    copy.deepcopy(sdfg)(A=a.copy(), B=ref)
    assert MinimizeStridePermutation().apply_pass(sdfg, {}) is not None
    assert _nest_param_order(sdfg) == ["k", "j", "i"]
    out = np.zeros((5, 7, 11))
    sdfg(A=a.copy(), B=out)
    assert np.allclose(out, ref) and np.allclose(out, a + 1.0)


@dace.program
def level_map_over_plane_map(A: dace.float64[N, K, M], B: dace.float64[N, K, M]):
    for k in dace.map[0:K]:
        for i, j in dace.map[0:N, 0:M]:
            B[i, k, j] = A[i, k, j] + 1.0


def _flat_param_order(sdfg: dace.SDFG) -> list[str]:
    """Every parameter of the first top-level map nest, outermost first, across multi-parameter levels."""
    for state in sdfg.states():
        children = state.scope_children()
        for node in children[None]:
            if isinstance(node, nodes.MapEntry):
                order, current = [], node
                while current is not None:
                    order += current.map.params
                    body = [c for c in children[current] if c is not state.exit_node(current)]
                    current = body[0] if len(body) == 1 and isinstance(body[0], nodes.MapEntry) else None
                return order
    return []


def test_multi_parameter_map_is_split_and_reordered():
    """A two-parameter map over ``B[j, i]`` (strides ``(M, 1)``) is split into one map per parameter so ``i``, the
    unit-stride one, can move innermost; nothing pre-expands it."""
    rng = np.random.default_rng(42)
    sdfg = transposed_2d_sym.to_sdfg(simplify=True)
    assert _flat_param_order(sdfg) == ["i", "j"]
    a = rng.random((12, 17))
    ref = np.zeros((12, 17))
    copy.deepcopy(sdfg)(A=a.copy(), B=ref, N=12, M=17)

    assert MinimizeStridePermutation().apply_pass(sdfg, {}) is not None
    assert _flat_param_order(sdfg) == ["j", "i"]
    assert MinimizeStridePermutation().apply_pass(sdfg, {}) is None

    out = np.zeros((12, 17))
    sdfg(A=a.copy(), B=out, N=12, M=17)
    assert np.allclose(out, ref)


def test_level_map_moves_between_the_plane_axes():
    """A map over the middle axis ``k`` around a plane map over ``(i, j)`` of ``B[i, k, j]``: ``k``'s stride ``M``
    sits between ``i``'s ``K*M`` and ``j``'s 1, so the nest becomes ``i, k, j`` -- the plane's outer axis outside
    the level, its contiguous axis still innermost."""
    rng = np.random.default_rng(42)
    sdfg = level_map_over_plane_map.to_sdfg(simplify=True)
    assert _flat_param_order(sdfg) == ["k", "i", "j"]
    a = rng.random((6, 5, 7))
    ref = np.zeros((6, 5, 7))
    copy.deepcopy(sdfg)(A=a.copy(), B=ref, N=6, K=5, M=7)

    assert MinimizeStridePermutation().apply_pass(sdfg, {}) is not None
    assert _flat_param_order(sdfg) == ["i", "k", "j"]

    out = np.zeros((6, 5, 7))
    sdfg(A=a.copy(), B=out, N=6, K=5, M=7)
    assert np.allclose(out, ref) and np.allclose(out, a + 1.0)


if __name__ == "__main__":
    test_integer_dims_two_level_reordered()
    test_integer_dims_already_canonical_is_noop()
    test_integer_dims_three_level_reordered()
    test_symbolic_dims_two_level_reordered()
    test_symbolic_dims_three_level_reordered()
    test_mixed_dims_2d_one_symbolic_reordered()
    test_mixed_dims_3d_outer_symbolic_reordered()
    test_mixed_dims_3d_two_symbolic_reordered()
    test_all_concrete_mixed_magnitudes_still_reorders()
    test_multi_parameter_map_is_split_and_reordered()
    test_level_map_moves_between_the_plane_axes()
