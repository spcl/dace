# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for the ``MinimizeStridePermutation`` canonicalization pass.

    The pass permutes perfectly-nested, single-parameter map dimensions so the
    innermost parameter indexes the smallest-stride (contiguous) array axis. It
    only emits ``MapInterchange`` applications, so the SDFG result is unchanged;
    a pure permutation of independent perfectly-nested maps is numerically
    identical regardless of order.

    Kernels use the dace Python frontend only. A single multi-dimensional
    ``dace.map`` is split into a clean nested single-parameter nest with the
    standard ``MapExpansion`` transformation (no intervening access nodes),
    which is exactly the shape the pass operates on.
"""
import copy

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.dataflow import MapExpansion
from dace.transformation.passes.minimize_stride_permutation import MinimizeStridePermutation

M, N, K = (dace.symbol(s) for s in ('M', 'N', 'K'))


@dace.program
def transposed_2d(A: dace.float64[N, M], B: dace.float64[N, M]):
    for i, j in dace.map[0:M, 0:N]:
        B[j, i] = A[j, i] * 2.0


@dace.program
def canonical_2d(A: dace.float64[M, N], B: dace.float64[M, N]):
    for i, j in dace.map[0:M, 0:N]:
        B[i, j] = A[i, j] * 2.0


@dace.program
def transposed_3d(A: dace.float64[5, 7, 11], B: dace.float64[5, 7, 11]):
    for i, j, k in dace.map[0:11, 0:7, 0:5]:
        B[k, j, i] = A[k, j, i] + 1.0


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


def test_reorders_transposed_access():
    n, m = 12, 17
    sdfg = _expanded(transposed_2d)
    assert _nest_param_order(sdfg) == ['i', 'j']

    a = np.random.rand(n, m)
    ref_b = np.zeros((n, m))
    ref_sdfg = copy.deepcopy(sdfg)
    ref_sdfg(A=a.copy(), B=ref_b, N=n, M=m)

    changed = MinimizeStridePermutation().apply_pass(sdfg, {})
    assert changed is not None, "pass did not reorder a sub-optimal nest"
    # ``j`` indexes the contiguous (stride-1) axis, so it must become innermost.
    assert _nest_param_order(sdfg) == ['j', 'i']

    out_b = np.zeros((n, m))
    sdfg(A=a.copy(), B=out_b, N=n, M=m)
    assert np.allclose(out_b, ref_b)
    assert np.allclose(out_b, a * 2.0)


def test_already_canonical_is_noop():
    sdfg = _expanded(canonical_2d)
    assert _nest_param_order(sdfg) == ['i', 'j']
    assert MinimizeStridePermutation().apply_pass(sdfg, {}) is None
    assert _nest_param_order(sdfg) == ['i', 'j']


def test_idempotence():
    n, m = 9, 13
    sdfg = _expanded(transposed_2d)
    MinimizeStridePermutation().apply_pass(sdfg, {})
    order_after_first = _nest_param_order(sdfg)
    assert MinimizeStridePermutation().apply_pass(sdfg, {}) is None
    assert _nest_param_order(sdfg) == order_after_first

    a = np.random.rand(n, m)
    b = np.zeros((n, m))
    sdfg(A=a.copy(), B=b, N=n, M=m)
    assert np.allclose(b, a * 2.0)


def test_three_level_nest_reordered():
    sdfg = _expanded(transposed_3d)
    assert _nest_param_order(sdfg) == ['i', 'j', 'k']

    a = np.random.rand(5, 7, 11)
    ref = np.zeros((5, 7, 11))
    ref_sdfg = copy.deepcopy(sdfg)
    ref_sdfg(A=a.copy(), B=ref)

    MinimizeStridePermutation().apply_pass(sdfg, {})
    # Concrete strides are (77, 11, 1): ``i`` (unit) innermost, ``k`` outermost.
    assert _nest_param_order(sdfg) == ['k', 'j', 'i']

    out = np.zeros((5, 7, 11))
    sdfg(A=a.copy(), B=out)
    assert np.allclose(out, ref)
    assert np.allclose(out, a + 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
