# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Reverse-mode differentiation of reductions whose results feed further computation (not only the loss)."""
import numpy as np

import pytest

import dace
from dace.autodiff import add_backward_pass
from dace.libraries.standard import Reduce

M, N = 4, 6


def _name_reduction_connectors(sdfg: dace.SDFG) -> None:
    """Gives the Reduce nodes named connectors (``_in``/``_out``), as frontends other than NumPy's create them."""
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, Reduce):
            node.add_in_connector('_in')
            node.add_out_connector('_out')
            for edge in state.in_edges(node):
                edge.dst_conn = '_in'
            for edge in state.out_edges(node):
                edge.src_conn = '_out'


def _gradient(program, x, named_connectors):
    sdfg = program.to_sdfg(simplify=True)
    if named_connectors:
        _name_reduction_connectors(sdfg)
    add_backward_pass(sdfg, outputs=['loss'], inputs=['x'])
    gradient = np.zeros_like(x)
    sdfg(x=x.copy(), loss=np.zeros(1), gradient_x=gradient, gradient_loss=np.ones(1))
    return gradient


@pytest.mark.parametrize('named_connectors', (False, True))
def test_intermediate_sum_reduction(named_connectors):

    @dace.program
    def row_sums_squared(x: dace.float64[M, N], loss: dace.float64[1]):
        rows = np.sum(x, axis=1)
        loss[0] = np.sum(rows * rows)

    x = np.random.rand(M, N)
    expected = np.repeat(2 * x.sum(axis=1, keepdims=True), N, axis=1)
    np.testing.assert_allclose(_gradient(row_sums_squared, x, named_connectors), expected)


@pytest.mark.parametrize('named_connectors', (False, True))
def test_intermediate_max_reduction(named_connectors):

    @dace.program
    def row_max_scaled(x: dace.float64[M, N], loss: dace.float64[1]):
        rows = np.max(x, axis=1)
        loss[0] = np.sum(rows * 3.0)

    x = np.random.rand(M, N)
    expected = np.where(x == x.max(axis=1, keepdims=True), 3.0, 0.0)
    np.testing.assert_allclose(_gradient(row_max_scaled, x, named_connectors), expected)


if __name__ == '__main__':
    for named in (False, True):
        test_intermediate_sum_reduction(named)
        test_intermediate_max_reduction(named)
