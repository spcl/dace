# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Backward passes of reductions that read and write parts of larger containers.

Under the nested SDFG contract the connectors of a backward nested SDFG are the containers the forward node reads
and writes, so the gradients have to be indexed through the forward memlets rather than through the shapes of the
containers.
"""
import numpy as np
import pytest

import dace
from dace.autodiff import add_backward_pass
from dace.sdfg.state import LoopRegion


def _reduce_into_window(wcr: str, identity: float) -> dace.SDFG:
    """``B[:, i] = reduce(X[:, i, :], axis=2)`` in a loop over ``i``, then ``__return = sum(B)``."""
    sdfg = dace.SDFG(f'reduce_into_window_{"max" if "max" in wcr else "sum"}')
    sdfg.add_array('X', [4, 3, 5], dace.float32)
    sdfg.add_array('B', [4, 3], dace.float32, transient=True)
    sdfg.add_array('__return', [1], dace.float32)

    loop = LoopRegion('rows', 'i < 3', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state('body', is_start_block=True)
    reduce = body.add_reduce(wcr, axes=[2], identity=identity)
    body.add_edge(body.add_read('X'), None, reduce, None, dace.Memlet('X[0:4, i, 0:5]'))
    body.add_edge(reduce, None, body.add_write('B'), None, dace.Memlet('B[0:4, i]'))

    total = sdfg.add_state_after(loop, 'total')
    total_reduce = total.add_reduce('lambda a, b: a + b', axes=None, identity=0)
    total.add_edge(total.add_read('B'), None, total_reduce, None, dace.Memlet('B[0:4, 0:3]'))
    total.add_edge(total_reduce, None, total.add_write('__return'), None, dace.Memlet('__return[0]'))
    sdfg.validate()
    return sdfg


@pytest.mark.autodiff
@pytest.mark.parametrize('kind', ['sum', 'max'])
def test_reduce_into_window_of_container(kind: str):
    sdfg = _reduce_into_window('lambda a, b: a + b' if kind == 'sum' else 'lambda a, b: max(a, b)',
                               0 if kind == 'sum' else -1e30)
    add_backward_pass(sdfg=sdfg, inputs=['X'], outputs=['__return'])

    X = np.random.default_rng(42).random((4, 3, 5)).astype(np.float32)
    gradient_X = np.zeros_like(X)
    sdfg(X=X.copy(), gradient_X=gradient_X, gradient___return=np.ones(1, np.float32))

    if kind == 'sum':
        expected = np.ones_like(X)
    else:
        expected = (X == X.max(axis=2, keepdims=True)).astype(np.float32)
    assert np.allclose(gradient_X, expected)


if __name__ == '__main__':
    test_reduce_into_window_of_container('sum')
    test_reduce_into_window_of_container('max')
