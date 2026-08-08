# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import dace.sdfg.nodes
import numpy as np
import pytest


@pytest.mark.parametrize('implementation', ['pure', 'OpenMP'])
def test_strided_reduce(implementation):
    """Reduce a strided input subset ``A[::2, ::2]`` over axis 0. The pure and
    OpenMP expansions index each reduced element with the ARRAY stride times the
    input subset's STEP, so the reduction walks the strided view correctly.
    (Long disabled with "Incorrect outputs" because the expansions dropped the
    subset step and read the input contiguously.)"""
    A = np.random.rand(50, 50)
    B = np.random.rand(25)

    # Python version of the SDFG below
    # @dace.program
    # def reduce_with_strides(A: dace.float64[50, 50], B: dace.float64[25]):
    #     B[:] = dace.reduce(lambda a,b: a+b, A[::2, ::2], axis=0,
    #                        identity=0)

    reduce_with_strides = dace.SDFG('reduce_with_strides')
    reduce_with_strides.add_array('A', [50, 50], dace.float64)
    reduce_with_strides.add_array('B', [25], dace.float64)

    state = reduce_with_strides.add_state()
    node_a = state.add_read('A')
    node_b = state.add_write('B')
    red = state.add_reduce('lambda a,b: a+b', [0], 0)
    red.implementation = implementation
    state.add_nedge(node_a, red, dace.Memlet.simple('A', '0:50:2, 0:50:2'))
    state.add_nedge(red, node_b, dace.Memlet.simple('B', '0:25'))
    reduce_with_strides(A=A, B=B)

    assert np.allclose(B, np.sum(A[::2, ::2], axis=0))


if __name__ == '__main__':
    test_strided_reduce('pure')
    test_strided_reduce('OpenMP')
