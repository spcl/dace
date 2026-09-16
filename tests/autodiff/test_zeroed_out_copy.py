# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests gradients through a copy into a transient that the backward pass zeroes out."""
import numpy as np
import pytest

import dace
from dace.autodiff import add_backward_pass

N = 8


@pytest.mark.autodiff
def test_copy_into_zeroed_out_scalar_in_map():
    """The copy ``A[i] -> T`` names ``A`` (with ``T``'s side as other subset) inside a map."""
    sdfg = dace.SDFG('zeroed_out_copy')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('B', [1], dace.float64)
    sdfg.add_scalar('T', dace.float64, transient=True)
    state = sdfg.add_state()
    a = state.add_read('A')
    b = state.add_write('B')
    me, mx = state.add_map('m', dict(i=f'0:{N}'))
    t = state.add_access('T')
    tasklet = state.add_tasklet('scale', {'__in'}, {'__out'}, '__out = 3.0 * __in')
    me.add_in_connector('IN_A')
    me.add_out_connector('OUT_A')
    state.add_edge(a, None, me, 'IN_A', dace.Memlet(f'A[0:{N}]'))
    state.add_edge(me, 'OUT_A', t, None, dace.Memlet(data='A', subset='i', other_subset='0'))
    state.add_edge(t, None, tasklet, '__in', dace.Memlet('T[0]'))
    mx.add_in_connector('IN_B')
    mx.add_out_connector('OUT_B')
    state.add_edge(tasklet, '__out', mx, 'IN_B', dace.Memlet('B[0]', wcr='lambda x, y: x + y'))
    state.add_edge(mx, 'OUT_B', b, None, dace.Memlet('B[0]', wcr='lambda x, y: x + y'))
    sdfg.validate()

    add_backward_pass(sdfg=sdfg, inputs=['A'], outputs=['B'], simplify=False)
    sdfg.validate()

    A = np.random.rand(N)
    B = np.zeros(1)
    gradient_A = np.zeros(N)
    gradient_B = np.ones(1)
    sdfg(A=A, B=B, gradient_A=gradient_A, gradient_B=gradient_B)
    assert np.allclose(B, np.sum(3.0 * A))
    assert np.allclose(gradient_A, np.full(N, 3.0))


if __name__ == '__main__':
    test_copy_into_zeroed_out_scalar_in_map()
