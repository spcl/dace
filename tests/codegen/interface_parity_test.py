# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests that must behave identically on BOTH Python <-> C interfaces.

Every test takes the ``compiler_interface`` fixture and therefore runs once
under ``ctypes`` and once under ``nanobind``; the fixture also drops the
``DACE_compiler_interface`` environment variable, which would otherwise
override the per-test configuration (the CI matrix sets it globally).
"""
import numpy as np
import pytest

import dace
from dace.config import set_temporary


@pytest.fixture(params=['ctypes', 'nanobind'])
def compiler_interface(request):
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv('DACE_compiler_interface', raising=False)
        with set_temporary('compiler', 'interface', value=request.param):
            yield request.param


def test_vector_return_exotic_strides(compiler_interface):
    """A vector-dtype return with PERMUTED, padded strides allocates
    correctly: the lane dimension is appended as a NEW trailing dimension
    with scalar stride 1, independent of where the vector-stride-1 dimension
    sits. Descriptor: shape (2, 3, 4), strides (3, 1, 12) in VECTOR elements
    (stride-1 in the middle, dim 2 padded), total_size 48 vector elements.

    Input and output use DIFFERENT strides (same shape) and the copy goes
    through a MAP: an identical-layout whole-array copy would lower to one
    memcpy and never exercise the per-dimension addressing that must agree
    with the returned array's strides."""
    vec2 = dace.vector(dace.float64, 2)
    sdfg = dace.SDFG(f'vec_ret_exotic_{compiler_interface}')
    sdfg.add_array('A', [2, 3, 4], vec2, strides=[12, 4, 1], total_size=24)
    sdfg.add_array('__return', [2, 3, 4], vec2, strides=[3, 1, 12], total_size=48)
    state = sdfg.add_state()
    state.add_mapped_tasklet('copy',
                             dict(i='0:2', j='0:3', k='0:4'),
                             dict(inp=dace.Memlet('A[i, j, k]')),
                             'out = inp',
                             dict(out=dace.Memlet('__return[i, j, k]')),
                             external_edges=True)

    csdfg = sdfg.compile()

    # Dense C-order input: vector strides (12, 4, 1) = a contiguous
    # (2, 3, 4, 2) float64 array.
    a = np.arange(48, dtype=np.float64).reshape(2, 3, 4, 2)

    result = csdfg(A=a)
    # The lane dimension trails with scalar stride 1; the descriptor's
    # (vector) strides scale to bytes: (3*16, 1*16, 12*16, 8).
    assert result.shape == (2, 3, 4, 2)
    assert result.strides == (48, 16, 192, 8)
    # Element-wise equality across the two different memory layouts: correct
    # only if the returned array's stride metadata matches the addressing the
    # kernel actually wrote with.
    assert np.array_equal(result, a)


if __name__ == '__main__':
    for interface in ('ctypes', 'nanobind'):
        with set_temporary('compiler', 'interface', value=interface):
            test_vector_return_exotic_strides(interface)
