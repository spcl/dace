# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for write-conflicted reductions on complex types."""

import numpy as np

import dace
from dace import dtypes


def test_multicore_complex_sum_reduction():
    """A reduction into a complex scalar must compile and compute the correct sum."""
    sdfg = dace.SDFG('multicore_complex_sum')
    sdfg.add_array('a', (1024, ), dace.complex128)
    sdfg.add_array('out', (1, ), dace.complex128)
    state = sdfg.add_state('main', is_start_block=True)
    state.add_mapped_tasklet('reduce_sum', {'i': '0:1024'}, {'inp': dace.Memlet('a[i]')},
                             'o = inp', {'o': dace.Memlet('out[0]', wcr='lambda x, y: x + y')},
                             schedule=dtypes.ScheduleType.CPU_Multicore,
                             external_edges=True)
    sdfg.validate()

    a = np.random.rand(1024) + 1j * np.random.rand(1024)
    out = np.zeros(1, dtype=np.complex128)
    sdfg(a=a, out=out)
    assert np.allclose(out[0], a.sum())


def test_simd_complex_sum_reduction():
    """A SIMD-reduced complex sum must compile and compute the correct value."""
    from dace.transformation.passes.mark_simd_maps import MarkSIMDMaps
    from dace.sdfg import infer_types

    sdfg = dace.SDFG('simd_complex_sum')
    sdfg.add_array('a', (1024, ), dace.complex128)
    sdfg.add_array('out', (1, ), dace.complex128)
    state = sdfg.add_state('main', is_start_block=True)
    state.add_mapped_tasklet('reduce_sum', {'i': '0:1024'}, {'inp': dace.Memlet('a[i]')},
                             'o = inp', {'o': dace.Memlet('out[0]', wcr='lambda x, y: x + y')},
                             schedule=dtypes.ScheduleType.CPU_Multicore,
                             external_edges=True)
    sdfg.validate()
    infer_types.set_default_schedule_and_storage_types(sdfg, None)
    MarkSIMDMaps().apply_pass(sdfg, {})

    a = np.random.rand(1024) + 1j * np.random.rand(1024)
    out = np.zeros(1, dtype=np.complex128)
    sdfg(a=a, out=out)
    assert np.allclose(out[0], a.sum())


def test_reduce_node_complex_sum():
    """A library Reduce node over a complex array must compile and compute the correct value."""
    sdfg = dace.SDFG('reduce_node_complex_sum')
    sdfg.add_array('a', (1024, ), dace.complex128)
    sdfg.add_array('out', (1, ), dace.complex128)
    state = sdfg.add_state('main', is_start_block=True)
    red = state.add_reduce('lambda x, y: x + y', axes=None, identity=0)
    state.add_edge(state.add_read('a'), None, red, None, dace.Memlet('a[0:1024]'))
    state.add_edge(red, None, state.add_write('out'), None, dace.Memlet('out[0]'))
    sdfg.validate()

    a = np.random.rand(1024) + 1j * np.random.rand(1024)
    out = np.zeros(1, dtype=np.complex128)
    sdfg(a=a, out=out)
    assert np.allclose(out[0], a.sum())


if __name__ == '__main__':
    test_multicore_complex_sum_reduction()
    test_simd_complex_sum_reduction()
    test_reduce_node_complex_sum()
