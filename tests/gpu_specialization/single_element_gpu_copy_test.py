# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Single-element copies between GPU and CPU through ``CopyLibraryNode``.

These cover the corner case where the copy descriptor / memlet has only one
element. Without proper pointer typing on the memcpy tasklet's connectors,
``infer_*_connector_type`` types ``_memcpy_in`` as ``T`` (a value) instead of
``T *`` (a buffer), and the emitted ``cudaMemcpyAsync(_memcpy_out,
_memcpy_in, ...)`` fails to compile because ``_memcpy_in`` is a dereferenced
local rather than a pointer.

The kernel matrix in ``npbench_new_gpu_pipeline_test.py`` exercises this
incidentally (e.g. syr2k's ``B_index``); these tests reproduce it directly
and run far faster.
"""
import numpy as np
import pytest

import dace
from dace import dtypes
from dace.libraries.standard.nodes.copy_node import CopyLibraryNode


def _build_explicit_copy_sdfg(direction: str, dtype: dtypes.typeclass = dtypes.float64) -> dace.SDFG:
    """One-state SDFG that copies a 1-element array between host and GPU via
    ``CopyLibraryNode``. ``direction`` is ``'h2d'`` or ``'d2h'``.
    """
    sdfg = dace.SDFG(f'single_elem_{direction}')
    sdfg.add_array('host', [1], dtype, storage=dtypes.StorageType.CPU_Heap)
    sdfg.add_array('dev', [1], dtype, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state('s')

    if direction == 'h2d':
        src, dst = state.add_read('host'), state.add_write('dev')
    elif direction == 'd2h':
        src, dst = state.add_read('dev'), state.add_write('host')
    else:
        raise ValueError(direction)

    # ``src_storage`` / ``dst_storage`` are derived from the surrounding edges
    # at expansion time, no need to set them on the constructor.
    cn = CopyLibraryNode(name=f'copy_{direction}')
    state.add_node(cn)
    state.add_edge(src, None, cn, '_in', dace.Memlet.from_array(src.data, sdfg.arrays[src.data]))
    state.add_edge(cn, '_out', dst, None, dace.Memlet.from_array(dst.data, sdfg.arrays[dst.data]))
    return sdfg


@pytest.mark.gpu
def test_single_element_h2d_compiles_and_copies():
    """Single-element host -> GPU copy."""
    pytest.importorskip('cupy')
    import cupy as cp
    sdfg = _build_explicit_copy_sdfg('h2d')

    host = np.array([3.14159], dtype=np.float64)
    dev = cp.zeros(1, dtype=cp.float64)

    compiled = sdfg.compile()
    compiled(host=host, dev=dev)

    np.testing.assert_allclose(cp.asnumpy(dev), host)


@pytest.mark.gpu
def test_two_element_h2d_compiles_and_copies():
    """Same direction as the failing scalar case, but with a 2-element subset --
    the codegen passes the host array by pointer, so the tasklet input binding
    works."""
    pytest.importorskip('cupy')
    import cupy as cp
    sdfg = dace.SDFG('two_elem_h2d')
    sdfg.add_array('host', [2], dtypes.float64, storage=dtypes.StorageType.CPU_Heap)
    sdfg.add_array('dev', [2], dtypes.float64, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state('s')
    src, dst = state.add_read('host'), state.add_write('dev')
    cn = CopyLibraryNode(name='copy_h2d_2')
    state.add_node(cn)
    state.add_edge(src, None, cn, '_in', dace.Memlet.from_array(src.data, sdfg.arrays[src.data]))
    state.add_edge(cn, '_out', dst, None, dace.Memlet.from_array(dst.data, sdfg.arrays[dst.data]))

    host = np.array([1.0, 2.0], dtype=np.float64)
    dev = cp.zeros(2, dtype=cp.float64)
    compiled = sdfg.compile()
    compiled(host=host, dev=dev)
    np.testing.assert_allclose(cp.asnumpy(dev), host)


@pytest.mark.gpu
def test_single_element_d2h_compiles_and_copies():
    """Single-element GPU -> host copy: must compile and round-trip the value."""
    pytest.importorskip('cupy')
    import cupy as cp
    sdfg = _build_explicit_copy_sdfg('d2h')

    dev = cp.array([2.71828], dtype=cp.float64)
    host = np.zeros(1, dtype=np.float64)

    compiled = sdfg.compile()
    compiled(host=host, dev=dev)

    np.testing.assert_allclose(host, cp.asnumpy(dev))


@pytest.mark.gpu
def test_memcpy_tasklet_connector_types_match_per_side_rule():
    """Per the per-side rule in ``_make_cuda_memcpy_expansion``: the GPU side
    is always a pointer-typed connector; a single-element CPU subset stays
    value-typed so the codegen's natural ``T x = ref`` binding works, with
    ``&_memcpy_<side>`` taking the address in the tasklet body. Multi-element
    CPU subsets are pointer-typed (parameter is already a pointer)."""
    pytest.importorskip('cupy')

    # d2h: in = GPU pointer, out = CPU value (single element).
    sdfg = _build_explicit_copy_sdfg('d2h')
    sdfg.expand_library_nodes()
    found = 0
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                for nstate in node.sdfg.states():
                    for nnode in nstate.nodes():
                        if isinstance(nnode, dace.nodes.Tasklet) and nnode.label == 'memcpy_tasklet':
                            assert isinstance(nnode.in_connectors['_memcpy_in'], dtypes.pointer), \
                                'GPU input must be pointer-typed'
                            assert not isinstance(nnode.out_connectors['_memcpy_out'], dtypes.pointer), \
                                'single-element CPU output must be value-typed'
                            assert '&_memcpy_out' in nnode.code.as_string, \
                                'CPU value-typed output must be addressed via & in the memcpy call'
                            found += 1
    assert found > 0, 'no memcpy_tasklet found in expanded d2h SDFG'


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
