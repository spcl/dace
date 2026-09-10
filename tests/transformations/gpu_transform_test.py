# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Unit tests for the GPU to-device transformation. """

import dace
import numpy as np
import pytest
from dace.transformation.dataflow import GPUTransformLocalStorage
from dace.transformation.interstate import GPUTransformSDFG


def test_toplevel_transient_lifetime():
    N = dace.symbol('N')

    @dace.program
    def program(A: dace.float64[20, 20]):
        for i in range(20):
            tmp = A[:i, :i]
            tmp2 = A[:5, :N]
            tmp *= 5
            tmp2 *= 10

    sdfg = program.to_sdfg()
    sdfg.apply_transformations(GPUTransformSDFG, options=dict(toplevel_trans=True))

    for name, desc in sdfg.arrays.items():
        if name == 'tmp2' and type(desc) is dace.data.Array:
            assert desc.lifetime is dace.AllocationLifetime.SDFG
        else:
            assert desc.lifetime is not dace.AllocationLifetime.SDFG


@pytest.mark.gpu
def test_scalar_to_symbol_in_nested_sdfg():
    """
    GPUTransformSDFG will automatically create copy-out states for GPU scalars that are used in host-side interstate
    edges. However, this process may only be applied in top-level SDFGs and not in NestedSDFGs that have GPU-device
    schedule but are not part of a single GPU kernel, leading to illegal memory accesses.
    """

    @dace.program
    def nested_program(a: dace.int32, out: dace.int32[10]):
        for i in range(10):
            if a < 5:
                out[i] = 0
                a *= 2
            else:
                out[i] = 10
                a /= 2

    @dace.program
    def main_program(a: dace.int32):
        out = np.ndarray((10, ), dtype=np.int32)
        nested_program(a, out)
        return out

    sdfg = main_program.to_sdfg(simplify=False)
    sdfg.apply_transformations(GPUTransformSDFG)
    out = sdfg(a=4)
    assert np.array_equal(out, np.array([0, 10] * 5, dtype=np.int32))


@pytest.mark.gpu
def test_write_subset():

    @dace.program
    def write_subset(A: dace.int32[20, 20]):
        for i, j in dace.map[2:18, 2:18]:
            A[i, j] = i + j

    sdfg = write_subset.to_sdfg(simplify=True)
    sdfg.apply_transformations(GPUTransformSDFG)

    ref = np.ones((20, 20), dtype=np.int32)
    val = np.copy(ref)

    write_subset.f(ref)
    sdfg(A=val)

    assert np.array_equal(ref, val)


def test_write_full():

    M, N = dace.symbol('M'), dace.symbol('N')

    @dace.program
    def write_full(A: dace.int32[M, N]):
        for i, j in dace.map[0:M, 0:N]:
            A[i, j] = i + j

    sdfg = write_full.to_sdfg(simplify=True)
    sdfg.apply_transformations(GPUTransformSDFG)

    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.AccessNode) and node.data == 'A':
                assert state.out_degree(node) == 0


@pytest.mark.gpu
def test_write_subset_dynamic():

    @dace.program
    def write_subset_dynamic(A: dace.int32[20, 20], x: dace.int32[20], y: dace.int32[20]):
        for i, j in dace.map[2:18, 2:18]:
            A[x[i], y[j]] = i + j

    sdfg = write_subset_dynamic.to_sdfg(simplify=True)
    sdfg.apply_transformations(GPUTransformSDFG)

    ref = np.ones((20, 20), dtype=np.int32)
    val = np.copy(ref)

    x = np.random.permutation(20).astype(np.int32)
    y = np.random.permutation(20).astype(np.int32)

    write_subset_dynamic.f(ref, x, y)
    sdfg(A=val, x=x, y=y)

    assert np.array_equal(ref, val)


@pytest.mark.parametrize(["transient", "scalar"], [[False, False], [False, True], [True, False], [True, True]])
def test_free_tasklet(transient, scalar):
    sdfg = dace.SDFG("assign")

    state = sdfg.add_state("main")
    if scalar:
        arr_name, arr = sdfg.add_scalar("A", dace.float32, transient=transient)
    else:
        arr_name, arr = sdfg.add_array("A", (4, ), dace.float32, transient=transient)

    an = state.add_access(arr_name)

    t = state.add_tasklet("assign", {}, {"_out"}, "_out = 2.0")
    state.add_edge(t, "_out", an, None, dace.memlet.Memlet("A" if scalar else "A[0]"))

    sdfg.validate()

    sdfg.apply_gpu_transformations(validate=True,
                                   validate_all=True,
                                   permissive=True,
                                   sequential_innermaps=True,
                                   register_transients=False,
                                   simplify=False)

    sdfg.validate()


def _row_doubling_body(shape):
    """``b[0, j] = 2 * a[0, j]``, with both connectors describing the whole container."""
    sdfg = dace.SDFG('body')
    sdfg.add_array('a', shape, dace.float64)
    sdfg.add_array('b', shape, dace.float64)
    sdfg.add_symbol('j', dace.int64)
    state = sdfg.add_state()
    tasklet = state.add_tasklet('t', {'x'}, {'y'}, 'y = x * 2')
    state.add_edge(state.add_read('a'), None, tasklet, 'x', dace.Memlet('a[0, j]'))
    state.add_edge(tasklet, 'y', state.add_write('b'), None, dace.Memlet('b[0, j]'))
    return sdfg


def test_gpu_local_storage_of_a_nested_sdfg_row():
    """The copy on the device holds one row, so the connectors describe a row rather than a matrix.

    ``GPUTransformLocalStorage`` copies only the part of each array the map reads, dropping the
    dimensions the copy is a single index of. Under the nested SDFG contract (see
    ``dace.sdfg.dealias.integrate_nested_sdfg``) a connector is the container it is connected to,
    so the connectors below have to be restated the same way.
    """
    shape = (4, 5)
    sdfg = dace.SDFG('gpu_local_storage_nested')
    sdfg.add_array('A', shape, dace.float64)
    sdfg.add_array('B', shape, dace.float64)
    state = sdfg.add_state()
    entry, exit_ = state.add_map('m', dict(j='0:%d' % shape[1]))
    node = state.add_nested_sdfg(_row_doubling_body(shape), {'a'}, {'b'}, {'j': 'j'})
    state.add_memlet_path(state.add_read('A'), entry, node, dst_conn='a', memlet=dace.Memlet('A[0, j]'))
    state.add_memlet_path(node, exit_, state.add_write('B'), src_conn='b', memlet=dace.Memlet('B[0, j]'))
    sdfg.validate()

    assert sdfg.apply_transformations(GPUTransformLocalStorage) == 1

    for edge in state.all_edges(node):
        connector = edge.dst_conn if edge.dst is node else edge.src_conn
        assert len(node.sdfg.arrays[connector].shape) == 1
        assert node.sdfg.arrays[connector].is_equivalent(sdfg.arrays[edge.data.data])
    sdfg.validate()


if __name__ == '__main__':
    test_toplevel_transient_lifetime()
    test_scalar_to_symbol_in_nested_sdfg()
    test_write_subset()
    test_write_full()
    test_write_subset_dynamic()
    test_gpu_local_storage_of_a_nested_sdfg_row()
    for scalar in [False, True]:
        for transient in [False, True]:
            test_free_tasklet(transient, scalar)
