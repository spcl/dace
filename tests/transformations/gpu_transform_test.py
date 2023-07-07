# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Unit tests for the GPU to-device transformation. """

import dace
import numpy as np
import pytest
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


if __name__ == '__main__':
    test_toplevel_transient_lifetime()
    test_scalar_to_symbol_in_nested_sdfg()
    test_write_subset()
    test_write_full()
    test_write_subset_dynamic()
