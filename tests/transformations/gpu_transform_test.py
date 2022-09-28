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
        out = np.ndarray((10,), dtype=np.int32)
        nested_program(a, out)
        return out
    
    sdfg = main_program.to_sdfg(simplify=False)
    sdfg.apply_transformations(GPUTransformSDFG)
    out = np.empty((10,), dtype=np.int32)
    sdfg(a=4, out=out)
    assert np.array_equal(out, np.array([0, 10] * 5, dtype=np.int32))


if __name__ == '__main__':
    test_toplevel_transient_lifetime()
    test_scalar_to_symbol_in_nested_sdfg()
