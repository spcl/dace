# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest
import dace.libraries.standard as std

_params = ['pure', 'CUDA (device)', 'pure-seq']


@pytest.mark.gpu
@pytest.mark.parametrize('impl', _params)
def test_multidim_gpu(impl):
    @dace.program
    def multidimred(a, b):
        b[:] = np.sum(a, axis=(0, 2, 3))

    a = np.random.rand(1, 64, 60, 60).astype(np.float32)
    b = np.random.rand(1, 64).astype(np.float32)
    sdfg = multidimred.to_sdfg(a, b)
    sdfg.apply_gpu_transformations()
    rednode = next(n for n, _ in sdfg.all_nodes_recursive()
                   if isinstance(n, std.Reduce))
    rednode.implementation = impl

    sdfg(a, b)

    assert np.allclose(b, np.sum(a, axis=(0, 2, 3)))


if __name__ == '__main__':
    for p in _params:
        test_multidim_gpu(p)
