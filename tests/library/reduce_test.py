# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest
import dace.libraries.standard as std

_params = ['pure', 'CUDA (device)', 'pure-seq', 'GPUAuto']
  


@pytest.mark.gpu
@pytest.mark.parametrize('impl', _params)
def test_multidim_gpu(impl):

    test_cases = [([1, 64, 60, 60], (0, 2, 3), [64], np.float32),
                    ([8, 512, 4096], (0,1), [4096], np.float32),
                    ([8, 512, 4096], (0,1), [4096], np.float64),
                    ([1024, 8], (0), [8], np.float32),
                    ([111, 111, 111], (0,1), [111], np.float64),
                    ([111, 111, 111], (1,2), [111], np.float64),
                    ([1000000], (0), [1], np.float64),
                    ([1111111], (0), [1], np.float64),
                    ([123,21,26,8], (1,2), [123,8], np.float32),
                    ([2, 512, 2], (0,2), [512], np.float32),
                    ([512, 555, 257], (0,2), [555], np.float64)]

    for in_shape, ax, out_shape, dtype in test_cases:
        print(in_shape, ax, out_shape, dtype)
        axes = ax
        @dace.program
        def multidimred(a, b):
            b[:] = np.sum(a, axis=axes)
        a = np.random.rand(*in_shape).astype(dtype)
        b = np.random.rand(*out_shape).astype(dtype)
        sdfg = multidimred.to_sdfg(a, b)
        sdfg.apply_gpu_transformations()
        rednode = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, std.Reduce))
        rednode.implementation = impl

        sdfg(a, b)

        assert np.allclose(b, np.sum(a, axis=axes))


if __name__ == '__main__':
    for p in _params:
        test_multidim_gpu(p)
