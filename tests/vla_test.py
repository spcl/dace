# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Tries to allocate a symbolically-sized array on a register and makes
    sure that it is allocated on the heap instead.
"""
import dace
import numpy as np
import warnings

N = dace.symbol('N')

sdfg = dace.SDFG('vla_test')
sdfg.add_array('A', [N], dace.float32)
sdfg.add_transient('tmp', [N], dace.float32, storage=dace.StorageType.Register)
sdfg.add_array('B', [N], dace.float32)
state = sdfg.add_state()
A = state.add_read('A')
tmp = state.add_access('tmp')
B = state.add_write('B')

state.add_nedge(A, tmp, dace.Memlet.simple('A', '0:N'))
state.add_nedge(tmp, B, dace.Memlet.simple('tmp', '0:N'))


def test():
    A = np.random.rand(12).astype(np.float32)
    B = np.random.rand(12).astype(np.float32)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sdfg(A=A, B=B, N=12)

        assert w
        assert any('Variable-length' in str(warn.message) for warn in w)

    diff = np.linalg.norm(A - B)
    print('Difference:', diff)
    assert diff < 1e-5


if __name__ == "__main__":
    test()
