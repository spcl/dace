# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest

sdfg = dace.SDFG('multistream')

_, A = sdfg.add_array('A', [2],
                      dace.float32,
                      storage=dace.StorageType.CPU_Pinned)
_, B = sdfg.add_array('B', [2],
                      dace.float32,
                      storage=dace.StorageType.CPU_Pinned)
_, C = sdfg.add_array('C', [2],
                      dace.float32,
                      storage=dace.StorageType.CPU_Pinned)

gA = sdfg.add_transient('gA', [2],
                        dace.float32,
                        storage=dace.StorageType.GPU_Global)
gB = sdfg.add_transient('gB', [2],
                        dace.float32,
                        storage=dace.StorageType.GPU_Global)
gC = sdfg.add_transient('gC', [2],
                        dace.float32,
                        storage=dace.StorageType.GPU_Global)

state = sdfg.add_state('s0')

a1 = state.add_read('A')
a2 = state.add_access('gA')

b1 = state.add_read('B')
b2 = state.add_access('gB')

c1 = state.add_access('gC')
c2 = state.add_write('C')

state.add_nedge(a1, a2, dace.Memlet.from_array('A', A))
state.add_nedge(b1, b2, dace.Memlet.from_array('B', B))
state.add_nedge(c1, c2, dace.Memlet.from_array('C', C))

state.add_nedge(a2, c1, dace.Memlet.simple('gA', '0'))
state.add_nedge(b2, c1, dace.Memlet.simple('gB', '1', other_subset_str='1'))

# Validate correctness of initial SDFG
sdfg.validate()


######################################
@pytest.mark.gpu
def test_multistream_copy():
    print('Multi-stream copy test')

    a = np.random.rand(2).astype(np.float32)
    b = np.random.rand(2).astype(np.float32)
    c = np.random.rand(2).astype(np.float32)

    sdfg(A=a, B=b, C=c)

    refC = np.array([a[0], b[1]], dtype=np.float32)
    diff = np.linalg.norm(c - refC)
    print('Difference:', diff)
    assert diff <= 1e-5


if __name__ == '__main__':
    test_multistream_copy()
