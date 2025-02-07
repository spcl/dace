# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest

sdfg = dace.SDFG('multistream_kernel')

sdfg.add_array('A', [2], dace.float32, storage=dace.StorageType.CPU_Pinned)
sdfg.add_array('B', [2], dace.float32, storage=dace.StorageType.CPU_Pinned)
sdfg.add_array('C', [2], dace.float32, storage=dace.StorageType.CPU_Pinned)

sdfg.add_transient('gA1', [2], dace.float32, storage=dace.StorageType.GPU_Global)
sdfg.add_transient('gA2', [2], dace.float32, storage=dace.StorageType.GPU_Global)
sdfg.add_transient('gB1', [2], dace.float32, storage=dace.StorageType.GPU_Global)
sdfg.add_transient('gB2', [2], dace.float32, storage=dace.StorageType.GPU_Global)
sdfg.add_transient('gC', [2], dace.float32, storage=dace.StorageType.GPU_Global)

state = sdfg.add_state('s0')

a = state.add_read('A')
ga1 = state.add_access('gA1')
ga2 = state.add_access('gA2')
state.add_nedge(a, ga1, dace.Memlet.simple('A', '0:2'))

b = state.add_read('B')
gb1 = state.add_access('gB1')
gb2 = state.add_access('gB2')
state.add_nedge(b, gb1, dace.Memlet.simple('B', '0:2'))

gc = state.add_access('gC')
c = state.add_write('C')
state.add_nedge(gc, c, dace.Memlet.simple('gC', '0:2'))

t1, me1, mx1 = state.add_mapped_tasklet('addone', dict(i='0:2'),
                                        dict(inp=dace.Memlet.simple('gA1', 'i')), 'out = inp + 1',
                                        dict(out=dace.Memlet.simple('gA2', 'i')), dace.ScheduleType.GPU_Device)
t2, me2, mx2 = state.add_mapped_tasklet('addtwo', dict(i='0:2'),
                                        dict(inp=dace.Memlet.simple('gB1', 'i')), 'out = inp + 2',
                                        dict(out=dace.Memlet.simple('gB2', 'i')), dace.ScheduleType.GPU_Device)

t2, me3, mx3 = state.add_mapped_tasklet('twoarrays', dict(i='0:2'),
                                        dict(inp1=dace.Memlet.simple('gA2', 'i'),
                                             inp2=dace.Memlet.simple('gB2', 'i')), 'out = inp1 * inp2',
                                        dict(out=dace.Memlet.simple('gC', 'i')), dace.ScheduleType.GPU_Device)

state.add_nedge(ga1, me1, dace.Memlet.simple('gA1', '0:2'))
state.add_nedge(gb1, me2, dace.Memlet.simple('gB1', '0:2'))
state.add_nedge(mx1, ga2, dace.Memlet.simple('gA2', '0:2'))
state.add_nedge(mx2, gb2, dace.Memlet.simple('gB2', '0:2'))

state.add_nedge(ga2, me3, dace.Memlet.simple('gA2', '0:2'))
state.add_nedge(gb2, me3, dace.Memlet.simple('gB2', '0:2'))
state.add_nedge(mx3, gc, dace.Memlet.simple('gC', '0:2'))

sdfg.fill_scope_connectors()

# Validate correctness of initial SDFG
sdfg.validate()


######################################
@pytest.mark.gpu
def test_multistream_kernel():
    print('Multi-stream kernel test')

    a = np.random.rand(2).astype(np.float32)
    b = np.random.rand(2).astype(np.float32)
    c = np.random.rand(2).astype(np.float32)

    sdfg(A=a, B=b, C=c)

    refC = (a + 1) * (b + 2)
    diff = np.linalg.norm(c - refC)
    print('Difference:', diff)
    assert diff <= 1e-5


if __name__ == "__main__":
    test_multistream_kernel()
