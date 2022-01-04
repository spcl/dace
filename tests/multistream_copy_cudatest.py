# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest


######################################
@pytest.mark.gpu
def test_multistream_copy():
    sdfg = dace.SDFG('multistream')

    _, A = sdfg.add_array('A', [2], dace.float32, storage=dace.StorageType.CPU_Pinned)
    _, B = sdfg.add_array('B', [2], dace.float32, storage=dace.StorageType.CPU_Pinned)
    _, C = sdfg.add_array('C', [2], dace.float32, storage=dace.StorageType.CPU_Pinned)

    gA = sdfg.add_transient('gA', [2], dace.float32, storage=dace.StorageType.GPU_Global)
    gB = sdfg.add_transient('gB', [2], dace.float32, storage=dace.StorageType.GPU_Global)
    gC = sdfg.add_transient('gC', [2], dace.float32, storage=dace.StorageType.GPU_Global)

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

    a = np.random.rand(2).astype(np.float32)
    b = np.random.rand(2).astype(np.float32)
    c = np.random.rand(2).astype(np.float32)

    sdfg(A=a, B=b, C=c)

    refC = np.array([a[0], b[1]], dtype=np.float32)
    diff = np.linalg.norm(c - refC)
    print('Difference:', diff)
    assert diff <= 1e-5


@pytest.mark.gpu
def test_copy_sync():
    sdfg = dace.SDFG('h2dsync')
    sdfg.add_scalar('scal_outer', dace.float32)
    sdfg.add_scalar('gpu_scal_outer', dace.float32, dace.StorageType.GPU_Global, transient=True)
    sdfg.add_array('output_outer', [1], dace.float32)

    nsdfg = dace.SDFG('nested')
    nsdfg.add_scalar('gpu_scal', dace.float32, dace.StorageType.GPU_Global)
    nsdfg.add_scalar('cpu_scal', dace.float32, transient=True)
    nsdfg.add_array('output', [1], dace.float32)

    nstate = nsdfg.add_state()
    r = nstate.add_read('gpu_scal')
    a = nstate.add_access('cpu_scal')
    nt = nstate.add_tasklet('addone', {'inp'}, {'out'}, 'out = inp + 1')
    w = nstate.add_write('output')
    nstate.add_nedge(r, a, dace.Memlet('gpu_scal'))
    nstate.add_edge(a, None, nt, 'inp', dace.Memlet('cpu_scal'))
    nstate.add_edge(nt, 'out', w, None, dace.Memlet('output'))

    state = sdfg.add_state()
    r = state.add_read('scal_outer')
    w = state.add_write('gpu_scal_outer')
    state.add_nedge(r, w, dace.Memlet('scal_outer'))

    state = sdfg.add_state_after(state)
    ro = state.add_read('gpu_scal_outer')
    wo = state.add_write('output_outer')
    nsdfg_node = state.add_nested_sdfg(nsdfg, None, {'gpu_scal'}, {'output'})
    state.add_edge(ro, None, nsdfg_node, 'gpu_scal', dace.Memlet('gpu_scal_outer'))
    state.add_edge(nsdfg_node, 'output', wo, None, dace.Memlet('output_outer'))

    out = np.random.rand(1).astype(np.float32)
    sdfg(scal_outer=np.float32(2), output_outer=out)
    assert np.allclose(out, 3)


if __name__ == '__main__':
    test_multistream_copy()
    test_copy_sync()
