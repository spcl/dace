# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

from scipy import sparse


def create_structure(name: str, **members) -> dace.data.Structure:

    StructureClass = type(name, (dace.data.Structure, ), {})
    return StructureClass(members)


def test_read_structure():

    M, N, nnz = (dace.symbol(s) for s in ('M', 'N', 'nnz'))
    CSR = create_structure('CSRMatrix',
                           indptr=dace.int32[M + 1],
                           indices=dace.int32[nnz],
                           data=dace.float32[nnz],
                           rows=M,
                           cols=N,
                           nnz=nnz)

    sdfg = dace.SDFG('csr_to_dense')

    sdfg.add_datadesc('A', CSR)
    sdfg.add_array('B', [M, N], dace.float32)

    sdfg.add_view('vindptr', CSR.members['indptr'].shape, CSR.members['indptr'].dtype)
    sdfg.add_view('vindices', CSR.members['indices'].shape, CSR.members['indices'].dtype)
    sdfg.add_view('vdata', CSR.members['data'].shape, CSR.members['data'].dtype)

    state = sdfg.add_state()

    A = state.add_access('A')
    B = state.add_access('B')

    indptr = state.add_access('vindptr')
    indices = state.add_access('vindices')
    data = state.add_access('vdata')

    state.add_edge(A, 'indptr', indptr, 'views', dace.Memlet.from_array('A.indptr', CSR.members['indptr']))
    state.add_edge(A, 'indices', indices, 'views', dace.Memlet.from_array('A.indices', CSR.members['indices']))
    state.add_edge(A, 'data', data, 'views', dace.Memlet.from_array('A.data', CSR.members['data']))

    ime, imx = state.add_map('i', dict(i='0:M'))
    jme, jmx = state.add_map('idx', dict(idx='start:stop'))
    jme.add_in_connector('start')
    jme.add_in_connector('stop')
    t = state.add_tasklet('indirection', {'j', '__val'}, {'__out'}, '__out[i, j] = __val')

    state.add_memlet_path(indptr, ime, jme, memlet=dace.Memlet(data='vindptr', subset='i'), dst_conn='start')
    state.add_memlet_path(indptr, ime, jme, memlet=dace.Memlet(data='vindptr', subset='i+1'), dst_conn='stop')
    state.add_memlet_path(indices, ime, jme, t, memlet=dace.Memlet(data='vindices', subset='idx'), dst_conn='j')
    state.add_memlet_path(data, ime, jme, t, memlet=dace.Memlet(data='vdata', subset='idx'), dst_conn='__val')
    state.add_memlet_path(t, jmx, imx, B, memlet=dace.Memlet(data='B', subset='0:M, 0:N', volume=1), src_conn='__out')

    func = sdfg.compile()

    rng = np.random.default_rng(42)
    A = sparse.random(20, 20, density=0.1, format='csr', dtype=np.float32, random_state=rng)
    B = np.zeros((20, 20), dtype=np.float32)

    inpA = CSR.dtype._typeclass.as_ctypes()(indptr=A.indptr.__array_interface__['data'][0],
                                            indices=A.indices.__array_interface__['data'][0],
                                            data=A.data.__array_interface__['data'][0],
                                            rows=A.shape[0],
                                            cols=A.shape[1],
                                            M=A.shape[0],
                                            N=A.shape[1],
                                            nnz=A.nnz)

    func(A=inpA, B=B, M=20, N=20, nnz=A.nnz)
    ref = A.toarray()

    assert np.allclose(B, ref)


def test_write_structure():

    M, N, nnz = (dace.symbol(s) for s in ('M', 'N', 'nnz'))
    CSR = create_structure('CSRMatrix',
                           indptr=dace.int32[M + 1],
                           indices=dace.int32[nnz],
                           data=dace.float32[nnz],
                           rows=M,
                           cols=N,
                           nnz=nnz)

    sdfg = dace.SDFG('dense_to_csr')

    sdfg.add_array('A', [M, N], dace.float32)
    sdfg.add_datadesc('B', CSR)

    sdfg.add_view('vindptr', CSR.members['indptr'].shape, CSR.members['indptr'].dtype)
    sdfg.add_view('vindices', CSR.members['indices'].shape, CSR.members['indices'].dtype)
    sdfg.add_view('vdata', CSR.members['data'].shape, CSR.members['data'].dtype)

    # Make If
    if_before = sdfg.add_state('if_before')
    if_guard = sdfg.add_state('if_guard')
    if_body = sdfg.add_state('if_body')
    if_after = sdfg.add_state('if_after')
    sdfg.add_edge(if_before, if_guard, dace.InterstateEdge())
    sdfg.add_edge(if_guard, if_body, dace.InterstateEdge(condition='A[i, j] != 0'))
    sdfg.add_edge(if_body, if_after, dace.InterstateEdge(assignments={'idx': 'idx + 1'}))
    sdfg.add_edge(if_guard, if_after, dace.InterstateEdge(condition='A[i, j] == 0'))
    A = if_body.add_access('A')
    B = if_body.add_access('B')
    indices = if_body.add_access('vindices')
    data = if_body.add_access('vdata')
    if_body.add_edge(A, None, data, None, dace.Memlet(data='A', subset='i, j', other_subset='idx'))
    if_body.add_edge(data, 'views', B, 'data', dace.Memlet(data='B.data', subset='0:nnz'))
    t = if_body.add_tasklet('set_indices', {}, {'__out'}, '__out = j')
    if_body.add_edge(t, '__out', indices, None, dace.Memlet(data='vindices', subset='idx'))
    if_body.add_edge(indices, 'views', B, 'indices', dace.Memlet(data='B.indices', subset='0:nnz'))
    # Make For Loop  for j
    j_before, j_guard, j_after = sdfg.add_loop(None,
                                               if_before,
                                               None,
                                               'j',
                                               '0',
                                               'j < N',
                                               'j + 1',
                                               loop_end_state=if_after)
    # Make For Loop  for i
    i_before, i_guard, i_after = sdfg.add_loop(None, j_before, None, 'i', '0', 'i < M', 'i + 1', loop_end_state=j_after)
    sdfg.start_state = sdfg.node_id(i_before)
    i_before_guard = sdfg.edges_between(i_before, i_guard)[0]
    i_before_guard.data.assignments['idx'] = '0'
    B = i_guard.add_access('B')
    indptr = i_guard.add_access('vindptr')
    t = i_guard.add_tasklet('set_indptr', {}, {'__out'}, '__out = idx')
    i_guard.add_edge(t, '__out', indptr, None, dace.Memlet(data='vindptr', subset='i'))
    i_guard.add_edge(indptr, 'views', B, 'indptr', dace.Memlet(data='B.indptr', subset='0:M+1'))
    B = i_after.add_access('B')
    indptr = i_after.add_access('vindptr')
    t = i_after.add_tasklet('set_indptr', {}, {'__out'}, '__out = nnz')
    i_after.add_edge(t, '__out', indptr, None, dace.Memlet(data='vindptr', subset='M'))
    i_after.add_edge(indptr, 'views', B, 'indptr', dace.Memlet(data='B.indptr', subset='0:M+1'))

    func = sdfg.compile()

    rng = np.random.default_rng(42)
    tmp = sparse.random(20, 20, density=0.1, format='csr', dtype=np.float32, random_state=rng)
    A = tmp.toarray()
    B = tmp.tocsr(copy=True)
    B.indptr[:] = -1
    B.indices[:] = -1
    B.data[:] = -1

    outB = CSR.dtype._typeclass.as_ctypes()(indptr=B.indptr.__array_interface__['data'][0],
                                            indices=B.indices.__array_interface__['data'][0],
                                            data=B.data.__array_interface__['data'][0],
                                            rows=tmp.shape[0],
                                            cols=tmp.shape[1],
                                            M=tmp.shape[0],
                                            N=tmp.shape[1],
                                            nnz=tmp.nnz)

    func(A=A, B=outB, M=tmp.shape[0], N=tmp.shape[1], nnz=tmp.nnz)

    assert np.allclose(A, B.toarray())


def test_read_nested_structure():
    M, N, nnz = (dace.symbol(s) for s in ('M', 'N', 'nnz'))
    CSR = create_structure('CSRMatrix',
                           indptr=dace.int32[M + 1],
                           indices=dace.int32[nnz],
                           data=dace.float32[nnz],
                           rows=M,
                           cols=N,
                           nnz=nnz)
    Wrapper = create_structure('WrapperClass', csr=CSR)

    sdfg = dace.SDFG('nested_csr_to_dense')

    sdfg.add_datadesc('A', Wrapper)
    sdfg.add_array('B', [M, N], dace.float32)

    spmat = Wrapper.members['csr']
    sdfg.add_view('vindptr', spmat.members['indptr'].shape, spmat.members['indptr'].dtype)
    sdfg.add_view('vindices', spmat.members['indices'].shape, spmat.members['indices'].dtype)
    sdfg.add_view('vdata', spmat.members['data'].shape, spmat.members['data'].dtype)

    state = sdfg.add_state()

    A = state.add_access('A')
    B = state.add_access('B')

    indptr = state.add_access('vindptr')
    indices = state.add_access('vindices')
    data = state.add_access('vdata')

    state.add_edge(A, 'indptr', indptr, 'views', dace.Memlet.from_array('A.csr.indptr', spmat.members['indptr']))
    state.add_edge(A, 'indices', indices, 'views', dace.Memlet.from_array('A.csr.indices', spmat.members['indices']))
    state.add_edge(A, 'data', data, 'views', dace.Memlet.from_array('A.csr.data', spmat.members['data']))

    ime, imx = state.add_map('i', dict(i='0:M'))
    jme, jmx = state.add_map('idx', dict(idx='start:stop'))
    jme.add_in_connector('start')
    jme.add_in_connector('stop')
    t = state.add_tasklet('indirection', {'j', '__val'}, {'__out'}, '__out[i, j] = __val')

    state.add_memlet_path(indptr, ime, jme, memlet=dace.Memlet(data='vindptr', subset='i'), dst_conn='start')
    state.add_memlet_path(indptr, ime, jme, memlet=dace.Memlet(data='vindptr', subset='i+1'), dst_conn='stop')
    state.add_memlet_path(indices, ime, jme, t, memlet=dace.Memlet(data='vindices', subset='idx'), dst_conn='j')
    state.add_memlet_path(data, ime, jme, t, memlet=dace.Memlet(data='vdata', subset='idx'), dst_conn='__val')
    state.add_memlet_path(t, jmx, imx, B, memlet=dace.Memlet(data='B', subset='0:M, 0:N', volume=1), src_conn='__out')

    func = sdfg.compile()

    rng = np.random.default_rng(42)
    A = sparse.random(20, 20, density=0.1, format='csr', dtype=np.float32, random_state=rng)
    B = np.zeros((20, 20), dtype=np.float32)

    structclass = CSR.dtype._typeclass.as_ctypes()
    inpCSR = structclass(indptr=A.indptr.__array_interface__['data'][0],
                         indices=A.indices.__array_interface__['data'][0],
                         data=A.data.__array_interface__['data'][0],
                         rows=A.shape[0],
                         cols=A.shape[1],
                         M=A.shape[0],
                         K=A.shape[1],
                         nnz=A.nnz)
    import ctypes
    inpW = Wrapper.dtype._typeclass.as_ctypes()(csr=ctypes.pointer(inpCSR))

    func(A=inpW, B=B, M=20, N=20, nnz=A.nnz)
    ref = A.toarray()

    assert np.allclose(B, ref)


if __name__ == "__main__":
    test_read_structure()
    test_write_structure()
    test_read_nested_structure()
