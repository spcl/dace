# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest

from scipy import sparse


def test_read_csr_tensor():

    M, N, nnz = (dace.symbol(s) for s in ('M', 'N', 'nnz'))
    csr_obj = dace.data.Tensor(dace.float32, (M, N), [(dace.data.TensorIndexDense(), 0),
                                                      (dace.data.TensorIndexCompressed(), 1)], nnz, "CSR_Tensor")

    sdfg = dace.SDFG('tensor_csr_to_dense')

    sdfg.add_datadesc('A', csr_obj)
    sdfg.add_array('B', [M, N], dace.float32)

    sdfg.add_view('vindptr', csr_obj.members['idx1_pos'].shape, csr_obj.members['idx1_pos'].dtype)
    sdfg.add_view('vindices', csr_obj.members['idx1_crd'].shape, csr_obj.members['idx1_crd'].dtype)
    sdfg.add_view('vdata', csr_obj.members['values'].shape, csr_obj.members['values'].dtype)

    state = sdfg.add_state()

    A = state.add_access('A')
    B = state.add_access('B')

    indptr = state.add_access('vindptr')
    indices = state.add_access('vindices')
    data = state.add_access('vdata')

    state.add_edge(A, None, indptr, 'views', dace.Memlet.from_array('A.idx1_pos', csr_obj.members['idx1_pos']))
    state.add_edge(A, None, indices, 'views', dace.Memlet.from_array('A.idx1_crd', csr_obj.members['idx1_crd']))
    state.add_edge(A, None, data, 'views', dace.Memlet.from_array('A.values', csr_obj.members['values']))

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

    inpA = csr_obj.dtype._typeclass.as_ctypes()(idx1_pos=A.indptr.__array_interface__['data'][0],
                                                idx1_crd=A.indices.__array_interface__['data'][0],
                                                values=A.data.__array_interface__['data'][0])

    func(A=inpA, B=B, M=A.shape[0], N=A.shape[1], nnz=A.nnz)
    ref = A.toarray()

    assert np.allclose(B, ref)


def test_csr_fields():

    M, N, nnz = (dace.symbol(s) for s in ('M', 'N', 'nnz'))

    csr = dace.data.Tensor(
        dace.float32,
        (M, N),
        [(dace.data.TensorIndexDense(), 0), (dace.data.TensorIndexCompressed(), 1)],
        nnz,
        "CSR_Matrix",
    )

    expected_fields = ["idx1_pos", "idx1_crd"]
    assert all(key in csr.members.keys() for key in expected_fields)


def test_dia_fields():

    M, N, nnz, num_diags = (dace.symbol(s) for s in ('M', 'N', 'nnz', 'num_diags'))

    diag = dace.data.Tensor(
        dace.float32,
        (M, N),
        [
            (dace.data.TensorIndexDense(), num_diags),
            (dace.data.TensorIndexRange(), 0),
            (dace.data.TensorIndexOffset(), 1),
        ],
        nnz,
        "DIA_Matrix",
    )

    expected_fields = ["idx1_offset", "idx2_offset"]
    assert all(key in diag.members.keys() for key in expected_fields)


def test_coo_fields():

    I, J, K, nnz = (dace.symbol(s) for s in ('I', 'J', 'K', 'nnz'))

    coo = dace.data.Tensor(
        dace.float32,
        (I, J, K),
        [
            (dace.data.TensorIndexCompressed(unique=False), 0),
            (dace.data.TensorIndexSingleton(unique=False), 1),
            (dace.data.TensorIndexSingleton(), 2),
        ],
        nnz,
        "COO_3D_Tensor",
    )

    expected_fields = ["idx0_pos", "idx0_crd", "idx1_crd", "idx2_crd"]
    assert all(key in coo.members.keys() for key in expected_fields)


if __name__ == "__main__":
    test_read_csr_tensor()
    test_csr_fields()
    test_dia_fields()
    test_coo_fields()
