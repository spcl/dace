import pytest
import dace
import numpy as np

from dace.libraries.sparse import TensorIndexNotation
from dace.transformation.interstate.sdfg_nesting import InlineSDFG
from dace.data import Tensor, TensorIndexCompressed
from scipy import sparse


def csr(M, N, nnz):
    return Tensor(
        dace.float32,
        (M, N),
        [(dace.data.TensorIndexDense(), 0), (dace.data.TensorIndexCompressed(), 1)],
        nnz,
        "CSR_Tensor")


def csr_data(rng, M, N):
    desc = csr(M, N, 1)

    m = sparse.random(M, N, density=0.1, format='csr', dtype=np.float32, random_state=rng)

    sparse_m = desc.dtype._typeclass.as_ctypes()(
        order=2,
        dim_sizes=0,
        values=m.data.__array_interface__['data'][0],
        idx1_pos=m.indptr.__array_interface__['data'][0],
        idx1_crd=m.indices.__array_interface__['data'][0],
    ) 

    return (m.todense(), sparse_m, m)


def csc(M, N, nnz):
    return Tensor(
        dace.float32,
        (M, N),
        [(dace.data.TensorIndexDense(), 1), (dace.data.TensorIndexCompressed(), 0)],
        nnz,
        "CSC_Tensor")


def csc_data(rng, M, N):
    desc = csc(M, N, 1)

    m = sparse.random(M, N, density=0.1, format='csc', dtype=np.float32, random_state=rng)

    sparse_m = desc.dtype._typeclass.as_ctypes()(
        order=2,
        dim_sizes=0,
        values=m.data.__array_interface__['data'][0],
        idx1_pos=m.indptr.__array_interface__['data'][0],
        idx1_crd=m.indices.__array_interface__['data'][0],
    ) 

    return (m.todense(), sparse_m, m)


def gen_data(rng, M, N, format):
    if format == 'dense':
        m = rng.random((M, N), dtype=np.float32)
        return (m, m, m)
    elif format == 'csr':
        return csr_data(rng, M, N)
    elif format == 'csc':
        return csc_data(rng, M, N)
    else:
        assert False



def build_mm(data: dict[str, str]):

    M, K, N, nnz = (dace.symbol(s) for s in ('M', 'K', 'N', 'nnz'))

    sdfg = dace.SDFG(f"tin_mm_{data['B']}_{data['C']}_{data['A']}_test")

    dims = {'B': (M, K), 'C': (K, N), 'A': (M, N)}

    for name, format in data.items():
        if format == 'dense':
            sdfg.add_array(name, dims[name], dace.float32)
        elif format == 'csr':
            sdfg.add_datadesc(name, csr(dims[name][0], dims[name][1], nnz))
        elif format == 'csc':
            sdfg.add_datadesc(name, csc(dims[name][0], dims[name][1], nnz))
        else:
            assert False

    state = sdfg.add_state()

    A = state.add_access('A')
    B = state.add_access('B')
    C = state.add_access('C')

    tin_node = TensorIndexNotation("test", "A(i,j) = B(i,k) * C(k,j)")
    tin_node.add_in_connector('tin_B')
    tin_node.add_in_connector('tin_C')
    tin_node.add_out_connector('tin_A') 

    state.add_node(tin_node)

    state.add_edge(B, None, tin_node, 'tin_B', memlet=dace.Memlet(data='B'))
    state.add_edge(C, None, tin_node, 'tin_C', memlet=dace.Memlet(data='C'))
    state.add_edge(tin_node, "tin_A", A, None, memlet=dace.Memlet(data='A'))

    sdfg.expand_library_nodes(recursive=True)
    sdfg.apply_transformations_repeated(InlineSDFG)

    func = sdfg.compile()

    return func


def test_mm(data: dict[str, str]):
    sizes = [
        (20, 20, 20),
        (40, 20, 30),
        (30, 40, 20),
    ]

    func = build_mm(data)

    rng = np.random.default_rng(42)

    for M, K, N in sizes:
        print(f"Tesing {data['B']}({M}, {K}) x {data['C']}({K}, {N}) -> {data['A']}({M}, {N})")

        _, AA, A_keep_alive = gen_data(rng, M, N, data['A'])
        B, BB, B_keep_alive = gen_data(rng, M, K, data['B'])
        C, CC, C_keep_alive = gen_data(rng, K, N, data['C'])

        # print(f"{AA=}")
        # print(f"{B=}")
        # print(f"{C=}")

        func(A=AA, B=BB, C=CC, M=M, K=K, N=N, nnz=1)

        A = B @ C
    
        if not np.allclose(A, AA):
            print(f"{A=}")
            print(f"{AA=}")

        assert np.allclose(AA, A)
        print(f"SUCCESS")




def test_basic_csr():
    M, K, N, nnz = (dace.symbol(s) for s in ('M', 'K', 'N', 'nnz'))
    csr_obj = dace.data.Tensor(
        dace.float32,
        (M, K),
        [(dace.data.TensorIndexDense(), 0), (dace.data.TensorIndexCompressed(), 1)],
        nnz,
        "CSR_Tensor")

    sdfg = dace.SDFG('tensor_index_notation_csr_test')

    sdfg.add_datadesc('B', csr_obj)
    sdfg.add_array('C', (K, N), dace.float32)
    sdfg.add_array('A', (M, N), dace.float32)

    state = sdfg.add_state()

    A = state.add_access('A')
    B = state.add_access('B')
    C = state.add_access('C')

    # tin_node = TensorIndexNotation("test", "A(i,l) = B(i,j,k) * C(j,l) * D(k,l)")
    tin_node = TensorIndexNotation("test", "A(i,j) = B(i,k) * C(k,j)")
    tin_node.add_in_connector('tin_B')
    tin_node.add_in_connector('tin_C')
    tin_node.add_out_connector('tin_A')

    state.add_node(tin_node)

    state.add_edge(B, None, tin_node, 'tin_B', memlet=dace.Memlet(data='B'))
    state.add_edge(C, None, tin_node, 'tin_C', memlet=dace.Memlet(data='C'))
    state.add_edge(tin_node, "tin_A", A, None, memlet=dace.Memlet(data='A'))

    sdfg.expand_library_nodes(recursive=True)
    sdfg.apply_transformations_repeated(InlineSDFG)

    func = sdfg.compile()

    rng = np.random.default_rng(42)
    B = sparse.random(20, 20, density=0.1, format='csr', dtype=np.float32, random_state=rng)
    C = rng.random((20, 20), dtype=np.float32)
    A = np.zeros((20, 20), dtype=np.float32)

    inpB = csr_obj.dtype._typeclass.as_ctypes()(
        order=2,
        dim_sizes=0,
        values=B.data.__array_interface__['data'][0],
        idx1_pos=B.indptr.__array_interface__['data'][0],
        idx1_crd=B.indices.__array_interface__['data'][0],
    )

    func(A=A, B=inpB, C=C, N=20, M=20, K=20, nnz=B.nnz)
    ref = B.dot(C)

    assert np.allclose(A, ref)
    print("SUCCESS")


def test_basic_csc():
    M, K, N, nnz = (dace.symbol(s) for s in ('M', 'K', 'N', 'nnz'))
    csc_obj = dace.data.Tensor(
        dace.float32,
        (K, N),
        [(dace.data.TensorIndexDense(), 1), (dace.data.TensorIndexCompressed(), 0)],
        nnz,
        "CSC_Tensor")

    sdfg = dace.SDFG('tensor_index_notation_csc_test')

    sdfg.add_array('B', [M, K], dace.float32)
    sdfg.add_datadesc('C', csc_obj)
    sdfg.add_array('A', [M, N], dace.float32)

    state = sdfg.add_state()

    A = state.add_access('A')
    B = state.add_access('B')
    C = state.add_access('C')

    # tin_node = TensorIndexNotation("test", "A(i,l) = B(i,j,k) * C(j,l) * D(k,l)")
    tin_node = TensorIndexNotation("test", "A(i,j) = B(i,k) * C(k,j)")
    tin_node.add_in_connector('tin_B')
    tin_node.add_in_connector('tin_C')
    tin_node.add_out_connector('tin_A')

    state.add_node(tin_node)

    state.add_edge(B, None, tin_node, 'tin_B', memlet=dace.Memlet(data='B'))
    state.add_edge(C, None, tin_node, 'tin_C', memlet=dace.Memlet(data='C'))
    state.add_edge(tin_node, "tin_A", A, None, memlet=dace.Memlet(data='A'))

    sdfg.expand_library_nodes(recursive=True)
    sdfg.apply_transformations_repeated(InlineSDFG)

    func = sdfg.compile()

    rng = np.random.default_rng(42)
    B = rng.random((20, 20), dtype=np.float32)
    C = sparse.random(20, 20, density=0.1, format='csc', dtype=np.float32, random_state=rng)
    A = np.zeros((20, 20), dtype=np.float32)

    inpC = csc_obj.dtype._typeclass.as_ctypes()(
        order=2,
        dim_sizes=0,
        values=C.data.__array_interface__['data'][0],
        idx1_pos=C.indptr.__array_interface__['data'][0],
        idx1_crd=C.indices.__array_interface__['data'][0],
    )

    func(A=A, B=B, C=inpC, N=20, M=20, K=20, nnz=C.nnz)
    ref = sparse.csr_matrix.dot(B, C)

    assert np.allclose(A, ref)
    print("SUCCESS")


def test_multiple_mm_tranform():
    M, K, N, nnz = (dace.symbol(s) for s in ('M', 'K', 'N', 'nnz'))

    sdfg = dace.SDFG('tin_multiple_mm_test')

    sdfg.add_array('B', [N, N], dace.float32)
    sdfg.add_array('C', [N, N], dace.float32)
    sdfg.add_array('D', [N, N], dace.float32)
    sdfg.add_array('E', [N, N], dace.float32)
    sdfg.add_array('BC', [N, N], dace.float32, transient=True)
    sdfg.add_array('DE', [N, N], dace.float32, transient=True)
    sdfg.add_array('A', [N, N], dace.float32)

    state = sdfg.add_state()

    A = state.add_access('A')
    B = state.add_access('B')
    C = state.add_access('C')
    D = state.add_access('D')
    E = state.add_access('E')
    BC = state.add_access('BC')
    DE = state.add_access('DE')

    tin_node1 = TensorIndexNotation("mm1", "BC(i,j) = B(i,k) * C(k,j)")
    tin_node1.add_in_connector('tin_B')
    tin_node1.add_in_connector('tin_C')
    tin_node1.add_out_connector('tin_BC')
    
    tin_node2 = TensorIndexNotation("mm2", "DE(i,j) = D(i,k) * E(k,j)")
    tin_node2.add_in_connector('tin_D')
    tin_node2.add_in_connector('tin_E')
    tin_node2.add_out_connector('tin_DE')

    tin_node3 = TensorIndexNotation("mm3", "A(i,j) = BC(i,k) * DE(k,j)")
    tin_node3.add_in_connector('tin_BC')
    tin_node3.add_in_connector('tin_DE')
    tin_node3.add_out_connector('tin_A')

    state.add_node(tin_node1)
    state.add_node(tin_node2)
    state.add_node(tin_node3)

    state.add_edge(B, None, tin_node1, 'tin_B', memlet=dace.Memlet(data='B'))
    state.add_edge(C, None, tin_node1, 'tin_C', memlet=dace.Memlet(data='C'))
    state.add_edge(tin_node1, "tin_BC", BC, None, memlet=dace.Memlet(data='BC'))

    state.add_edge(D, None, tin_node2, 'tin_D', memlet=dace.Memlet(data='D'))
    state.add_edge(E, None, tin_node2, 'tin_E', memlet=dace.Memlet(data='E'))
    state.add_edge(tin_node2, "tin_DE", DE, None, memlet=dace.Memlet(data='DE'))

    state.add_edge(BC, None, tin_node3, 'tin_BC', memlet=dace.Memlet(data='BC'))
    state.add_edge(DE, None, tin_node3, 'tin_DE', memlet=dace.Memlet(data='DE'))
    state.add_edge(tin_node3, "tin_A", A, None, memlet=dace.Memlet(data='A'))

    print(f"DEBUG {res=}")

    sdfg.save("sdfg.sdfg")

    sdfg.expand_library_nodes(recursive=True)
    sdfg.apply_transformations_repeated(InlineSDFG)

    func = sdfg.compile()

    rng = np.random.default_rng(42)
    B = rng.random((20, 20), dtype=np.float32)
    C = rng.random((20, 20), dtype=np.float32)
    D = rng.random((20, 20), dtype=np.float32)
    E = rng.random((20, 20), dtype=np.float32)
    A = np.zeros((20, 20), dtype=np.float32)

    func(A=A, B=B, C=C, D=D, E=E, N=20)
    ref = (B @ C) @ (D @ E)

    assert np.allclose(A, ref)
    print("SUCCESS")


def test_multiple_mm_fix():
    M, K, N, nnz = (dace.symbol(s) for s in ('M', 'K', 'N', 'nnz'))

    csf_obj = Tensor(
        dace.float32,
        (N, N),
        [(TensorIndexCompressed(), 0), (TensorIndexCompressed(), 1)],
        nnz,
        "CSF_Tensor",
        transient=True)

    sdfg = dace.SDFG('tin_multiple_mm_fix_test')

    sdfg.add_array('B', [N, N], dace.float32)
    sdfg.add_array('C', [N, N], dace.float32)
    sdfg.add_array('D', [N, N], dace.float32)
    sdfg.add_array('E', [N, N], dace.float32)
    # sdfg.add_array('BC', [N, N], dace.float32, transient=True)
    sdfg.add_datadesc('BC', csf_obj)
    sdfg.add_array('DE', [N, N], dace.float32, transient=True)
    sdfg.add_array('A', [N, N], dace.float32)

    state = sdfg.add_state()

    A = state.add_access('A')
    B = state.add_access('B')
    C = state.add_access('C')
    D = state.add_access('D')
    E = state.add_access('E')
    BC = state.add_access('BC')
    DE = state.add_access('DE')

    tin_node1 = TensorIndexNotation("mm1", "BC(i,j) = B(i,k) * C(k,j)")
    tin_node1.add_in_connector('tin_B')
    tin_node1.add_in_connector('tin_C')
    tin_node1.add_out_connector('tin_BC')
    
    tin_node2 = TensorIndexNotation("mm2", "DE(i,j) = D(i,k) * E(k,j)")
    tin_node2.add_in_connector('tin_D')
    tin_node2.add_in_connector('tin_E')
    tin_node2.add_out_connector('tin_DE')

    tin_node3 = TensorIndexNotation("mm3", "A(i,j) = BC(i,k) * DE(k,j)")
    tin_node3.add_in_connector('tin_BC')
    tin_node3.add_in_connector('tin_DE')
    tin_node3.add_out_connector('tin_A')

    state.add_node(tin_node1)
    state.add_node(tin_node2)
    state.add_node(tin_node3)

    state.add_edge(B, None, tin_node1, 'tin_B', memlet=dace.Memlet(data='B'))
    state.add_edge(C, None, tin_node1, 'tin_C', memlet=dace.Memlet(data='C'))
    state.add_edge(tin_node1, "tin_BC", BC, None, memlet=dace.Memlet(data='BC'))

    state.add_edge(D, None, tin_node2, 'tin_D', memlet=dace.Memlet(data='D'))
    state.add_edge(E, None, tin_node2, 'tin_E', memlet=dace.Memlet(data='E'))
    state.add_edge(tin_node2, "tin_DE", DE, None, memlet=dace.Memlet(data='DE'))

    state.add_edge(BC, None, tin_node3, 'tin_BC', memlet=dace.Memlet(data='BC'))
    state.add_edge(DE, None, tin_node3, 'tin_DE', memlet=dace.Memlet(data='DE'))
    state.add_edge(tin_node3, "tin_A", A, None, memlet=dace.Memlet(data='A'))

    sdfg.expand_library_nodes(recursive=True)
    sdfg.apply_transformations_repeated(InlineSDFG)

    func = sdfg.compile()

    rng = np.random.default_rng(42)
    B = rng.random((20, 20), dtype=np.float32)
    C = rng.random((20, 20), dtype=np.float32)
    D = rng.random((20, 20), dtype=np.float32)
    E = rng.random((20, 20), dtype=np.float32)
    A = np.zeros((20, 20), dtype=np.float32)

    func(A=A, B=B, C=C, D=D, E=E, N=20)
    ref = (B @ C) @ (D @ E)

    assert np.allclose(A, ref)
    print("SUCCESS")


def test_mttkrp():
    M, K, L, N, nnz = (dace.symbol(s) for s in ('M', 'K', 'L', 'N', 'nnz'))

    sdfg = dace.SDFG('tin_mttkrp_test')

    sdfg.add_array('A', [M, N], dace.float32)
    sdfg.add_array('B', [M, K, L], dace.float32)
    sdfg.add_array('C', [K, N], dace.float32)
    sdfg.add_array('D', [L, N], dace.float32)

    state = sdfg.add_state()

    A = state.add_access('A')
    B = state.add_access('B')
    C = state.add_access('C')
    D = state.add_access('D')

    tin_node = TensorIndexNotation("mttkrp", "A(i,j) = B(i,k,l) * D(l,j) * C(k,j)")
    tin_node.add_in_connector('tin_B')
    tin_node.add_in_connector('tin_C')
    tin_node.add_in_connector('tin_D')
    tin_node.add_out_connector('tin_A')
    
    state.add_node(tin_node)

    state.add_edge(B, None, tin_node, 'tin_B', memlet=dace.Memlet(data='B'))
    state.add_edge(C, None, tin_node, 'tin_C', memlet=dace.Memlet(data='C'))
    state.add_edge(D, None, tin_node, 'tin_D', memlet=dace.Memlet(data='D'))
    state.add_edge(tin_node, "tin_A", A, None, memlet=dace.Memlet(data='A'))

    sdfg.expand_library_nodes(recursive=True)
    sdfg.apply_transformations_repeated(InlineSDFG)

    func = sdfg.compile()

    rng = np.random.default_rng(42)
    B = rng.random((20, 20, 20), dtype=np.float32)
    C = rng.random((20, 20), dtype=np.float32)
    D = rng.random((20, 20), dtype=np.float32)
    A = np.zeros((20, 20), dtype=np.float32)

    func(A=A, B=B, C=C, D=D, M=20, K=20, L=20, N=20)
    # ref = (B @ C) @ (D @ E)

    # assert np.allclose(A, ref)
    print("SUCCESS")


if __name__ == "__main__":
    # test_basic_csr()
    # test_basic_csc()
    # test_mm({'B': 'dense', 'C': 'dense', 'A': 'dense'})
    # test_mm({'B': 'csr', 'C': 'dense', 'A': 'dense'})
    # test_mm({'B': 'dense', 'C': 'csc', 'A': 'dense'})
    # test_mm({'B': 'csr', 'C': 'csc', 'A': 'dense'})
    # test_mm({'B': 'csr', 'C': 'csc', 'A': 'csr'})
    # test_mm({'B': 'csr', 'C': 'csc', 'A': 'csc'})
    test_mm({'B': 'csc', 'C': 'csr', 'A': 'dense'})
    # test_mm({'B': 'csc', 'C': 'csr', 'A': 'csr'})
    # test_mm({'B': 'csc', 'C': 'csr', 'A': 'csc'})
    # test_multiple_mm_fix()
    # test_multiple_mm_tranform()
    # test_mttkrp()

 