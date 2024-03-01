import pytest
import ctypes
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
        "CSR_Tensor",
    )


def csr_data(rng, M, N):
    desc = csr(M, N, 1)

    m = sparse.random(
        M, N, density=0.5, format="csr", dtype=np.float32, random_state=rng
    )

    sparse_m = desc.dtype._typeclass.as_ctypes()(
        order=2,
        dim_sizes=0,
        values=m.data.__array_interface__["data"][0],
        idx1_pos=m.indptr.__array_interface__["data"][0],
        idx1_crd=m.indices.__array_interface__["data"][0],
    )

    return (m.todense(), sparse_m, m)


def csc(M, N, nnz):
    return Tensor(
        dace.float32,
        (M, N),
        [(dace.data.TensorIndexDense(), 1), (dace.data.TensorIndexCompressed(), 0)],
        nnz,
        "CSC_Tensor",
    )


def csc_data(rng, M, N):
    desc = csc(M, N, 1)

    m = sparse.random(
        M, N, density=0.5, format="csc", dtype=np.float32, random_state=rng
    )

    sparse_m = desc.dtype._typeclass.as_ctypes()(
        order=2,
        dim_sizes=0,
        values=m.data.__array_interface__["data"][0],
        idx1_pos=m.indptr.__array_interface__["data"][0],
        idx1_crd=m.indices.__array_interface__["data"][0],
    )

    return (m.todense(), sparse_m, m)


def gen_data(rng, M, N, format):
    if format == "dense":
        m = rng.random((M, N), dtype=np.float32)
        return (m, m, m)
    elif format == "csr":
        return csr_data(rng, M, N)
    elif format == "csc":
        return csc_data(rng, M, N)
    else:
        assert False


def parse_csr(data, M, N):
    pos_ptr = ctypes.cast(data.idx1_pos, ctypes.POINTER(ctypes.c_int))
    crd_ptr = ctypes.cast(data.idx1_crd, ctypes.POINTER(ctypes.c_int))
    val_ptr = ctypes.cast(data.values, ctypes.POINTER(ctypes.c_float))

    pos = np.ctypeslib.as_array(pos_ptr, shape=(M + 1,))
    if pos[-1] == 0:
        return np.zeros((M, N), np.float32)

    crd = np.ctypeslib.as_array(crd_ptr, shape=(pos[-1],))
    val = np.ctypeslib.as_array(val_ptr, shape=(pos[-1],))

    return sparse.csr_matrix((val, crd, pos)).toarray(order="C")


def parse_csc(data, M, N):
    pos_ptr = ctypes.cast(data.idx1_pos, ctypes.POINTER(ctypes.c_int))
    crd_ptr = ctypes.cast(data.idx1_crd, ctypes.POINTER(ctypes.c_int))
    val_ptr = ctypes.cast(data.values, ctypes.POINTER(ctypes.c_float))

    pos = np.ctypeslib.as_array(pos_ptr, shape=(N + 1,))
    if pos[-1] == 0:
        return np.zeros((M, N), np.float32)

    crd = np.ctypeslib.as_array(crd_ptr, shape=(pos[-1],))
    val = np.ctypeslib.as_array(val_ptr, shape=(pos[-1],))

    return sparse.csc_matrix((val, crd, pos)).toarray(order="C")


def parse_data(data, M, N, format):
    if format == "dense":
        return data
    elif format == "csr":
        return parse_csr(data, M, N)
    elif format == "csc":
        return parse_csc(data, M, N)
    else:
        assert False


def build_mm(data: dict[str, str]):

    M, K, N, nnz = (dace.symbol(s) for s in ("M", "K", "N", "nnz"))

    sdfg = dace.SDFG(f"tin_mm_{data['B']}_{data['C']}_{data['A']}_test")

    dims = {"B": (M, K), "C": (K, N), "A": (M, N)}

    for name, format in data.items():
        if format == "dense":
            sdfg.add_array(name, dims[name], dace.float32)
        elif format == "csr":
            sdfg.add_datadesc(name, csr(dims[name][0], dims[name][1], nnz))
        elif format == "csc":
            sdfg.add_datadesc(name, csc(dims[name][0], dims[name][1], nnz))
        else:
            assert False

    state = sdfg.add_state()

    A = state.add_access("A")
    B = state.add_access("B")
    C = state.add_access("C")

    tin_node = TensorIndexNotation("test", "A(i,j) = B(i,k) * C(k,j)")
    tin_node.add_in_connector("tin_B")
    tin_node.add_in_connector("tin_C")
    tin_node.add_out_connector("tin_A")

    state.add_node(tin_node)

    state.add_edge(B, None, tin_node, "tin_B", memlet=dace.Memlet(data="B"))
    state.add_edge(C, None, tin_node, "tin_C", memlet=dace.Memlet(data="C"))
    state.add_edge(tin_node, "tin_A", A, None, memlet=dace.Memlet(data="A"))

    sdfg.expand_library_nodes(recursive=True)
    sdfg.apply_transformations_repeated(InlineSDFG)

    func = sdfg.compile()

    return func


def test_mm(data: dict[str, str]):
    sizes = [
        (3, 5, 7),
        (20, 20, 20),
        (40, 20, 30),
        (30, 40, 20),
    ]

    func = build_mm(data)

    rng = np.random.default_rng(42)

    for M, K, N in sizes:
        print(
            f"Tesing {data['B']}({M}, {K}) x {data['C']}({K}, {N}) -> {data['A']}({M}, {N})"
        )

        _, AA, A_keep_alive = gen_data(rng, M, N, data["A"])
        B, BB, B_keep_alive = gen_data(rng, M, K, data["B"])
        C, CC, C_keep_alive = gen_data(rng, K, N, data["C"])

        func(A=AA, B=BB, C=CC, M=M, K=K, N=N, nnz=1)

        AA = parse_data(AA, M, N, data["A"])
        A = B @ C

        if not np.allclose(A, AA):
            print(f"{A=}")
            print(f"{AA=}")

        assert np.allclose(AA, A)
        print(f"SUCCESS")


def test_basic_mm():
    M, K, N, nnz = (dace.symbol(s) for s in ("M", "K", "N", "nnz"))

    CSR = Tensor.CSR((M, K), nnz)

    @dace.program
    def tin_mm_csr(A: dace.float32[M, N], B: CSR, C: dace.float32[K, N]):
        TensorIndexNotation("test", "A(i,j) = B(i,k) * C(k,j)", A=A, B=B, C=C)

    sdfg = tin_mm_csr.to_sdfg()

    sdfg.expand_library_nodes(recursive=True)
    sdfg.apply_transformations_repeated(InlineSDFG)

    func = sdfg.compile()

    rng = np.random.default_rng(42)
    B = sparse.random(
        20, 30, density=0.1, format="csr", dtype=np.float32, random_state=rng
    )
    C = rng.random((30, 40), dtype=np.float32)
    A = np.zeros((20, 40), dtype=np.float32)

    inpB = CSR.dtype._typeclass.as_ctypes()(
        order=2,
        dim_sizes=0,
        values=B.data.__array_interface__["data"][0],
        idx1_pos=B.indptr.__array_interface__["data"][0],
        idx1_crd=B.indices.__array_interface__["data"][0],
    )

    func(A=A, B=inpB, C=C, N=40, M=20, K=30, nnz=B.nnz)
    ref = B.dot(C)

    assert np.allclose(A, ref)
    print("basic mm: SUCCESS")


def test_basic_mm_double():
    M, K, N, nnz = (dace.symbol(s) for s in ("M", "K", "N", "nnz"))

    CSR = Tensor.CSR((M, K), nnz, dtype=dace.float64)

    @dace.program
    def tin_mm_csr_double(A: dace.float64[M, N], B: CSR, C: dace.float64[K, N]):
        TensorIndexNotation("test", "A(i,j) = B(i,k) * C(k,j)", A=A, B=B, C=C)

    sdfg = tin_mm_csr_double.to_sdfg()

    sdfg.expand_library_nodes(recursive=True)
    sdfg.apply_transformations_repeated(InlineSDFG)

    func = sdfg.compile()

    rng = np.random.default_rng(42)
    B = sparse.random(
        20, 30, density=0.1, format="csr", dtype=np.float64, random_state=rng
    )
    C = rng.random((30, 40), dtype=np.float64)
    A = np.zeros((20, 40), dtype=np.float64)

    inpB = CSR.dtype._typeclass.as_ctypes()(
        order=2,
        dim_sizes=0,
        values=B.data.__array_interface__["data"][0],
        idx1_pos=B.indptr.__array_interface__["data"][0],
        idx1_crd=B.indices.__array_interface__["data"][0],
    )

    func(A=A, B=inpB, C=C, N=40, M=20, K=30, nnz=B.nnz)
    ref = B.dot(C)

    assert np.allclose(A, ref)
    print("basic mm double: SUCCESS")


def test_basic_mm_csr_csc_csr_param():
    M, K, N, nnz = (dace.symbol(s) for s in ("M", "K", "N", "nnz"))

    CSR = Tensor.CSR((M, K), nnz)
    CSC = Tensor.CSC((K, N), nnz)
    CSR_out = Tensor.CSR((M, N), nnz)

    @dace.program
    def tin_mm_csr_csc_param(A: CSR_out, B: CSR, C: CSC):
        TensorIndexNotation(
            "test",
            "A(i,j) = B(i,k) * C(k,j)",
            ["-s=reorder(i,j,k)", '-s=assemble(A,Insert)'],
            A=A,
            B=B,
            C=C,
        )

    sdfg = tin_mm_csr_csc_param.to_sdfg()

    sdfg.expand_library_nodes(recursive=True)
    sdfg.apply_transformations_repeated(InlineSDFG)

    func = sdfg.compile()

    rng = np.random.default_rng(42)
    B = sparse.random(
        20, 30, density=0.1, format="csr", dtype=np.float32, random_state=rng
    )
    C = sparse.random(
        30, 40, density=0.1, format="csc", dtype=np.float32, random_state=rng
    )
    # A = np.zeros((20, 40), dtype=np.float32)

    inpB = CSR.dtype._typeclass.as_ctypes()(
        order=2,
        dim_sizes=0,
        values=B.data.__array_interface__["data"][0],
        idx1_pos=B.indptr.__array_interface__["data"][0],
        idx1_crd=B.indices.__array_interface__["data"][0],
    )
    inpC = CSR.dtype._typeclass.as_ctypes()(
        order=2,
        dim_sizes=0,
        values=C.data.__array_interface__["data"][0],
        idx1_pos=C.indptr.__array_interface__["data"][0],
        idx1_crd=C.indices.__array_interface__["data"][0],
    )
    inpA = CSR_out.dtype._typeclass.as_ctypes()(
        order=2,
        dim_sizes=0,
        values=C.data.__array_interface__["data"][0],
        idx1_pos=C.indptr.__array_interface__["data"][0],
        idx1_crd=C.indices.__array_interface__["data"][0],
    )

    func(A=inpA, B=inpB, C=inpC, N=40, M=20, K=30, nnz=B.nnz)

    A = parse_data(inpA, 20, 40, "csr")
    ref = B.dot(C).toarray()

    assert np.allclose(A, ref)
    print("mm CSR x CSC (with tuning): SUCCESS")


def test_basic_spmv():
    M, N, nnz = (dace.symbol(s) for s in ("M", "N", "nnz"))

    CSR = Tensor.CSR((M, N), nnz)

    @dace.program
    def tin_spmv_csr(y: dace.float32[M], A: CSR, x: dace.float32[N]):
        TensorIndexNotation("spmv", "y(i) = A(i,j) * x(j)", y=y, A=A, x=x)

    sdfg = tin_spmv_csr.to_sdfg()

    sdfg.expand_library_nodes(recursive=True)
    sdfg.apply_transformations_repeated(InlineSDFG)

    func = sdfg.compile()

    rng = np.random.default_rng(42)
    y = np.zeros(20, dtype=np.float32)
    x = rng.random(30, dtype=np.float32)
    A = sparse.random(
        20, 30, density=0.1, format="csr", dtype=np.float32, random_state=rng
    )
    A_dense = A.toarray()

    input_A = CSR.dtype._typeclass.as_ctypes()(
        order=2,
        dim_sizes=0,
        values=A.data.__array_interface__["data"][0],
        idx1_pos=A.indptr.__array_interface__["data"][0],
        idx1_crd=A.indices.__array_interface__["data"][0],
    )

    func(y=y, A=input_A, x=x, N=30, M=20, nnz=A.nnz)
    ref = A.dot(x)

    assert np.allclose(y, ref)
    print("basic spmv: SUCCESS")


def test_multiple_mm():
    N, nnz = (dace.symbol(s) for s in ("N", "nnz"))

    CSR = Tensor.CSR((N, N), nnz)
    CSC = Tensor.CSC((N, N), nnz)

    @dace.program
    def tin_multiple_mm(A: dace.float32[N, N], B: CSR, C: CSR, D: CSR, E: CSR):

        BC = dace.define_local_structure(CSR)
        TensorIndexNotation("mm1", "BC(i,j) = B(i,k) * C(k,j)", BC=BC, B=B, C=C)

        DE = dace.define_local_structure(CSR)
        TensorIndexNotation("mm2", "DE(i,j) = D(i,k) * E(k,j)", DE=DE, D=D, E=E)

        TensorIndexNotation("mm3", "A(i,j) = BC(i,k) * DE(k,j)", A=A, BC=BC, DE=DE)

    sdfg = tin_multiple_mm.to_sdfg()

    sdfg.expand_library_nodes(recursive=True)
    sdfg.apply_transformations_repeated(InlineSDFG)

    func = sdfg.compile()

    rng = np.random.default_rng(42)

    B, BB, B_keep_alive = gen_data(rng, 20, 20, "csr")
    C, CC, C_keep_alive = gen_data(rng, 20, 20, "csr")
    D, DD, D_keep_alive = gen_data(rng, 20, 20, "csr")
    E, EE, E_keep_alive = gen_data(rng, 20, 20, "csr")
    A = np.zeros((20, 20), dtype=np.float32)

    func(A=A, B=BB, C=CC, D=DD, E=EE, N=20)
    ref = (B @ C) @ (D @ E)

    assert np.allclose(A, ref)
    print("multiple mm: SUCCESS")


def test_mttkrp():
    M, K, L, N, nnz = (dace.symbol(s) for s in ("M", "K", "L", "N", "nnz"))

    @dace.program
    def mttkrp(
        A: dace.float32[M, N],
        B: dace.float32[M, K, L],
        C: dace.float32[K, N],
        D: dace.float32[L, N],
    ):
        TensorIndexNotation(
            "mttkrp", "A(i,j) = B(i,k,l) * D(l,j) * C(k,j)", A=A, B=B, C=C, D=D
        )

    sdfg = mttkrp.to_sdfg()

    sdfg.expand_library_nodes(recursive=True)
    sdfg.apply_transformations_repeated(InlineSDFG)

    func = sdfg.compile()

    rng = np.random.default_rng(42)
    B = rng.random((20, 20, 20), dtype=np.float32)
    C = rng.random((20, 20), dtype=np.float32)
    D = rng.random((20, 20), dtype=np.float32)
    A = np.zeros((20, 20), dtype=np.float32)

    func(A=A, B=B, C=C, D=D, M=20, K=20, L=20, N=20)

    print("mttkrp: SUCCESS (not checked numerically)")


if __name__ == "__main__":
    # test_basic_spmv()
    # test_basic_mm()
    # test_basic_mm_double()
    test_basic_mm_csr_csc_csr_param()
    # test_mm({'B': 'dense', 'C': 'dense', 'A': 'dense'})
    # test_mm({'B': 'csr', 'C': 'dense', 'A': 'dense'})
    # test_mm({'B': 'dense', 'C': 'csc', 'A': 'dense'})
    # test_mm({'B': 'csr', 'C': 'csr', 'A': 'csr'})
    # test_mm({'B': 'csr', 'C': 'csc', 'A': 'dense'})
    # test_mm({"B": "csr", "C": "csc", "A": "csr"})
    # test_mm({'B': 'csr', 'C': 'csc', 'A': 'csc'})
    # test_mm({'B': 'csc', 'C': 'csr', 'A': 'dense'})
    # test_mm({'B': 'csc', 'C': 'csr', 'A': 'csr'}) # taco generates incorrect code
    # test_mm({'B': 'csc', 'C': 'csr', 'A': 'csc'}) # taco generates incorrect code
    # test_multiple_mm()
    # test_mttkrp()
