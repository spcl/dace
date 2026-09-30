# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import numpy as np
import pytest
import scipy

W = dace.symbol('W')
H = dace.symbol('H')
nnz = dace.symbol('nnz')


@dace.program
def spmv(A_row: dace.uint32[H + 1], A_col: dace.uint32[nnz], A_val: dace.float32[nnz], x: dace.float32[W],
         b: dace.float32[H]):

    @dace.mapscope(_[0:H])
    def compute_row(i):

        @dace.map(_[A_row[i]:A_row[i + 1]])
        def compute(j):
            a << A_val[j]
            in_x << x[A_col[j]]
            out >> b(1, lambda x, y: x + y)[i]

            out = a * in_x


@pytest.mark.gpu
def test_dynamic_map():
    height = 1024
    width = 1024

    # Prepare spmv SDFG for GPU
    sdfg = spmv.to_sdfg()
    sdfg.apply_gpu_transformations()

    for node in sdfg.all_nodes_recursive():
        if isinstance(node[0], dace.sdfg.nodes.MapEntry) \
                and node[0].schedule == dace.dtypes.ScheduleType.Sequential:
            node[0].schedule = dace.dtypes.ScheduleType.GPU_ThreadBlock_Dynamic

    # Fill input data
    # each row has up (including) 256 elements
    A_row = np.random.randint(257, size=height + 1, dtype=dace.uint32.type)
    A_row[0] = 0
    A_row = np.cumsum(A_row, dtype=dace.uint32.type)

    # Column data
    A_col = dace.ndarray([A_row[height]], dtype=dace.uint32)
    for i in range(height):
        A_col[A_row[i]:A_row[i + 1]] = np.sort(np.random.choice(width, A_row[i + 1] - A_row[i], replace=False))

    # values
    A_val = np.random.rand(A_row[height]).astype(dace.float32.type)

    A_sparse = scipy.sparse.csr_matrix((A_val, A_col, A_row), dtype=dace.float32.type, shape=(1024, 1024))

    x = np.random.rand(width).astype(dace.float32.type)
    b = np.zeros(height, dtype=dace.float32.type)

    sdfg(A_row=A_row, A_col=A_col, A_val=A_val, x=x, b=b, H=A_sparse.shape[0], W=A_sparse.shape[1], nnz=A_sparse.nnz)

    diff = np.linalg.norm(A_sparse.dot(x) - b) / float(height)
    print("Difference:", diff)
    print("==== Program end ====")
    assert diff <= 1e-5


@pytest.mark.gpu
def test_dynamic_maps():
    """ Tests the case of multiple dynamic maps in a row that share dynamic inputs."""

    W = dace.symbol('W')
    H = dace.symbol('H')
    nnz = dace.symbol('nnz')

    @dace.program
    def spmv_2x(A_row: dace.uint32[H + 1], A_col: dace.uint32[nnz], A_val: dace.float32[nnz], x: dace.float32[W],
                b: dace.float32[H], c: dace.float32[H]):

        for i in range(H):
            row_start = A_row[i]
            row_end = A_row[i + 1]
            for j in dace.map[row_start:row_end]:
                b[i] += A_val[j] * x[A_col[j]]
            for j in dace.map[row_start:row_end]:
                c[i] += A_val[j] * x[A_col[j]]

    height = 1024
    width = 1024

    # Prepare spmv SDFG for GPU
    sdfg = spmv_2x.to_sdfg()
    # Rename dynamic inputs to cause name clashes
    main_entry = None
    main_dict = {}
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, dace.sdfg.nodes.MapEntry):
            if main_entry is None:
                main_entry = node
                for e in dace.sdfg.dynamic_map_inputs(state, node):
                    main_dict[e.data.data] = e.dst_conn
            else:
                repl_dict = {}
                for e in dace.sdfg.dynamic_map_inputs(state, node):
                    node.remove_in_connector(e.dst_conn)
                    node.add_in_connector(main_dict[e.data.data])
                    repl_dict[e.dst_conn] = main_dict[e.data.data]
                    e._dst_conn = main_dict[e.data.data]
                node.map.range.replace(repl_dict)

    sdfg.apply_gpu_transformations()

    for node in sdfg.all_nodes_recursive():
        if isinstance(node[0], dace.sdfg.nodes.MapEntry) \
                and node[0].schedule == dace.dtypes.ScheduleType.Sequential:
            node[0].schedule = dace.dtypes.ScheduleType.GPU_ThreadBlock_Dynamic

    # Fill input data
    # each row has up (including) 256 elements
    A_row = np.random.randint(257, size=height + 1, dtype=dace.uint32.type)
    A_row[0] = 0
    A_row = np.cumsum(A_row, dtype=dace.uint32.type)

    # Column data
    A_col = dace.ndarray([A_row[height]], dtype=dace.uint32)
    for i in range(height):
        A_col[A_row[i]:A_row[i + 1]] = np.sort(np.random.choice(width, A_row[i + 1] - A_row[i], replace=False))

    # values
    A_val = np.random.rand(A_row[height]).astype(dace.float32.type)

    A_sparse = scipy.sparse.csr_matrix((A_val, A_col, A_row), dtype=dace.float32.type, shape=(1024, 1024))

    x = np.random.rand(width).astype(dace.float32.type)
    b = np.zeros(height, dtype=dace.float32.type)
    c = np.zeros(height, dtype=dace.float32.type)

    sdfg(A_row=A_row,
         A_col=A_col,
         A_val=A_val,
         x=x,
         b=b,
         c=c,
         H=A_sparse.shape[0],
         W=A_sparse.shape[1],
         nnz=A_sparse.nnz)

    diff0 = np.linalg.norm(A_sparse.dot(x) - b) / float(height)
    diff1 = np.linalg.norm(A_sparse.dot(x) - c) / float(height)
    assert diff0 <= 1e-5
    assert diff1 <= 1e-5


@pytest.mark.gpu
def test_nested_dynamic_map():
    """ Tests the case where the dynamic map inputs are defined in an outer scope. """

    M = dace.symbol('M')
    N = dace.symbol('N')
    K = dace.symbol('K')
    nnz_A = dace.symbol('nnz_A')
    nnz_D = dace.symbol('nnz_D')

    @dace.program
    def sddmm(D_vals: dace.float32[nnz_D], A2_crd: dace.int32[nnz_A], A2_pos: dace.int32[M + 1],
              A_vals: dace.float32[nnz_A], B: dace.float32[M, K], C: dace.float32[K, N]):
        for i in dace.map[0:M]:
            for j in dace.map[A2_pos[i]:A2_pos[i + 1]]:
                for k in dace.map[0:K]:
                    D_vals[j] += A_vals[j] * B[i, k] * C[k, A2_crd[j]]

    sdfg = sddmm.to_sdfg(simplify=True)

    ime, jme, kme = None, None, None
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.sdfg.nodes.MapEntry):
                if node.map.params[0] == 'i':
                    ime = node
                elif node.map.params[0] == 'j':
                    jme = node
                elif node.map.params[0] == 'k':
                    kme = node
    assert ime is not None and jme is not None and kme is not None

    from dace.transformation.dataflow import MapInterchange, TrivialTaskletElimination
    MapInterchange.apply_to(sdfg, outer_map_entry=jme, inner_map_entry=kme)
    sdfg.apply_transformations_repeated(TrivialTaskletElimination)

    sdfg.apply_gpu_transformations()
    ime.map.schedule = dace.ScheduleType.GPU_Device
    kme.map.schedule = dace.ScheduleType.GPU_ThreadBlock_Dynamic

    dtype = np.float32
    rng = np.random.default_rng(42)
    problem_size = 1024
    density = 0.01
    B = rng.random((problem_size, problem_size), dtype=dtype)
    C = rng.random((problem_size, problem_size), dtype=dtype)
    A = scipy.sparse.random(problem_size, problem_size, density=density, format='csr', dtype=dtype, random_state=rng)
    val = np.zeros_like(A.data)
    ref = np.empty_like(A.data)

    sdfg(D_vals=val,
         A2_crd=A.indices.copy(),
         A2_pos=A.indptr.copy(),
         A_vals=A.data.copy(),
         B=B,
         C=C,
         M=problem_size,
         N=problem_size,
         K=problem_size,
         nnz_A=A.nnz,
         nnz_D=A.nnz)
    tmp = B @ C
    for row in range(problem_size):
        for j in range(A.indptr[row], A.indptr[row + 1]):
            col = A.indices[j]
            ref[j] = A.data[j] * tmp[row, col]
    assert np.allclose(val, ref.data)


@pytest.mark.gpu
def test_dynamic_map_with_step():

    M = dace.symbol('M')
    N = dace.symbol('N')
    nnz_A = dace.symbol('nnz_A')
    nnz_D = dace.symbol('nnz_D')

    @dace.program
    def sddvm(D_vals: dace.float32[nnz_D], A2_crd: dace.int32[nnz_A], A2_pos: dace.int32[M + 1],
              A_vals: dace.float32[nnz_A], B: dace.float32[M], C: dace.float32[N]):
        for i in dace.map[0:M]:
            for j in dace.map[A2_pos[i]:A2_pos[i + 1]]:
                D_vals[j] += A_vals[j] * B[i] * C[A2_crd[j]]

    sdfg = sddvm.to_sdfg(simplify=True)

    ime, jme = None, None
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.sdfg.nodes.MapEntry):
                if node.map.params[0] == 'i':
                    ime = node
                elif node.map.params[0] == 'j':
                    jme = node
    assert ime is not None and jme is not None

    from dace.transformation.dataflow import StripMining, TrivialTaskletElimination
    sdfg.apply_transformations_repeated(TrivialTaskletElimination)
    StripMining.apply_to(sdfg, map_entry=jme)

    tile_jme = None, None
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.sdfg.nodes.MapEntry):
                if node.map.params[0] == 'tile_j':
                    tile_jme = node
    assert tile_jme is not None

    sdfg.apply_gpu_transformations()
    ime.map.schedule = dace.ScheduleType.GPU_Device
    tile_jme.map.schedule = dace.ScheduleType.GPU_ThreadBlock_Dynamic

    dtype = np.float32
    rng = np.random.default_rng(42)
    problem_size = 1024
    density = 0.01
    B = rng.random((problem_size, ), dtype=dtype)
    C = rng.random((problem_size, ), dtype=dtype)
    A = scipy.sparse.random(problem_size, problem_size, density=density, format='csr', dtype=dtype, random_state=rng)
    val = np.zeros_like(A.data)
    ref = np.empty_like(A.data)

    sdfg(D_vals=val,
         A2_crd=A.indices.copy(),
         A2_pos=A.indptr.copy(),
         A_vals=A.data.copy(),
         B=B,
         C=C,
         M=problem_size,
         N=problem_size,
         nnz_A=A.nnz,
         nnz_D=A.nnz)
    tmp = np.outer(B, C)
    for row in range(problem_size):
        for j in range(A.indptr[row], A.indptr[row + 1]):
            col = A.indices[j]
            ref[j] = A.data[j] * tmp[row, col]
    assert np.allclose(val, ref.data)


@pytest.mark.gpu
def test_dynamic_multidim_map():

    @dace.program
    def tester(a: dace.float32[H, W, nnz]):
        A = dace.ndarray([H, W, nnz], dtype=dace.float32, storage=dace.StorageType.GPU_Global)
        A[:] = a
        for i, j in dace.map[0:H, 0:W] @ dace.ScheduleType.GPU_Device:
            for k in dace.map[0:nnz] @ dace.ScheduleType.GPU_ThreadBlock_Dynamic:
                A[i, j, k] = i * 110 + j * 11 + k
        a[:] = A

    a = np.zeros((10, 11, 65), dtype=np.float32)
    tester(a)
    assert np.allclose(a, np.fromfunction(lambda i, j, k: i * 110 + j * 11 + k, (10, 11, 65), dtype=np.float32))


@pytest.mark.skip('Nested maps with work-stealing thread-block schedule are currently unsupported')
def test_dynamic_nested_map():

    @dace.program
    def nested2(A: dace.float32[W], i: dace.int32, j: dace.int32):
        A[j] = i * 10 + j

    @dace.program
    def nested1(A: dace.float32[W], i: dace.int32):
        for j in dace.map[0:W] @ dace.ScheduleType.GPU_ThreadBlock_Dynamic:
            nested2(A, i, j)

    @dace.program
    def dynamic_nested_map(a: dace.float32[H, W]):
        A = dace.ndarray([H, W], dtype=dace.float32, storage=dace.StorageType.GPU_Global)
        A[:] = a
        for i in dace.map[0:H] @ dace.ScheduleType.GPU_Device:
            nested1(A[i], i)

        a[:] = A

    a = np.zeros((10, 11), dtype=np.float32)
    sdfg = dynamic_nested_map.to_sdfg(simplify=False)
    for _, _, arr in sdfg.arrays_recursive():
        if arr.storage in (dace.StorageType.GPU_Shared, dace.StorageType.Default):
            arr.storage = dace.StorageType.Register
    sdfg(a, H=10, W=11)
    assert np.allclose(a, np.fromfunction(lambda i, j: i * 10 + j, (10, 11), dtype=np.float32))


@pytest.mark.gpu
def test_dynamic_default_schedule():
    N = dace.symbol('N')

    @dace.program
    def tester(a: dace.float32[N, 10]):
        A = dace.ndarray([N, 10], dtype=dace.float32, storage=dace.StorageType.GPU_Global)
        A[:] = a
        for i in dace.map[0:N] @ dace.ScheduleType.GPU_Device:
            smem = np.empty((10, ), dtype=np.float32) @ dace.StorageType.GPU_Shared
            smem[:] = 1
            for j in dace.map[0:10] @ dace.ScheduleType.GPU_ThreadBlock_Dynamic:
                A[i, j] = i * 65 + smem[j]
        a[:] = A

    a = np.zeros((65, 10), dtype=np.float32)
    tester(a)
    assert np.allclose(a, np.fromfunction(lambda i, j: i * 65 + 1, (65, 10), dtype=np.float32))


if __name__ == '__main__':
    test_dynamic_map()
    test_dynamic_maps()
    test_nested_dynamic_map()
    test_dynamic_map_with_step()
    test_dynamic_multidim_map()
    # test_dynamic_nested_map()
    test_dynamic_default_schedule()
