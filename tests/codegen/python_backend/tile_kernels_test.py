# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A GPU map over BLOCK tile nodes runs as a kernel of a tile DSL: one program per map iteration, one DSL call per
tile node. Every program runs in every dialect; the text each test asserts is the dialect's spelling of the node."""

import contextlib
from collections.abc import Iterator

import numpy as np
import pytest

import dace
from dace.codegen.exceptions import CodegenError
from dace.libraries.tileops.dispatch import TileGroup
from dace.libraries.tileops.kinds import TILE
from dace.libraries.tileops.nodes import TileBinop
from dace.libraries.tileops.nodes.tile_op import TileOp
from dace.sdfg.validation import InvalidSDFGNodeError
from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.canonicalize.finalize import offload_to_gpu
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.vectorize_gpu import VectorizeGPU

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.tile,
    # Triton's own code generator builds an ast.AnnAssign Python 3.14 deprecates; not this backend's code
    pytest.mark.filterwarnings("ignore:AnnAssign.__init__ missing:DeprecationWarning"),
    # TileLang's own script builder calls typing._eval_type the way Python 3.14 deprecates
    pytest.mark.filterwarnings("ignore:Failing to pass a value to the 'type_params':DeprecationWarning"),
]

DIALECTS = [
    pytest.param("triton", marks=pytest.mark.triton),
    pytest.param("cutile", marks=pytest.mark.cutile),
    pytest.param("tilelang", marks=pytest.mark.tilelang),
]

#: Per test, the text of each dialect's kernel that shows the node lowered as the test names.
SPELLINGS = {
    "masked": {
        "triton": ["tl.program_id(0) * 512", "mask=_tile_iter_mask", "(int_ceil(N, 512),)"],
        "cutile": ["ct.bid(0) * 512", "mask=_tile_iter_mask", "(int_ceil(N, 512),)"],
        "tilelang": ["bx * 512", "_tile_iter_mask[lane0]", "T.Kernel(T.ceildiv(N, 512)"],
    },
    "gemm": {
        "triton": ["tl.full([16, 16], (0.0), tl.float32)", "tl.dot(A_tile, B_tile)"],
        "cutile": ["ct.full((16, 16), (0.0), ct.float32)", "ct.matmul(A_tile, B_tile)"],
        "tilelang": ["A_tile = T.alloc_shared((16, 16,)", "T.gemm(A_tile, B_tile, acc)"],
    },
    "element": {
        "triton": ["tl.store(S_gpu + r * 1, sum)"],
        "cutile": ["ct.scatter(S_gpu, (r,), sum)"],
        "tilelang": ["S_gpu[r * 1] = sum"],
    },
    "gather": {
        "triton": ["tl.load(x + (0 + _idx_x_0_load0) * 1, mask="],
        "cutile": ["ct.gather(x, (0 + _idx_x_0_load0,), mask="],
        "tilelang": ["x[(0 + _idx_x_0_load0[lane0]) * 1]"],
    },
    "scatter": {
        "triton": ["tl.store(y + (0 + _idx_scatter_y_0_load0) * 1, y_scatter_out, mask="],
        "cutile": ["ct.scatter(y, (0 + _idx_scatter_y_0_load0,), y_scatter_out, mask="],
        "tilelang": ["y[(0 + _idx_scatter_y_0_load0[lane0]) * 1] = y_scatter_out[lane0]"],
    },
    "invariant": {
        "triton": ["tl.store(a + (0 + 0 * tl.arange(0, 512)) * 1, ", ", TritonPointer(c), t[0])"],
        "cutile": ["ct.scatter(a, (0 + 0 * ct.arange(512, dtype=ct.int32),), ", ", c, t[0],)"],
        "tilelang": ["a[(0 + 0 * lane0) * 1] = "],
    },
    "atomic": {
        "triton": ["tl.atomic_add(S_gpu + "],
        "cutile": ["ct.atomic_add(S_gpu, "],
        "tilelang": ["T.atomic_add(S_gpu["],
    },
}

M, N, K, L = (dace.symbol(name) for name in "MNKL")


@contextlib.contextmanager
def tile_backend(dialect: str) -> Iterator[None]:
    with (
        dace.config.set_temporary("compiler", "backend", value="python"),
        dace.config.set_temporary("compiler", "python", "tile_dialect", value=dialect),
    ):
        yield


def kernel_source(compiled) -> str:
    """The kernels and host function of a compiled program, without the prelude."""
    source = compiled.path.read_text()
    return source[source.index("\ndef kernel_") :]


def assert_spelled(compiled, test: str, dialect: str) -> None:
    source = kernel_source(compiled)
    for text in SPELLINGS[test][dialect]:
        assert text in source, f"{dialect} kernel lacks {text!r}"


@dace.program
def scale_add(x: dace.float32[N], y: dace.float32[N], z: dace.float32[N]):
    for i in dace.map[0:N]:
        z[i] = x[i] * 2.0 + y[i]


@pytest.mark.parametrize("dialect", DIALECTS)
def test_a_vectorized_map_with_a_ragged_tail_runs_as_one_masked_kernel(dialect: str):
    import cupy

    sdfg = scale_add.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.simplify()
    offload_to_gpu(sdfg)
    VectorizeGPU(VectorizeConfig(widths=(4,))).apply_pass(sdfg, {})
    x, y, z = (cupy.arange(1000, dtype=np.float32), cupy.ones(1000, np.float32), cupy.zeros(1000, np.float32))

    with tile_backend(dialect):
        compiled = sdfg.compile()
        compiled(x=x, y=y, z=z, N=1000)

    assert np.array_equal(z.get(), 2 * np.arange(1000, dtype=np.float32) + 1)
    assert_spelled(compiled, "masked", dialect)


BM, BN, BK = 16, 16, 16


@dace.program
def blocked_gemm(A: dace.float32[M, K], B: dace.float32[K, N], C: dace.float32[M, N]):
    for i, j in dace.map[0:M:BM, 0:N:BN]:
        acc = dace.define_local([BM, BN], dace.float32, storage=dace.StorageType.Register)
        dace.tile.fill(acc, 0.0)
        for k in range(0, K, BK):
            dace.tile.mma(A[i : i + BM, k : k + BK], B[k : k + BK, j : j + BN], acc)
        dace.tile.store(C[i : i + BM, j : j + BN], acc)


@pytest.mark.parametrize("dialect", DIALECTS)
def test_a_blocked_gemm_accumulates_a_tile_product_in_a_loop(dialect: str):
    sdfg = blocked_gemm.to_sdfg()
    sdfg.apply_gpu_transformations()
    rng = np.random.default_rng(0)
    a, b, c = rng.random((64, 48), np.float32), rng.random((48, 32), np.float32), np.zeros((64, 32), np.float32)

    with tile_backend(dialect):
        compiled = sdfg.compile()
        compiled(A=a, B=b, C=c, M=64, N=32, K=48)

    np.testing.assert_allclose(c, a @ b, rtol=1e-3, atol=1e-3)
    assert_spelled(compiled, "gemm", dialect)


@dace.program
def fused_rows(A: dace.float64[L, 64], B: dace.float64[L, 64], D: dace.float64[L, 64], S: dace.float64[L]):
    for r in dace.map[0:L]:
        row = dace.tile.fma(A[r, :], B[r, :], dace.tile.exp(A[r, :]))
        dace.tile.store(D[r, :], row)
        S[r : r + 1] = dace.tile.sum(row)


@pytest.mark.parametrize("dialect", DIALECTS)
def test_elementwise_tiles_and_a_row_sum_store_one_element(dialect: str):
    sdfg = fused_rows.to_sdfg()
    sdfg.apply_gpu_transformations()
    rng = np.random.default_rng(1)
    a, b, d, s = rng.random((4, 64)), rng.random((4, 64)), np.zeros((4, 64)), np.zeros(4)

    with tile_backend(dialect):
        compiled = sdfg.compile()
        compiled(A=a, B=b, D=d, S=s, L=4)

    expected = a * b + np.exp(a)
    np.testing.assert_allclose(d, expected, rtol=1e-12)
    np.testing.assert_allclose(s, expected.sum(axis=1), rtol=1e-12)
    assert_spelled(compiled, "element", dialect)


W = 256


@dace.program
def row_reductions(A: dace.float64[L, W], R: dace.float64[L, 4], C: dace.float64[L, W], X: dace.float64[L, W]):
    for r in dace.map[0:L]:
        R[r, 0:1] = dace.tile.sum(A[r, :])
        R[r, 1:2] = dace.tile.prod(A[r, :])
        R[r, 2:3] = dace.tile.max(A[r, :])
        R[r, 3:4] = dace.tile.min(A[r, :])
        dace.tile.store(C[r, :], dace.tile.cumsum(A[r, :]))
        dace.tile.store(X[r, :], dace.tile.cummax(A[r, :]))


@pytest.mark.parametrize(
    "dialect",
    [
        *DIALECTS[:2],
        pytest.param(
            "tilelang",
            marks=[
                pytest.mark.tilelang,
                pytest.mark.xfail(reason="TileLang has no product reduction", strict=True, raises=CodegenError),
            ],
        ),
    ],
)
def test_reductions_and_scans_match_numpy(dialect: str):
    sdfg = row_reductions.to_sdfg()
    sdfg.apply_gpu_transformations()
    a = 0.5 + np.random.default_rng(2).random((4, W))
    r, c, x = np.zeros((4, 4)), np.zeros((4, W)), np.zeros((4, W))

    with tile_backend(dialect):
        sdfg(A=a, R=r, C=c, X=x, L=4)

    np.testing.assert_allclose(r[:, 0], a.sum(axis=1), rtol=1e-12)
    np.testing.assert_allclose(r[:, 1], a.prod(axis=1), rtol=1e-12)
    np.testing.assert_array_equal(r[:, 2], a.max(axis=1))
    np.testing.assert_array_equal(r[:, 3], a.min(axis=1))
    np.testing.assert_allclose(c, np.cumsum(a, axis=1), rtol=1e-12)
    np.testing.assert_array_equal(x, np.maximum.accumulate(a, axis=1))


@dace.program
def total(A: dace.float64[L], S: dace.float64[1]):
    for i in dace.map[0:L:64]:
        S[0:1] += dace.tile.sum(A[i : i + 64])


@pytest.mark.parametrize("dialect", DIALECTS)
def test_a_sum_accumulated_by_every_program_adds_atomically(dialect: str):
    sdfg = total.to_sdfg()
    sdfg.apply_gpu_transformations()
    a, s = np.random.default_rng(3).random(64 * 6), np.zeros(1)

    with tile_backend(dialect):
        compiled = sdfg.compile()
        compiled(A=a, S=s, L=a.size)

    np.testing.assert_allclose(s, [a.sum()], rtol=1e-12)
    assert_spelled(compiled, "atomic", dialect)


@dace.program
def two_row_kernels(A: dace.float64[L, 64], B: dace.float64[L, 64]):
    for r in dace.map[0:L]:
        dace.tile.store(A[r, :], dace.tile.exp(A[r, :]))
    for r in dace.map[0:L]:
        dace.tile.store(B[r, :], dace.tile.sqrt(B[r, :]))


@pytest.mark.parametrize("dialect", DIALECTS)
def test_two_kernels_of_one_label_stay_two_kernels(dialect: str):
    sdfg = two_row_kernels.to_sdfg()
    sdfg.apply_gpu_transformations()
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry):
            node.map.label = "rows"
    a, b = np.random.default_rng(4).random((2, 4, 64))

    with tile_backend(dialect):
        out_a, out_b = a.copy(), b.copy()
        sdfg(A=out_a, B=out_b, L=4)

    np.testing.assert_allclose(out_a, np.exp(a), rtol=1e-12)
    np.testing.assert_allclose(out_b, np.sqrt(b), rtol=1e-12)


@dace.program
def gather_doubled(x: dace.float64[N], idx: dace.int64[N], y: dace.float64[N]):
    for i in dace.map[0:N]:
        y[i] = x[idx[i]] * 2.0


@dace.program
def rows_gathered(x: dace.float64[8, 12], idx: dace.int64[8], y: dace.float64[8, 12]):
    for i, j in dace.map[0:8, 0:12]:
        y[i, j] = x[idx[i], j] * 2.0


@pytest.mark.parametrize("dialect", DIALECTS)
def test_a_row_gather_over_a_two_dim_tile_reads_every_indexed_row(dialect: str):
    """The index tile varies over the row dim only: its shape is ``(4, ONE)``, the marker one lane wide."""
    import cupy

    sdfg = rows_gathered.to_sdfg(simplify=True)
    offload_to_gpu(sdfg)
    VectorizeGPU(VectorizeConfig(widths=(4, 4), tile_programs=True)).apply_pass(sdfg, {})
    rng = np.random.default_rng(9)
    permutation, x, y = rng.permutation(8), rng.random((8, 12)), cupy.zeros((8, 12))

    with tile_backend(dialect):
        sdfg.compile()(x=cupy.asarray(x), idx=cupy.asarray(permutation), y=y)

    assert np.array_equal(y.get(), 2.0 * x[permutation])


@dace.program
def scatter_incremented(x: dace.float64[N], idx: dace.int64[N], y: dace.float64[N]):
    for i in dace.map[0:N]:
        y[idx[i]] = x[i] + 1.0


def vectorized_gpu_sdfg(program, tile_programs: bool = False) -> dace.SDFG:
    sdfg = program.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.simplify()
    offload_to_gpu(sdfg)
    VectorizeGPU(VectorizeConfig(widths=(4,), tile_programs=tile_programs)).apply_pass(sdfg, {})
    return sdfg


@pytest.mark.parametrize("dialect", DIALECTS)
def test_a_load_through_an_index_tile_reads_every_indexed_element(dialect: str):
    import cupy

    permutation = np.random.default_rng(3).permutation(1000)
    x, y = cupy.arange(1000, dtype=np.float64), cupy.zeros(1000)

    with tile_backend(dialect):
        compiled = vectorized_gpu_sdfg(gather_doubled).compile()
        compiled(x=x, idx=cupy.asarray(permutation), y=y, N=1000)

    assert np.array_equal(y.get(), 2.0 * permutation)
    assert_spelled(compiled, "gather", dialect)


@pytest.mark.parametrize("dialect", DIALECTS)
def test_a_store_through_an_index_tile_writes_every_indexed_element(dialect: str):
    import cupy

    permutation = np.random.default_rng(4).permutation(1000)
    x, y = cupy.arange(1000, dtype=np.float64), cupy.zeros(1000)
    expected = np.zeros(1000)
    expected[permutation] = np.arange(1000) + 1.0

    with tile_backend(dialect):
        compiled = vectorized_gpu_sdfg(scatter_incremented).compile()
        compiled(x=x, idx=cupy.asarray(permutation), y=y, N=1000)

    assert np.array_equal(y.get(), expected)
    assert_spelled(compiled, "scatter", dialect)


@dace.program
def row_recurrence(A: dace.float64[M, N]):
    for r in range(1, M):
        for j in dace.map[0:N]:
            A[r, j] = A[r - 1, j] + 1.0


@pytest.mark.parametrize("dialect", DIALECTS)
def test_a_kernel_inside_a_host_loop_reads_the_loop_iterator(dialect: str):
    import cupy

    a = cupy.zeros((5, 600))
    sdfg = vectorized_gpu_sdfg(row_recurrence)

    with tile_backend(dialect):
        compiled = sdfg.compile()
        compiled(A=a, M=5, N=600)

    assert np.array_equal(a.get(), np.repeat(np.arange(5.0)[:, None], 600, axis=1))
    signature = next(line for line in kernel_source(compiled).splitlines() if line.startswith("def kernel_"))
    assert "_loop_it_0" in signature


@dace.program
def shrinking_rows(a: dace.float64[N], A: dace.float64[N, N]):
    for r in range(N):
        for j in range(r + 1, N):
            a[j] = a[j] + A[r, j]


@pytest.mark.parametrize("dialect", DIALECTS)
def test_a_kernel_over_an_empty_map_launches_nothing(dialect: str):
    import cupy

    a, matrix = cupy.zeros(4), cupy.ones((4, 4))
    sdfg = vectorized_gpu_sdfg(shrinking_rows)

    with tile_backend(dialect):
        compiled = sdfg.compile()
        compiled(a=a, A=matrix, N=4)

    assert np.array_equal(a.get(), np.arange(4.0))
    assert " > 0:" in kernel_source(compiled)


@dace.program
def shifted_rows(A: dace.float64[M, N]):
    for r in range(M):
        for j in dace.map[r + 1 : N]:
            A[r, j] = A[r, j] + 1.0


@pytest.mark.parametrize("dialect", DIALECTS)
def test_a_map_inside_a_kernel_keeps_block_tiles_when_a_program_runs_each_tile(dialect: str):
    import cupy

    a = cupy.zeros((5, 700))
    sdfg = vectorized_gpu_sdfg(shifted_rows, tile_programs=True)

    with tile_backend(dialect):
        compiled = sdfg.compile()
        compiled(A=a, M=5, N=700)

    assert np.array_equal(a.get(), np.triu(np.ones((5, 700)), 1))
    assert all(n.group is TileGroup.BLOCK for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TileOp))


@dace.program
def total_of_squares(x: dace.float64[N], s: dace.float64[1]):
    for i in dace.map[0:N]:
        s[0] += x[i] * x[i]


@pytest.mark.parametrize("dialect", DIALECTS)
def test_a_kernel_reduction_keeps_block_tiles_when_a_program_runs_each_tile(dialect: str):
    import cupy

    x, s = np.random.default_rng(8).random(3000), cupy.zeros(1)
    sdfg = vectorized_gpu_sdfg(total_of_squares, tile_programs=True)

    with tile_backend(dialect):
        sdfg.compile()(x=cupy.asarray(x), s=s, N=3000)

    assert np.isclose(s.get()[0], np.dot(x, x))
    assert all(n.group is TileGroup.BLOCK for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TileOp))


@dace.program
def indirect_reversed(b: dace.float64[N], c: dace.float64[N], ip: dace.int32[N], a: dace.float64[N]):
    for i in range(N):
        k = ip[i]
        a[i] = b[i] + c[N - k - 1]


@pytest.mark.parametrize("dialect", DIALECTS)
def test_an_index_a_kernel_assigns_between_states_is_read_from_its_global_array(dialect: str):
    import cupy

    rng = np.random.default_rng(10)
    permutation, b, c = rng.permutation(1000).astype(np.int32), rng.random(1000), rng.random(1000)
    a = cupy.zeros(1000)

    with tile_backend(dialect):
        compiled = vectorized_gpu_sdfg(indirect_reversed).compile()
        compiled(b=cupy.asarray(b), c=cupy.asarray(c), ip=cupy.asarray(permutation), a=a, N=1000)

    assert np.array_equal(a.get(), b + c[999 - permutation])


@dace.program
def offset_gathered_sum(
    a: dace.float64[N],
    aa: dace.float64[N, N],
    ip: dace.int32[N],
    j: dace.int32,
    inc: dace.int32,
    out: dace.float64[1],
):
    s = 0.0
    for i in range(N - 1):
        s = s + a[inc + i] * aa[j - 1, ip[i]]
    out[0] = s


@pytest.mark.parametrize("dialect", DIALECTS)
def test_a_scalar_index_argument_reaches_a_tile_kernel_as_a_symbol(dialect: str):
    import cupy

    sdfg = offset_gathered_sum.to_sdfg(simplify=True)
    canonicalize(sdfg, target="gpu")
    offload_to_gpu(sdfg)
    VectorizeGPU(VectorizeConfig(widths=(4,), tile_programs=True)).apply_pass(sdfg, {})
    rng = np.random.default_rng(5)
    a, aa, permutation = rng.random(600), rng.random((600, 600)), rng.permutation(600).astype(np.int32)
    out = cupy.zeros(1)

    with tile_backend(dialect):
        sdfg.compile()(a=cupy.asarray(a), aa=cupy.asarray(aa), ip=cupy.asarray(permutation), j=4, inc=1, out=out, N=600)

    assert np.isclose(out.get()[0], np.dot(a[1:600], aa[3, permutation[:599]]))


@dace.program
def carried_product(b: dace.float64[N], c: dace.float64[N], a: dace.float64[N]):
    t = 0.0
    for i in range(N):
        s = b[i] * c[i]
        a[i] = s + t
        t = s


def canonical_tile_sdfg(program) -> dace.SDFG:
    sdfg = program.to_sdfg(simplify=True)
    canonicalize(sdfg, target="gpu")
    offload_to_gpu(sdfg)
    VectorizeGPU(VectorizeConfig(widths=(4,), tile_programs=True)).apply_pass(sdfg, {})
    return sdfg


def global_output_binop(sdfg: dace.SDFG) -> TileBinop:
    """The one binary tile node that writes a global array directly, not through a store."""
    found = [
        node
        for node, state in sdfg.all_nodes_recursive()
        if isinstance(node, TileBinop)
        and any(state.sdfg.arrays[e.data.data].storage is dace.StorageType.GPU_Global for e in state.out_edges(node))
    ]
    assert len(found) == 1
    return found[0]


@pytest.mark.parametrize("dialect", DIALECTS)
def test_a_lane_invariant_tile_node_stores_the_one_global_element_it_writes(dialect: str):
    """The canonicalizer peels the first iteration of the carried loop into a size-one map, whose tile node adds
    two scalars into ``a[0]``; the carry ``t`` it reads is a host scalar, which the kernel takes by value."""
    import cupy

    sdfg = canonical_tile_sdfg(carried_product)
    global_output_binop(sdfg)
    b, c, a = cupy.arange(1.0, 1001.0), cupy.full(1000, 2.0), cupy.zeros(1000)
    products = 2.0 * np.arange(1.0, 1001.0)

    with tile_backend(dialect):
        compiled = sdfg.compile()
        compiled(b=b, c=c, a=a, N=1000)

    assert np.array_equal(a.get(), products + np.concatenate(([0.0], products[:-1])))
    assert_spelled(compiled, "invariant", dialect)


def test_a_tile_node_writing_a_tile_into_one_global_element_is_refused():
    """The store above stores one value from every lane, which holds only for a node without a tile operand."""
    sdfg = canonical_tile_sdfg(carried_product)
    global_output_binop(sdfg).kind_a = TILE

    with tile_backend("triton"), pytest.raises(InvalidSDFGNodeError, match="output-kind rule violated"):
        sdfg.compile()


@dace.program
def column_sweep(A: dace.float64[N, M], B: dace.float64[N, M]):
    for i in dace.map[0:M]:
        for j in range(1, N):
            A[j, i] = A[j - 1, i] + B[j, i]


@dace.program
def triangular_rows(A: dace.float64[N, M], out: dace.float64[M]):
    for i in dace.map[0:M]:
        for j in range(i):
            out[i] = out[i] + A[j, i]


@pytest.mark.parametrize("dialect", DIALECTS)
def test_a_lane_uniform_loop_runs_inside_the_tile_kernel(dialect: str):
    """Every lane runs the sweep along ``j`` in step, carrying its own column through ``A[j - 1, i]``."""
    import cupy

    rng = np.random.default_rng(8)
    a, b = rng.random((7, 1000)), rng.random((7, 1000))
    device = cupy.asarray(a)

    with tile_backend(dialect):
        compiled = vectorized_gpu_sdfg(column_sweep, tile_programs=True).compile()
        compiled(A=device, B=cupy.asarray(b), N=7, M=1000)

    expected = a.copy()
    for j in range(1, 7):
        expected[j] = expected[j - 1] + b[j]
    assert np.allclose(device.get(), expected)
    assert "in range(1, N):" in kernel_source(compiled)


def test_a_loop_bounded_by_the_tiled_param_leaves_the_map_untiled():
    """``j < i`` gives each lane its own trip count, so the map keeps no tile node and the backend refuses it."""
    sdfg = vectorized_gpu_sdfg(triangular_rows, tile_programs=True)

    assert not any(isinstance(node, TileOp) for node, _ in sdfg.all_nodes_recursive())
    with tile_backend("triton"), pytest.raises(CodegenError, match="holds no tile ops"):
        sdfg.compile()


if __name__ == "__main__":
    # TileLang, which has no product reduction, runs under pytest
    for dialect in ("triton", "cutile"):
        test_a_vectorized_map_with_a_ragged_tail_runs_as_one_masked_kernel(dialect)
        test_a_blocked_gemm_accumulates_a_tile_product_in_a_loop(dialect)
        test_elementwise_tiles_and_a_row_sum_store_one_element(dialect)
        test_reductions_and_scans_match_numpy(dialect)
        test_a_sum_accumulated_by_every_program_adds_atomically(dialect)
        test_two_kernels_of_one_label_stay_two_kernels(dialect)
        test_a_row_gather_over_a_two_dim_tile_reads_every_indexed_row(dialect)
        test_a_load_through_an_index_tile_reads_every_indexed_element(dialect)
        test_a_store_through_an_index_tile_writes_every_indexed_element(dialect)
        test_a_kernel_inside_a_host_loop_reads_the_loop_iterator(dialect)
        test_a_kernel_over_an_empty_map_launches_nothing(dialect)
        test_a_map_inside_a_kernel_keeps_block_tiles_when_a_program_runs_each_tile(dialect)
        test_a_kernel_reduction_keeps_block_tiles_when_a_program_runs_each_tile(dialect)
        test_an_index_a_kernel_assigns_between_states_is_read_from_its_global_array(dialect)
        test_a_scalar_index_argument_reaches_a_tile_kernel_as_a_symbol(dialect)
        test_a_lane_invariant_tile_node_stores_the_one_global_element_it_writes(dialect)
        test_a_lane_uniform_loop_runs_inside_the_tile_kernel(dialect)
    test_a_tile_node_writing_a_tile_into_one_global_element_is_refused()
    test_a_loop_bounded_by_the_tiled_param_leaves_the_map_untiled()
