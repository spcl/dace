# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A matrix product of one tile, computed by a whole GPU thread block: a library node for use inside GPU kernels.

``tile_matmul(a, b, c, work, rows)`` computes ``c = a @ b`` for an ``M x K`` matrix ``a`` and a ``K x N`` matrix ``b``,
in which ``K`` is the contiguous dimension of both (``a`` row-major, ``b`` column-major, as the transpose of a weight
matrix is). ``b`` may also be given in slices of ``K``, as a ``K / k_step x N x k_step`` array (``b[k, n]`` is
``b[k // k_step, n, k % k_step]``): weights packed so, once, are read contiguously. The first ``rows`` rows of ``c`` are
computed, the others are zero. With ``a_index``, row ``i`` of the product is row ``a_index[i]`` of ``a`` (a gather;
rows whose index is negative are zero), and ``M`` is the length of ``a_index``. ``work`` is shared memory for the node
to stage its operands in, which the caller provides so that several products in one kernel can share it (see
``workspace_shape``). ``c`` may be a view of ``work``: the node writes it once it is done with its operands. When
the node finishes, ``c`` is complete for every thread of the block.

Expansions:

* ``mma`` (default): CUDA, bfloat16 or float16 operands. The thread block's warps split the columns of the tile. Slices
  of ``a`` and ``b`` are copied asynchronously into shared memory (``cp.async``, ``stages`` slices in flight), loaded
  into tensor core fragments (``ldmatrix``), and multiplied (``mma.sync`` m16n8k16) with ``float`` accumulators. Rows
  of 16 that are all beyond ``rows`` are neither loaded nor multiplied.
* ``pure``: one thread per output element, for any types and devices.
"""

import copy
import math

import dace
import dace.library
from dace import SDFG, Memlet, SDFGState, dtypes, nodes, properties, symbolic
from dace.codegen.targets import cpp
from dace.frontend.common import op_repository as oprepo
from dace.transformation.transformation import ExpandTransformation

WARP_SIZE = 32


def workspace_shape(m: int, n: int, k_step: int = 32, stages: int = 3) -> list[int]:
    """The shape of the shared memory (in elements of the operand type) that ``tile_matmul`` needs for an ``m x n``
    tile, in ``k_step``-wide slices with ``stages`` of them in flight."""
    return [stages, m + n, k_step]


def _operands(node: "TileMatmul", sdfg: SDFG, state: SDFGState) -> dict[str, tuple[Memlet, dace.data.Data]]:
    """The memlet and data descriptor of every connector of ``node``."""
    result = {}
    for e in state.in_edges(node):
        result[e.dst_conn] = (e.data, sdfg.arrays[e.data.data])
    for e in state.out_edges(node):  # The workspace as written, unless no one reads it later (dead dataflow)
        result[e.src_conn] = (e.data, sdfg.arrays[e.data.data])
    return result


def _operand(memlet: Memlet, desc: dace.data.Data, dims: tuple[int, ...]) -> tuple[list, list]:
    """The shape and strides of the array that a memlet accesses (dimensions of size one removed), which must have one
    of the numbers of dimensions ``dims``."""
    subset = copy.deepcopy(memlet.subset)
    kept = subset.squeeze()
    shape = subset.size()
    strides = [desc.strides[d] for d in kept]
    if len(shape) not in dims:
        raise ValueError(f"tile_matmul operand {memlet.data} has the shape {shape}, expected {dims} dimensions")
    return shape, strides


def _b_matrix(shape: list) -> tuple:
    """The rows (``K``) and columns (``N``) of operand ``b``, and the width of its slices (None if not sliced)."""
    if len(shape) == 2:
        return shape[0], shape[1], None
    return shape[0] * shape[2], shape[1], shape[2]


def _nested_sdfg(node: "TileMatmul", parent_sdfg: SDFG, parent_state: SDFGState) -> tuple[SDFG, dict]:
    """A nested SDFG with the node's connectors as arguments, and the operand shapes and strides."""
    operands = _operands(node, parent_sdfg, parent_state)
    sdfg = SDFG(f"{node.label}_sdfg")
    info = {}
    for conn in ("_a", "_b", "_c", "_work"):
        memlet, desc = operands[conn]
        if conn == "_work":
            shape, strides = list(memlet.subset.size()), list(desc.strides)
        else:
            shape, strides = _operand(memlet, desc, (2, 3) if conn == "_b" else (2,))
        sdfg.add_array(conn, shape, desc.dtype, storage=desc.storage, strides=strides)
        info[conn] = (shape, strides, desc.dtype)
    if "_rows" in operands:
        sdfg.add_scalar("_rows", operands["_rows"][1].dtype, storage=operands["_rows"][1].storage)
    if "_a_index" in operands:
        memlet, desc = operands["_a_index"]
        sdfg.add_array("_a_index", memlet.subset.size(), desc.dtype, storage=desc.storage, strides=desc.strides[-1:])
        info["_a_index"] = (memlet.subset.size(), desc.strides[-1:], desc.dtype)
    (m, k), (m2, n2) = info["_a"][0], info["_c"][0]
    if "_a_index" in info:
        m = info["_a_index"][0][0]
    k2, n, _ = _b_matrix(info["_b"][0])
    if (k != k2) == True or (m != m2) == True or (n != n2) == True:
        raise ValueError(f"tile_matmul shapes do not match: {info['_a'][0]} @ {info['_b'][0]} -> {info['_c'][0]}")
    info["m"] = m
    return sdfg, info


@dace.library.expansion
class ExpandPure(ExpandTransformation):
    """One thread per output element (in a thread-block map of ``node.warps`` warps, each thread computes the elements
    at its index plus multiples of the block size)."""

    environments = []

    @staticmethod
    def expansion(node: "TileMatmul", parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        sdfg, info = _nested_sdfg(node, parent_sdfg, parent_state)
        m, k = info["m"], info["_a"][0][1]
        _, n, slice_width = _b_matrix(info["_b"][0])
        b_element = "b[kk, j]" if slice_width is None else f"b[kk // {slice_width}, j, kk % {slice_width}]"
        gather = "_a_index" in sdfg.arrays
        threads = node.warps * WARP_SIZE
        rows = "_rows" if "_rows" in sdfg.arrays else str(m)
        state = sdfg.add_state(node.label)
        code = f"""
for idx in range(w * {WARP_SIZE} + lane, {m} * {n}, {threads}):
    i = idx // {n}
    j = idx % {n}
    s = dace.float32(0)
    row = {"a_index[i]" if gather else "i"}
    if i < nrows and row >= 0:
        for kk in range({k}):
            s = s + dace.float32(a[row, kk]) * dace.float32({b_element})
    c[i, j] = s
"""
        inputs = {"a": Memlet.from_array("_a", sdfg.arrays["_a"]), "b": Memlet.from_array("_b", sdfg.arrays["_b"])}
        if gather:
            inputs["a_index"] = Memlet.from_array("_a_index", sdfg.arrays["_a_index"])
        if "_rows" in sdfg.arrays:
            inputs["nrows"] = Memlet("_rows[0]")
        else:
            code = code.replace("nrows", rows)
        c_memlet = Memlet.from_array("_c", sdfg.arrays["_c"])
        c_memlet.dynamic = True
        state.add_mapped_tasklet(
            f"{node.label}_pure",
            {"w": f"0:{node.warps}", "lane": f"0:{WARP_SIZE}"},
            inputs,
            code,
            {"c": c_memlet},
            schedule=dtypes.ScheduleType.GPU_ThreadBlock,
            external_edges=True,
        )
        return sdfg


@dace.library.expansion
class ExpandMMA(ExpandTransformation):
    """A software-pipelined tensor core product (see the module documentation)."""

    environments = []

    @staticmethod
    def expansion(node: "TileMatmul", parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        sdfg, info = _nested_sdfg(node, parent_sdfg, parent_state)
        m, k, sa = info["m"], info["_a"][0][1], info["_a"][1]
        gather = "_a_index" in sdfg.arrays
        _, n, slice_width = _b_matrix(info["_b"][0])
        sb = info["_b"][1]
        sc, ctype = info["_c"][1], info["_c"][2].ctype
        work_shape, _, work_dtype = info["_work"]
        operand = info["_a"][2]
        if operand not in (dace.bfloat16, dace.float16) or info["_b"][2] != operand or work_dtype != operand:
            raise TypeError(f"The mma expansion of tile_matmul needs bfloat16 or float16 operands, got {operand}")
        mma_type = "bf16" if operand == dace.bfloat16 else "f16"
        if (sa[1] != 1) == True or (sb[-1 if slice_width else 0] != 1) == True:
            raise ValueError("The mma expansion of tile_matmul needs K to be the contiguous dimension of a and b")
        for value, what in ((m, "rows"), (n, "columns"), (k, "inner dimension")):
            if symbolic.issymbolic(value):
                raise ValueError(f"The mma expansion of tile_matmul needs a constant number of {what}, got {value}")
        m, n, k = int(m), int(n), int(k)
        stages, k_step = int(work_shape[0]), int(work_shape[2])
        warps = node.warps
        if work_shape[1] != m + n:
            raise ValueError(
                f"The workspace of tile_matmul must have the shape {workspace_shape(m, n, k_step, stages)}"
            )
        if m % 16 or n % (8 * warps) or (n // warps) % 16 or k % k_step or k_step not in (32, 64):
            raise ValueError(
                f"The mma expansion of tile_matmul does not support a {m} x {n} x {k} tile with {warps} warps "
                f"and slices of {k_step}"
            )
        if slice_width is not None and slice_width != k_step:
            raise ValueError(f"The slices of operand b ({slice_width}) must be as wide as the workspace's ({k_step})")
        # A slice from k0 starts k0 elements into a row of a, and k0 elements (or k0 / KS slices) into b
        a_row = "a_index[r]" if gather else "r"
        b_offset = "k0" if slice_width is None else f"(k0 / KS) * ({cpp.sym2cpp(sb[0])})"
        b_column = cpp.sym2cpp(sb[1])
        # A row of a slice is ``k_step`` elements, i.e., ``chunks`` 16-byte chunks. Chunks are swizzled (XOR with the
        # row) so that the eight rows ``ldmatrix`` reads at once, and the rows ``cp.async`` writes, hit distinct banks.
        chunks = k_step // 8
        swizzle = "((r) >> 1) & 3" if chunks == 4 else "(r) & 7"
        nrows = "min((int)_rows_in, {m})" if "_rows" in sdfg.arrays else str(m)
        code = f"""
constexpr int M = {m}, N = {n}, K = {k}, KS = {k_step}, STAGES = {stages}, CHUNKS = {chunks};
constexpr int THREADS = {warps * WARP_SIZE}, WN = N / {warps}, MF = M / 16, NF = WN / 8;
const int tid = w * {WARP_SIZE} + lane;
const int nrows = {nrows.format(m=m)};
const uint32_t smem = (uint32_t) __cvta_generic_to_shared(work);
#define DACE_TM_SWZ(r, c) ((c) ^ ({swizzle}))
#define DACE_TM_ADDR(s, r, c) (smem + ((((s) * (M + N) + (r)) * CHUNKS + DACE_TM_SWZ(r, c)) << 4))

// The chunks of every slice this thread copies: where they start in A and B (null for rows of A that are
// zero-filled: beyond nrows, or with a negative index)
constexpr int A_COPIES = (M * CHUNKS + THREADS - 1) / THREADS, B_COPIES = (N * CHUNKS + THREADS - 1) / THREADS;
const {info["_a"][2].ctype} *a_src[A_COPIES];
const {info["_b"][2].ctype} *b_src[B_COPIES];
#pragma unroll
for (int j = 0; j < A_COPIES; ++j) {{
    const int i = tid + j * THREADS, r = i / CHUNKS, c = i % CHUNKS;
    const int row = (i < M * CHUNKS && r < nrows) ? (int)({a_row}) : -1;
    a_src[j] = row >= 0 ? a + row * ({cpp.sym2cpp(sa[0])}) + c * 8 : nullptr;
}}
#pragma unroll
for (int j = 0; j < B_COPIES; ++j) {{
    const int i = tid + j * THREADS, r = i / CHUNKS, c = i % CHUNKS;
    b_src[j] = b + r * ({b_column}) + c * 8;
}}

// Copies the slice of A and B starting at k0 into stage s
auto load_stage = [&](int s, int k0) {{
    #pragma unroll
    for (int j = 0; j < A_COPIES; ++j) {{
        const int i = tid + j * THREADS, r = i / CHUNKS, c = i % CHUNKS;
        if (M * CHUNKS % THREADS == 0 || i < M * CHUNKS)
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\\n"
                         :: "r"(DACE_TM_ADDR(s, r, c)), "l"(a_src[j] ? a_src[j] + k0 : a), "r"(a_src[j] ? 16 : 0));
    }}
    #pragma unroll
    for (int j = 0; j < B_COPIES; ++j) {{
        const int i = tid + j * THREADS, r = i / CHUNKS, c = i % CHUNKS;
        if (N * CHUNKS % THREADS == 0 || i < N * CHUNKS)
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\\n"
                         :: "r"(DACE_TM_ADDR(s, M + r, c)), "l"(b_src[j] + {b_offset}));
    }}
}};

float acc[MF][NF][4];
#pragma unroll
for (int i = 0; i < MF; ++i)
    #pragma unroll
    for (int j = 0; j < NF; ++j)
        acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

__syncthreads();  // The previous user of the workspace is done with it
constexpr int KT = K / KS;
#pragma unroll
for (int s = 0; s < STAGES - 1; ++s) {{
    if (s < KT) load_stage(s, s * KS);
    asm volatile("cp.async.commit_group;\\n" ::);
}}
for (int kt = 0; kt < KT; ++kt) {{
    asm volatile("cp.async.wait_group %0;\\n" :: "n"(STAGES - 2));
    __syncthreads();
    const int next = kt + STAGES - 1;
    if (next < KT) load_stage(next % STAGES, next * KS);
    asm volatile("cp.async.commit_group;\\n" ::);
    const int s = kt % STAGES;
    #pragma unroll
    for (int kk = 0; kk < KS / 16; ++kk) {{
        uint32_t bf[NF][2];
        #pragma unroll
        for (int j = 0; j < NF / 2; ++j) {{
            const int r = M + w * WN + j * 16 + (lane >> 4) * 8 + (lane & 7);
            const int c = kk * 2 + ((lane >> 3) & 1);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%0, %1, %2, %3}}, [%4];\\n"
                         : "=r"(bf[2 * j][0]), "=r"(bf[2 * j][1]), "=r"(bf[2 * j + 1][0]), "=r"(bf[2 * j + 1][1])
                         : "r"(DACE_TM_ADDR(s, r, c)));
        }}
        #pragma unroll
        for (int i = 0; i < MF; ++i) {{
            if (i * 16 >= nrows) break;
            uint32_t af[4];
            const int r = i * 16 + (lane & 15);
            const int c = kk * 2 + (lane >> 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%0, %1, %2, %3}}, [%4];\\n"
                         : "=r"(af[0]), "=r"(af[1]), "=r"(af[2]), "=r"(af[3])
                         : "r"(DACE_TM_ADDR(s, r, c)));
            #pragma unroll
            for (int j = 0; j < NF; ++j)
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.{mma_type}.{mma_type}.f32 "
                             "{{%0, %1, %2, %3}}, {{%4, %5, %6, %7}}, {{%8, %9}}, {{%0, %1, %2, %3}};\\n"
                             : "+f"(acc[i][j][0]), "+f"(acc[i][j][1]), "+f"(acc[i][j][2]), "+f"(acc[i][j][3])
                             : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]), "r"(bf[j][0]), "r"(bf[j][1]));
        }}
    }}
}}
asm volatile("cp.async.wait_group 0;\\n" ::);
__syncthreads();  // Every warp is done with the workspace, which c may alias
#undef DACE_TM_ADDR
#undef DACE_TM_SWZ

// Accumulator fragment: rows lane / 4 and lane / 4 + 8, columns 2 * (lane % 4) and the next
#pragma unroll
for (int i = 0; i < MF; ++i)
    #pragma unroll
    for (int j = 0; j < NF; ++j)
        #pragma unroll
        for (int h = 0; h < 2; ++h) {{
            const int row = i * 16 + (lane >> 2) + h * 8, col = w * WN + j * 8 + (lane & 3) * 2;
            c[row * ({cpp.sym2cpp(sc[0])}) + col * ({cpp.sym2cpp(sc[1])})] = ({ctype}) acc[i][j][2 * h];
            c[row * ({cpp.sym2cpp(sc[0])}) + (col + 1) * ({cpp.sym2cpp(sc[1])})] = ({ctype}) acc[i][j][2 * h + 1];
        }}
__syncthreads();  // The product is complete for every thread of the block
"""
        state = sdfg.add_state(node.label)
        entry, exit_ = state.add_map(
            f"{node.label}_warps",
            {"w": f"0:{warps}", "lane": f"0:{WARP_SIZE}"},
            schedule=dtypes.ScheduleType.GPU_ThreadBlock,
        )
        inputs = {"a", "b"} | ({"_rows_in"} if "_rows" in sdfg.arrays else set()) | ({"a_index"} if gather else set())
        tasklet = state.add_tasklet(f"{node.label}_mma", inputs, {"c", "work"}, code, language=dtypes.Language.CPP)
        for conn, name in (("a", "_a"), ("b", "_b"), ("_rows_in", "_rows"), ("a_index", "_a_index")):
            if conn in inputs:
                state.add_memlet_path(
                    state.add_read(name),
                    entry,
                    tasklet,
                    dst_conn=conn,
                    memlet=Memlet.from_array(name, sdfg.arrays[name]),
                )
        for conn, name in (("c", "_c"), ("work", "_work")):
            memlet = Memlet.from_array(name, sdfg.arrays[name])
            memlet.dynamic = True
            state.add_memlet_path(tasklet, exit_, state.add_write(name), src_conn=conn, memlet=memlet)
        return sdfg


@dace.library.node
class TileMatmul(nodes.LibraryNode):
    """``_c = _a @ _b`` for the first ``_rows`` rows (all if not connected), computed by one GPU thread block. ``_work``
    is shared memory for staging operands (see ``workspace_shape``)."""

    implementations = {"mma": ExpandMMA, "pure": ExpandPure}
    default_implementation = "mma"

    warps = properties.Property(dtype=int, default=4, desc="Warps of the thread block that computes the product")

    def __init__(self, name: str, warps: int = 4, rows: bool = False, gather: bool = False, **kwargs):
        # The workspace is scratch memory, read and written (an output alone would be dead to dataflow analysis)
        inputs = {"_a", "_b", "_work"} | ({"_rows"} if rows else set()) | ({"_a_index"} if gather else set())
        super().__init__(name, inputs=inputs, outputs={"_c", "_work"}, **kwargs)
        self.warps = warps


@oprepo.replaces("tile_matmul.tile_matmul")
def tile_matmul_libnode(
    pv, sdfg: SDFG, state: SDFGState, a: str, b: str, c: str, work: str, rows=None, a_index=None, warps=4
):
    """Frontend replacement of ``tile_matmul``."""
    if rows is not None and not isinstance(rows, str):
        raise TypeError("tile_matmul takes the number of rows as a scalar container (or None, for all rows)")
    node = TileMatmul("tile_matmul", warps=warps, rows=rows is not None, gather=a_index is not None)
    state.add_node(node)
    optional = (("_rows", rows), ("_a_index", a_index))
    for conn, name in (("_a", a), ("_b", b), ("_work", work)) + tuple(o for o in optional if o[1] is not None):
        state.add_edge(state.add_read(name), None, node, conn, Memlet.from_array(name, sdfg.arrays[name]))
    for conn, name in (("_c", c), ("_work", work)):
        state.add_edge(node, conn, state.add_write(name), None, Memlet.from_array(name, sdfg.arrays[name]))
    return []


def tile_matmul(a, b, c, work, rows=None, a_index=None, warps: int = 4):
    """``c = a @ b`` (or ``a[a_index] @ b``) for the first ``rows`` rows, computed by the whole thread block (see the
    module documentation). For use in DaCe programs, inside GPU kernels."""
    raise NotImplementedError("tile_matmul is only available in DaCe programs")


if __name__ == "__main__":
    # Checks both expansions against PyTorch on tiles of a 256 x 512 x 1024 product: with b column-major, with b packed,
    # with gathered rows of a, and with the product in the memory of the workspace; for all rows, and for the first five
    # of every tile
    import tile_matmul as tm  # This file as a module: the frontend finds replacements by the name of their module
    import torch

    BM, BN, KS, STAGES = 64, 128, 32, 2
    M, N, K = 256, 512, 1024
    WORKSPACE = workspace_shape(BM, BN, KS, STAGES)
    FORMS = 4
    TILES = 3  # Of the forms, those with their own tile
    STAGING = math.prod(WORKSPACE)

    b_column_major = dace.data.Array(dace.bfloat16, [K, N], strides=[1, K])

    @dace.program
    def product(
        a: dace.bfloat16[M, K],
        b: b_column_major,
        b_packed: dace.bfloat16[N // BN, K // KS, BN, KS],
        index: dace.int32[M],
        rows: dace.int32[1],
        c: dace.bfloat16[FORMS, M, N],
    ):
        for i, j in dace.map[0 : M // BM, 0 : N // BN]:
            work = dace.ndarray(WORKSPACE, dtype=dace.bfloat16, storage=dace.StorageType.GPU_Shared)
            # Beyond the static shared memory of a thread block, together with the workspace
            tile = dace.ndarray([TILES, BM, BN], dtype=dace.bfloat16, storage=dace.StorageType.GPU_Shared(dynamic=True))
            nr = rows[0]
            tm.tile_matmul(a[i * BM : i * BM + BM, :], b[:, j * BN : j * BN + BN], tile[0], work, nr)
            tm.tile_matmul(a[i * BM : i * BM + BM, :], b_packed[j], tile[1], work, nr)
            tm.tile_matmul(a, b_packed[j], tile[2], work, nr, index[i * BM : i * BM + BM])
            staging = dace.ndarray([STAGING], dtype=dace.bfloat16, storage=dace.StorageType.GPU_Shared(dynamic=True))
            in_staging = staging[0 : BM * BN].reshape([BM, BN])
            tm.tile_matmul(a[i * BM : i * BM + BM, :], b_packed[j], in_staging, staging.reshape(WORKSPACE), nr)
            for w, lane in dace.map[0:4, 0:32] @ dace.ScheduleType.GPU_ThreadBlock:
                for q in range(BM * BN // 128):
                    with dace.tasklet:
                        t << in_staging[(q * 128 + w * 32 + lane) // BN, (w * 32 + lane) % BN]
                        o >> c[TILES, i * BM + (q * 128 + w * 32 + lane) // BN, j * BN + (w * 32 + lane) % BN]
                        o = t
            for w, lane in dace.map[0:4, 0:32] @ dace.ScheduleType.GPU_ThreadBlock:
                for q in range(TILES * BM * BN // 128):
                    with dace.tasklet:
                        t << tile[q // 64, (q % 64 * 128 + w * 32 + lane) // BN, (w * 32 + lane) % BN]
                        o >> c[q // 64, i * BM + (q % 64 * 128 + w * 32 + lane) // BN, j * BN + (w * 32 + lane) % BN]
                        o = t

    a = torch.randn(M, K, device="cuda").to(torch.bfloat16)
    b_t = torch.randn(N, K, device="cuda").to(torch.bfloat16)
    b_packed = b_t.view(N // BN, BN, K // KS, KS).transpose(1, 2).contiguous()
    index = torch.randint(-1, M, (M,), dtype=torch.int32, device="cuda")  # -1: a missing row
    for implementation in tm.TileMatmul.implementations:
        sdfg = product.to_sdfg(simplify=True)
        for desc in sdfg.arrays.values():
            if not desc.transient:
                desc.storage = dace.StorageType.GPU_Global
        for node, _ in sdfg.all_nodes_recursive():
            if isinstance(node, dace.nodes.MapEntry) and node.map.params == ["i", "j"]:
                node.map.schedule = dace.ScheduleType.GPU_Device
            elif isinstance(node, tm.TileMatmul):
                node.implementation = implementation
        for rows in (BM, 5):
            c = torch.zeros(FORMS, M, N, device="cuda", dtype=torch.bfloat16)
            nrows = torch.tensor([rows], dtype=torch.int32, device="cuda")
            sdfg(a=a, b=b_t.T, b_packed=b_packed, index=index, rows=nrows, c=c)
            gathered = torch.where(index[:, None] >= 0, a[index.clamp(min=0).long()], 0)
            for form in range(FORMS):
                expected = ((gathered if form == 2 else a).float() @ b_t.float().T).view(M // BM, BM, N)
                expected[:, rows:] = 0
                expected = expected.view(M, N)
                error = ((c[form].float() - expected).norm() / expected.norm()).item()
                assert error < 1e-2, f"{implementation}, {rows} rows, form {form}: relative error {error:.2e}"
        print(f"{implementation}: correct")
