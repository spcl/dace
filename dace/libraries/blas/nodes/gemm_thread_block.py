# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Matrix products that one GPU thread block computes inside a kernel: the ``ThreadBlock`` and ``ThreadBlockMMA``
implementations of ``Gemm``.

A ``Gemm`` in device code that the threads of a block run together (see ``dace.sdfg.scope.is_thread_block_level``),
such as a task of a persistent kernel, is computed by the whole thread block; ``MatMul`` selects one of these
implementations for it (see ``thread_block_implementation``).

* ``ThreadBlockMMA``: tensor cores (``mma.sync`` m16n8k16) for bfloat16 or float16 operands with a constant number of
  columns ``N`` and inner dimension ``K``, ``K`` contiguous in both operands (as in ``x @ w.T`` for row-major ``x``
  and ``w``). The rows, any number of them (also data-dependent, as in ``x[r0:r0 + nr] @ w.T``), are computed in
  chunks of up to 64, the columns split among the warps of the block. Slices of ``K`` are copied asynchronously
  (``cp.async``, which also fetches the next slices into L2) into a workspace, a transient in dynamic shared memory,
  and loaded into tensor-core fragments (``ldmatrix``). The products accumulate in registers of every thread, which a
  second state of the expansion writes to the output: the workspace is then no longer in use, and may share memory
  with the output and with the containers of other code of the block (see ``SharedMemoryInterference``). Rows of ``a`` gathered through ``_a_index`` (see
  ``dace.libraries.blas.indexed``) are read directly, as are operands stored in blocks of columns along ``K`` (see
  ``dace.libraries.blas.blocked``, blocks of a multiple of 32 columns), whose slices are then contiguous. The
  tensor-core code needs CUDA compute capability 8.0; elsewhere (older GPUs, HIP), the same expansion multiplies element
  by element.
* ``ThreadBlock``: any types, shapes and layouts; every thread computes elements of the product.

Both write only the elements of the output (view), and support ``alpha`` but not ``beta``.
"""

import math

import dace
from dace import SDFG, InterstateEdge, Memlet, SDFGState, dtypes, nodes, symbolic
from dace.codegen.common import sym2cpp
from dace.config import Config
from dace.libraries.blas.blocked import operand_view
from dace.sdfg.scope import is_devicelevel_gpu, is_thread_block_level, kernel_scope
from dace.sdfg.state import LoopRegion
from dace.transformation.transformation import ExpandTransformation

WARP_SIZE = 32
#: The rows of a chunk of the tensor-core product, and the elements of a slice of the inner dimension
MMA_ROWS, MMA_K_STEP, MMA_STAGES = 64, 32, 2
#: The bytes around every copied chunk that L2 fetches with it: the following slices of a row-major operand, whose
#: rows a slice only reads 64 bytes of, are then read from L2 rather than from another DRAM page each time
MMA_L2_FETCH = 256


def kernel_threads(sdfg: SDFG, state: SDFGState, node: nodes.Node) -> int:
    """The threads of a block of the kernel around ``node``: the kernel's block size if it has one, otherwise the
    largest thread-block map (that is not strided over the block) in it, or the configured default."""
    found = kernel_scope(sdfg, state, node)
    if found is not None:
        kstate, kernel = found
        if isinstance(kernel, nodes.MapEntry) and kernel.map.gpu_block_size:
            return math.prod(int(s) for s in kernel.map.gpu_block_size)
        sizes = []
        for n in kstate.scope_subgraph(kernel).nodes():
            inner = n.sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG) else [(n, kstate)]
            for m, _ in inner:
                if (
                    isinstance(m, nodes.MapEntry)
                    and m.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock
                    and not m.map.gpu_strided
                ):
                    size = m.map.range.num_elements()
                    if not symbolic.issymbolic(size):
                        sizes.append(int(size))
        if sizes:
            return max(sizes)
    return math.prod(int(s) for s in Config.get("compiler", "cuda", "default_block_size").split(","))


def _operands(node: nodes.LibraryNode, state: SDFGState, sdfg: SDFG) -> dict[str, tuple[list, list, object]]:
    """The size and strides of every connector's view (see ``operand_view``), and the descriptor it views."""
    result = {}
    for e in state.in_edges(node) + state.out_edges(node):
        connector = e.dst_conn if e.dst is node else e.src_conn
        desc = sdfg.arrays[e.data.data]
        size, dims = operand_view(e.data.subset, connector in node.blocked)
        result[connector] = (list(size), [desc.strides[d] for d in dims], desc)
    return result


class _Operand:
    """An input of a thread-block product as its code reads it: the matrix ``op(x)`` (``x``, or ``x`` transposed),
    where ``x`` is the operand's view, stored in rows or in blocks of columns (see ``dace.libraries.blas.blocked``)."""

    def __init__(self, size: list, strides: list, transposed: bool, blocked: bool):
        self.strides, self.transposed, self.blocked = strides, transposed, blocked
        if blocked:
            blocks, rows, self.width = size
            stored = [rows, blocks * self.width]
        else:
            stored = list(size)
        self.shape = stored[::-1] if transposed else stored

    def _stored(self, r: str, c: str) -> str:
        """The offset of element ``(r, c)`` of ``x``."""
        s = [sym2cpp(stride) for stride in self.strides]
        if self.blocked:
            w = sym2cpp(self.width)
            return f"(({c}) / ({w})) * ({s[0]}) + ({r}) * ({s[1]}) + (({c}) % ({w})) * ({s[2]})"
        return f"({r}) * ({s[0]}) + ({c}) * ({s[1]})"

    def offset(self, i: str, j: str) -> str:
        """The offset of element ``(i, j)`` of ``op(x)``."""
        return self._stored(j, i) if self.transposed else self._stored(i, j)

    def _inner_is_columns(self, inner: int) -> bool:
        """Whether dimension ``inner`` of ``op(x)`` (the product's inner dimension) is that of the columns of ``x``."""
        return (inner == 1) != self.transposed

    def slices_contiguous(self, inner: int) -> bool:
        """Whether the slices of ``MMA_K_STEP`` elements of dimension ``inner`` of ``op(x)`` are contiguous."""
        if self.blocked:
            return (
                self._inner_is_columns(inner)
                and symbolic.equal(self.strides[2], 1) is True
                and not symbolic.issymbolic(self.width)
                and int(self.width) % MMA_K_STEP == 0
            )
        strides = self.strides[::-1] if self.transposed else self.strides
        return symbolic.equal(strides[inner], 1) is True

    def outer_offset(self, x: str, inner: int) -> str:
        """The offset of index ``x`` of the other dimension of ``op(x)`` than ``inner``, at the start of the inner one."""
        return self.offset(x, "0") if inner == 1 else self.offset("0", x)

    def slice_offset(self, k: str) -> str:
        """The offset of the slice at ``k`` of the inner dimension, a multiple of ``MMA_K_STEP`` (from
        ``outer_offset``; see ``slices_contiguous``)."""
        if self.blocked:
            w, s0 = sym2cpp(self.width), sym2cpp(self.strides[0])
            return f"(({k}) / ({w})) * ({s0}) + ({k}) % ({w})"
        return f"({k})"


class _Product:
    """The shape of a thread-block product: ``c[M, N] = alpha * op(a)[M, K] @ op(b)[K, N]``, with its inputs (see
    ``_Operand``) and the strides of ``c`` (row, column)."""

    def __init__(self, node, operands: dict):
        if node.beta != 0:
            raise ValueError(f'The thread-block implementations of "{node.label}" do not support beta')
        if "_b_index" in operands or "_c_index" in operands:
            raise ValueError(f'The thread-block implementations of "{node.label}" only take an index of a')
        (a, sa, adesc), (b, sb, bdesc), (c, sc, cdesc) = (operands[k] for k in ("_a", "_b", "_c"))
        self.gather = "_a_index" in operands
        if self.gather and node.transA:
            raise ValueError(f'"{node.label}": a gathered operand cannot be transposed')
        self.a = _Operand(a, sa, node.transA, "_a" in node.blocked)
        self.b = _Operand(b, sb, node.transB, "_b" in node.blocked)
        self.m, self.n, self.k = c[0], c[1], self.a.shape[1]
        self.sc = sc
        self.types = adesc.dtype, bdesc.dtype, cdesc.dtype
        self.alpha = node.alpha


def _check_context(node: nodes.LibraryNode, state: SDFGState, sdfg: SDFG) -> None:
    if not is_devicelevel_gpu(sdfg, state, node) or not is_thread_block_level(sdfg, state, node):
        raise ValueError(
            f'"{node.label}" is computed by a thread block, so it must be in a GPU kernel, outside thread-block maps'
        )
    for e in state.out_edges(node):
        if e.data.wcr is not None:
            raise ValueError(f'The thread-block implementations of "{node.label}" do not write with conflicts')


def mma_ineligibility(node: nodes.LibraryNode, state: SDFGState, sdfg: SDFG) -> str | None:
    """Why the tensor-core implementation does not apply to a matrix product (a ``Gemm``, or a ``MatMul`` that
    specializes to one), or None if it applies."""
    try:
        product = _Product(node, _operands(node, state, sdfg))
    except ValueError as ex:
        return str(ex)
    a, b, _ = product.types
    if a != b or a not in (dace.bfloat16, dace.float16):
        return f"the operands are {a} and {b}, not both bfloat16 or float16"
    for value, what in ((product.n, "columns"), (product.k, "inner dimension")):
        if symbolic.issymbolic(value, sdfg.constants):
            return f"the {what} ({value}) is not a constant"
    n, k = (int(symbolic.evaluate(v, sdfg.constants)) for v in (product.n, product.k))
    if not product.a.slices_contiguous(1) or not product.b.slices_contiguous(0):
        return (
            f"the inner dimension is not contiguous in both operands (in blocks of columns, of a multiple of "
            f"{MMA_K_STEP})"
        )
    warps = kernel_threads(sdfg, state, node) // WARP_SIZE
    if k % MMA_K_STEP or warps == 0 or n % warps or (n // warps) % 16:
        return f"a {n}-column, {k}-deep product does not split among {warps} warps in slices of {MMA_K_STEP}"
    return None


def thread_block_implementation(node: nodes.LibraryNode, state: SDFGState, sdfg: SDFG) -> str | None:
    """The implementation for a matrix product in device code that a thread block computes, or None elsewhere."""
    if not is_devicelevel_gpu(sdfg, state, node) or not is_thread_block_level(sdfg, state, node):
        return None
    if "_b_index" in node.in_connectors or "_c_index" in node.in_connectors or node.beta != 0:
        return None
    return "ThreadBlockMMA" if mma_ineligibility(node, state, sdfg) is None else "ThreadBlock"


def _rows_bound(node: nodes.LibraryNode, state: SDFGState, sdfg: SDFG) -> int | float:
    """An upper bound of the rows of a product: those of its output, or of the container its output is in (infinite
    if not a constant)."""
    for e in state.out_edges(node):
        if e.src_conn == "_c":
            size, dims = operand_view(e.data.subset, False)
            for rows in (size[0], sdfg.arrays[e.data.data].shape[dims[0]]):
                if not symbolic.issymbolic(rows, sdfg.constants):
                    return int(symbolic.evaluate(rows, sdfg.constants))
    return math.inf


def _nested(node, operands: dict, phases: list, threads: int, scratch: dict, chunks: tuple | None = None) -> SDFG:
    """A nested SDFG that runs ``phases`` one after the other, each a state with a thread-block map of ``threads``
    threads. A phase is ``(name, code, inputs, outputs)``, a C++ tasklet in the map, where ``inputs`` and ``outputs``
    map the tasklet's connectors to the containers they access: the operands (``_a``, ``_b``, ``_a_index``, ``_c``)
    and the transients in ``scratch`` (name to shape, type and storage). A phase may also be ``(name, build)``, where
    ``build(state, entry, exit_)`` fills the map. With ``chunks``, a pair of the number of rows and
    the rows of a chunk, the phases are the body of a loop over the first row of every chunk, ``m0``."""
    sdfg = SDFG(f"{node.label}_sdfg")
    for connector, (size, strides, desc) in operands.items():
        if connector not in sdfg.arrays:
            sdfg.add_array(connector, size, desc.dtype, storage=desc.storage, strides=strides)
    for name, (shape, dtype, storage) in scratch.items():
        sdfg.add_transient(name, shape, dtype, storage=storage)
    region = sdfg
    if chunks is not None:
        rows, chunk = chunks
        sdfg.add_symbol("m0", dtypes.int64)
        region = LoopRegion(f"{node.label}_chunks", f"m0 < {rows}", "m0", "m0 = 0", f"m0 = m0 + {chunk}", sdfg=sdfg)
        sdfg.add_node(region, is_start_block=True)
    previous = None
    for phase in phases:
        label = node.label if len(phases) == 1 else f"{node.label}_{phase[0]}"
        state = region.add_state(label, is_start_block=previous is None)
        if previous is not None:
            region.add_edge(previous, state, InterstateEdge())
        previous = state
        entry, exit_ = state.add_map(
            f"{label}_threads", {"tid": f"0:{threads}"}, schedule=dtypes.ScheduleType.GPU_ThreadBlock
        )
        if len(phase) == 2:
            phase[1](state, entry, exit_)
            continue
        _, code, inputs, outputs = phase
        tasklet = state.add_tasklet(f"{label}_tasklet", set(inputs), set(outputs), code, language=dtypes.Language.CPP)
        for connector, name in inputs.items():
            memlet = Memlet.from_array(name, sdfg.arrays[name])
            state.add_memlet_path(state.add_read(name), entry, tasklet, dst_conn=connector, memlet=memlet)
        for connector, name in outputs.items():
            memlet = Memlet.from_array(name, sdfg.arrays[name])
            memlet.dynamic = True
            state.add_memlet_path(tasklet, exit_, state.add_write(name), src_conn=connector, memlet=memlet)
    return sdfg


def _inputs(product: "_Product") -> dict[str, str]:
    """The input connectors of the tasklets that read the operands, and the operands they read."""
    return {"a": "_a", "b": "_b", **({"a_index": "_a_index"} if product.gather else {})}


def _elementwise(product: _Product, threads: int) -> str:
    """C++ code in which every thread computes elements of the product (strided over the block)."""
    acc = "double" if dtypes.float64 in product.types else "float"
    sc = [sym2cpp(s) for s in product.sc]
    row = "a_index[i]" if product.gather else "i"
    scale = "" if product.alpha == 1 else f"({acc})({sym2cpp(product.alpha)}) * "
    return f"""
for (long long idx = tid; idx < (long long)({sym2cpp(product.m)}) * ({sym2cpp(product.n)}); idx += {threads}) {{
    const long long i = idx / ({sym2cpp(product.n)}), j = idx % ({sym2cpp(product.n)});
    const long long row = {row};
    {acc} sum = 0;
    for (long long kk = 0; kk < {sym2cpp(product.k)}; ++kk)
        sum += ({acc}) a[{product.a.offset("row", "kk")}] * ({acc}) b[{product.b.offset("kk", "j")}];
    c[i * ({sc[0]}) + j * ({sc[1]})] = ({product.types[2].ctype})({scale}sum);
}}
"""


@dace.library.expansion
class ExpandGemmThreadBlock(ExpandTransformation):
    """Every thread of the block computes elements of the product, for any types and shapes."""

    environments = []

    @staticmethod
    def expansion(node, state: SDFGState, sdfg: SDFG):
        _check_context(node, state, sdfg)
        operands = _operands(node, state, sdfg)
        product = _Product(node, operands)
        threads = kernel_threads(sdfg, state, node)
        code = _elementwise(product, threads) + "__syncthreads();  // The product is complete for every thread\n"
        return _nested(node, operands, [("product", code, _inputs(product), {"c": "_c"})], threads, {})


class _MMACode:
    """The C++ code of the tensor-core product of a chunk of rows starting at ``m0`` (a variable of the code): its
    ``compute`` part, which copies slices of the operands into the workspace and multiplies them into the per-thread
    accumulators, and its ``store`` part, which writes the accumulators to ``c``. Both expect the declarations of
    ``header`` before them, and the accumulators as the array ``acc`` (a fragment of every warp, see ``ACC``)."""

    def __init__(self, product: _Product, threads: int, rows: int, n: int, k: int):
        self.product, self.threads = product, threads
        warps = threads // WARP_SIZE
        self.operand, self.ctype = product.types[0], product.types[2].ctype
        self.header = f"""
constexpr int MT = {rows}, N = {n}, K = {k}, KS = {MMA_K_STEP}, STAGES = {MMA_STAGES}, CHUNKS = {MMA_K_STEP // 8};
constexpr int THREADS = {threads}, WN = N / {warps}, MF = MT / 16, NF = WN / 8;
const long long M_TOTAL = {sym2cpp(product.m)};
const int w = tid / {WARP_SIZE}, lane = tid % {WARP_SIZE};
// Accumulator e of fragment (i, j) of the thread: row i * 16 + lane / 4 + (e / 2) * 8 and column
// w * WN + j * 8 + 2 * (lane % 4) + e % 2 of the chunk
#define DACE_TB_ACC(i, j, e) acc[((i) * NF + (j)) * 4 + (e)]
"""
        self.footer = "#undef DACE_TB_ACC\n"

    @staticmethod
    def accumulators(rows: int, n: int, threads: int) -> int:
        """The number of accumulators of every thread (``MF * NF * 4`` in the constants of ``header``)."""
        return (rows // 16) * (n // (threads // WARP_SIZE) // 8) * 4

    def compute(self) -> str:
        product, operand = self.product, self.operand
        mma_type = "bf16" if operand == dace.bfloat16 else "f16"
        chunks = MMA_K_STEP // 8
        # Chunks of 16 bytes are swizzled (XOR with the row) so that the eight rows ``ldmatrix`` reads at once, and the
        # rows ``cp.async`` writes, hit distinct banks
        swizzle = "((r) >> 1) & 3" if chunks == 4 else "(r) & 7"
        # Where the slices of a row of op(a) and of a column of op(b) start, and how far a slice at k0 is from them
        a_row = "a_index[m0 + r]" if product.gather else "(m0 + r)"
        a_start, b_start = product.a.outer_offset(f"(long long){a_row}", 1), product.b.outer_offset("r", 0)
        a_slice, b_slice = product.a.slice_offset("k0"), product.b.slice_offset("k0")
        fallback_row = "(long long)a_index[m0 + row]" if product.gather else "(m0 + row)"
        return f"""{{
const int nrows = (int) min((long long) MT, M_TOTAL - m0);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
const uint32_t smem = (uint32_t) __cvta_generic_to_shared(workspace);
#define DACE_TB_SWZ(r, c) ((c) ^ ({swizzle}))
#define DACE_TB_ADDR(s, r, c) (smem + ((((s) * (MT + N) + (r)) * CHUNKS + DACE_TB_SWZ(r, c)) << 4))
constexpr int A_COPIES = (MT * CHUNKS + THREADS - 1) / THREADS, B_COPIES = (N * CHUNKS + THREADS - 1) / THREADS;
// The chunks of every slice this thread copies: where they start in A (null for rows beyond the product) and B
const {operand.ctype} *a_src[A_COPIES];
const {operand.ctype} *b_src[B_COPIES];
#pragma unroll
for (int j = 0; j < A_COPIES; ++j) {{
    const int i = tid + j * THREADS, r = i / CHUNKS, c = i % CHUNKS;
    a_src[j] = (i < MT * CHUNKS && r < nrows) ? a + {a_start} + c * 8 : nullptr;
}}
#pragma unroll
for (int j = 0; j < B_COPIES; ++j) {{
    const int i = tid + j * THREADS, r = i / CHUNKS, c = i % CHUNKS;
    b_src[j] = b + {b_start} + c * 8;
}}
// Copies the slices of A and B starting at k0 into stage s (rows of A beyond the product are zero-filled)
auto load_stage = [&](int s, int k0) {{
    #pragma unroll
    for (int j = 0; j < A_COPIES; ++j) {{
        const int i = tid + j * THREADS, r = i / CHUNKS, c = i % CHUNKS;
        if (MT * CHUNKS % THREADS == 0 || i < MT * CHUNKS)
            asm volatile("cp.async.cg.shared.global.L2::{MMA_L2_FETCH}B [%0], [%1], 16, %2;\\n"
                         :: "r"(DACE_TB_ADDR(s, r, c)), "l"(a_src[j] ? a_src[j] + {a_slice} : a), "r"(a_src[j] ? 16 : 0));
    }}
    #pragma unroll
    for (int j = 0; j < B_COPIES; ++j) {{
        const int i = tid + j * THREADS, r = i / CHUNKS, c = i % CHUNKS;
        if (N * CHUNKS % THREADS == 0 || i < N * CHUNKS)
            asm volatile("cp.async.cg.shared.global.L2::{MMA_L2_FETCH}B [%0], [%1], 16;\\n"
                         :: "r"(DACE_TB_ADDR(s, MT + r, c)), "l"(b_src[j] + {b_slice}));
    }}
}};

#pragma unroll
for (int i = 0; i < MF; ++i)
    #pragma unroll
    for (int j = 0; j < NF; ++j)
        DACE_TB_ACC(i, j, 0) = DACE_TB_ACC(i, j, 1) = DACE_TB_ACC(i, j, 2) = DACE_TB_ACC(i, j, 3) = 0.f;

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
            const int r = MT + w * WN + j * 16 + (lane >> 4) * 8 + (lane & 7);
            const int c = kk * 2 + ((lane >> 3) & 1);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%0, %1, %2, %3}}, [%4];\\n"
                         : "=r"(bf[2 * j][0]), "=r"(bf[2 * j][1]), "=r"(bf[2 * j + 1][0]), "=r"(bf[2 * j + 1][1])
                         : "r"(DACE_TB_ADDR(s, r, c)));
        }}
        #pragma unroll
        for (int i = 0; i < MF; ++i) {{
            if (i * 16 >= nrows) break;
            uint32_t af[4];
            const int r = i * 16 + (lane & 15);
            const int c = kk * 2 + (lane >> 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%0, %1, %2, %3}}, [%4];\\n"
                         : "=r"(af[0]), "=r"(af[1]), "=r"(af[2]), "=r"(af[3])
                         : "r"(DACE_TB_ADDR(s, r, c)));
            #pragma unroll
            for (int j = 0; j < NF; ++j)
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.{mma_type}.{mma_type}.f32 "
                             "{{%0, %1, %2, %3}}, {{%4, %5, %6, %7}}, {{%8, %9}}, {{%0, %1, %2, %3}};\\n"
                             : "+f"(DACE_TB_ACC(i, j, 0)), "+f"(DACE_TB_ACC(i, j, 1)), "+f"(DACE_TB_ACC(i, j, 2)),
                               "+f"(DACE_TB_ACC(i, j, 3))
                             : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]), "r"(bf[j][0]), "r"(bf[j][1]));
        }}
    }}
}}
asm volatile("cp.async.wait_group 0;\\n" ::);
#undef DACE_TB_ADDR
#undef DACE_TB_SWZ
#else  // Without the tensor-core instructions: every accumulator of the thread element by element
#pragma unroll
for (int i = 0; i < MF; ++i)
    #pragma unroll
    for (int j = 0; j < NF; ++j)
        #pragma unroll
        for (int e = 0; e < 4; ++e) {{
            const int row = i * 16 + (lane >> 2) + (e >> 1) * 8, col = w * WN + j * 8 + (lane & 3) * 2 + (e & 1);
            float sum = 0.f;
            if (row < nrows) {{
                const long long ar = {fallback_row};
                for (int kk = 0; kk < K; ++kk)
                    sum += (float) a[{product.a.offset("ar", "kk")}] * (float) b[{product.b.offset("kk", "col")}];
            }}
            DACE_TB_ACC(i, j, e) = sum;
        }}
#endif
}}
"""

    def store(self) -> str:
        product = self.product
        sc0, sc1 = (sym2cpp(s) for s in product.sc)
        scale = "" if product.alpha == 1 else f"(float)({sym2cpp(product.alpha)}) * "
        return f"""{{
const int nrows = (int) min((long long) MT, M_TOTAL - m0);
#pragma unroll
for (int i = 0; i < MF; ++i)
    #pragma unroll
    for (int j = 0; j < NF; ++j)
        #pragma unroll
        for (int h = 0; h < 2; ++h) {{
            const int row = i * 16 + (lane >> 2) + h * 8, col = w * WN + j * 8 + (lane & 3) * 2;
            if (row < nrows) {{
                {self.ctype} *out = c + (m0 + row) * ({sc0});
                out[col * ({sc1})] = ({self.ctype})({scale}DACE_TB_ACC(i, j, 2 * h));
                out[(col + 1) * ({sc1})] = ({self.ctype})({scale}DACE_TB_ACC(i, j, 2 * h + 1));
            }}
        }}
}}
"""


def _store_accumulators(product: _Product, threads: int, rows: int, n: int, m0: str):
    """A phase (see ``_nested``) that writes the accumulators of every thread (see ``_MMACode``) to ``c``: the
    elements of the chunk of rows starting at ``m0`` that are rows of the product."""
    warps = threads // WARP_SIZE
    wn = n // warps
    row = f"{m0} + i * 16 + (tid % {WARP_SIZE}) // 4 + (e // 2) * 8"
    column = f"(tid // {WARP_SIZE}) * {wn} + j * 8 + (tid % 4) * 2 + e % 2"
    scale = "" if product.alpha == 1 else f"{symbolic.symstr(product.alpha)} * "
    code = f"if {row} < {symbolic.symstr(product.m)}:\n    out = {scale}value"

    def build(state: SDFGState, entry: nodes.MapEntry, exit_: nodes.MapExit):
        fragment_entry, fragment_exit = state.add_map(
            "fragment", {"i": f"0:{rows // 16}", "j": f"0:{wn // 8}", "e": "0:4"}, unroll=True
        )
        tasklet = state.add_tasklet("store", {"value"}, {"out"}, code)
        value = Memlet(f"_acc[(i * {wn // 8} + j) * 4 + e]")
        state.add_memlet_path(state.add_read("_acc"), entry, fragment_entry, tasklet, dst_conn="value", memlet=value)
        out = Memlet(f"_c[{row}, {column}]", dynamic=True)
        state.add_memlet_path(tasklet, fragment_exit, exit_, state.add_write("_c"), src_conn="out", memlet=out)

    return ("store", build)


def _runs_states_as_block(sdfg: SDFG, state: SDFGState, node: nodes.Node) -> bool:
    """Whether the states of a nested SDFG at ``node`` run as states of the thread block's code, each starting with a
    barrier if it accesses shared memory and ending with one if it runs a thread-block map: in a ``GPU_Device``
    kernel (a ``GPU_Persistent`` one synchronizes its states differently)."""
    found = kernel_scope(sdfg, state, node)
    return found is not None and found[1].schedule == dtypes.ScheduleType.GPU_Device


@dace.library.expansion
class ExpandGemmThreadBlockMMA(ExpandTransformation):
    """A software-pipelined tensor-core product (see the module documentation)."""

    environments = []

    @staticmethod
    def expansion(node, state: SDFGState, sdfg: SDFG):
        _check_context(node, state, sdfg)
        reason = mma_ineligibility(node, state, sdfg)
        if reason is not None:
            raise ValueError(f'The ThreadBlockMMA implementation does not apply to "{node.label}": {reason}')
        operands = _operands(node, state, sdfg)
        product = _Product(node, operands)
        threads = kernel_threads(sdfg, state, node)
        n, k = (int(symbolic.evaluate(v, sdfg.constants)) for v in (product.n, product.k))
        m = product.m
        rows = MMA_ROWS
        if not symbolic.issymbolic(m, sdfg.constants):
            rows = min(MMA_ROWS, -(-int(symbolic.evaluate(m, sdfg.constants)) // 16) * 16)
        code = _MMACode(product, threads, rows, n, k)
        accumulators = _MMACode.accumulators(rows, n, threads)
        # A single chunk of rows if there are at most those of the output's container
        single_chunk = _rows_bound(node, state, sdfg) <= MMA_ROWS
        storage = dtypes.StorageType.GPU_Shared(dynamic=True)
        scratch = {"_workspace": ([MMA_STAGES, rows + n, MMA_K_STEP], product.types[0], storage)}
        inputs = _inputs(product)
        if _runs_states_as_block(sdfg, state, node):
            # Two states: the copies and products, then the stores, with the accumulators of every thread in between,
            # in a loop over the chunks of rows unless there is one. The states synchronize the block where the
            # tasklets would have to (before the workspace is used, between the last use of the workspace and the first
            # write of c, and once c is complete), and show that the workspace is no longer used once c is written: it
            # may share memory with c (see ``SharedMemoryInterference``).
            scratch["_acc"] = ([accumulators], dace.float32, dtypes.StorageType.Register)
            first = "const long long m0 = 0;\n" if single_chunk else ""  # The first row of the chunk
            chunks = None if single_chunk else (f"({product.m})", rows)
            phases = [
                (
                    "compute",
                    code.header + first + code.compute() + code.footer,
                    inputs,
                    {"workspace": "_workspace", "acc": "_acc"},
                ),
                _store_accumulators(product, threads, rows, n, "0" if single_chunk else "m0"),
            ]
            return _nested(node, operands, phases, threads, scratch, chunks)
        else:
            # Chunks of rows in a loop of one tasklet, which synchronizes the block itself (in a persistent kernel)
            body = f"""float acc[MF * NF * 4];
for (long long m0 = 0; m0 < M_TOTAL; m0 += MT) {{
__syncthreads();  // The previous user of the workspace is done with it
{code.compute()}{code.store()}}}
__syncthreads();  // The product is complete for every thread of the block
"""
            phases = [("product", code.header + body + code.footer, inputs, {"c": "_c", "workspace": "_workspace"})]
        return _nested(node, operands, phases, threads, scratch)
