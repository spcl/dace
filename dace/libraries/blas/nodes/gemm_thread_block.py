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
  (``cp.async``, which also fetches the next slices into L2) into a workspace, a transient in dynamic shared memory
  (which the products in other states of the block's code share, see ``PlanSharedMemory``), and loaded into
  tensor-core fragments (``ldmatrix``). Rows of ``a`` gathered through ``_a_index`` (see
  ``dace.libraries.blas.indexed``) are read directly. The tensor-core code needs CUDA compute capability 8.0;
  elsewhere (older GPUs, HIP), the same expansion multiplies element by element.
* ``ThreadBlock``: any types and shapes; every thread computes elements of the product.

Both write only the elements of the output (view), and support ``alpha`` but not ``beta``.
"""

import math

import dace
from dace import SDFG, Memlet, SDFGState, dtypes, nodes, symbolic
from dace.codegen.common import sym2cpp
from dace.config import Config
from dace.libraries.blas.blas_helpers import matrix_view
from dace.sdfg.scope import is_devicelevel_gpu, is_thread_block_level, kernel_scope
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
    """The size and strides of every connector's view (see ``matrix_view``), and the descriptor it views."""
    result = {}
    for e in state.in_edges(node) + state.out_edges(node):
        connector = e.dst_conn if e.dst is node else e.src_conn
        desc = sdfg.arrays[e.data.data]
        size, dims = matrix_view(e.data.subset)
        result[connector] = (list(size), [desc.strides[d] for d in dims], desc)
    return result


class _Product:
    """The shape of a thread-block product: ``c[M, N] = alpha * op(a)[M, K] @ op(b)[K, N]``, with the strides of
    ``op(a)`` (row, inner), ``op(b)`` (inner, column) and ``c`` (row, column)."""

    def __init__(self, node, operands: dict):
        if node.beta != 0:
            raise ValueError(f'The thread-block implementations of "{node.label}" do not support beta')
        if "_b_index" in operands or "_c_index" in operands:
            raise ValueError(f'The thread-block implementations of "{node.label}" only take an index of a')
        (a, sa, adesc), (b, sb, bdesc), (c, sc, cdesc) = (operands[k] for k in ("_a", "_b", "_c"))
        self.gather = "_a_index" in operands
        if self.gather and node.transA:
            raise ValueError(f'"{node.label}": a gathered operand cannot be transposed')
        if node.transA:
            a, sa = a[::-1], sa[::-1]
        if node.transB:
            b, sb = b[::-1], sb[::-1]
        self.m, self.n, self.k = c[0], c[1], a[1]
        self.sa, self.sb, self.sc = sa, sb, sc
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
    if symbolic.equal(product.sa[1], 1) is not True or symbolic.equal(product.sb[0], 1) is not True:
        return "the inner dimension is not contiguous in both operands"
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


def _nested(node, operands: dict, product: _Product, code: str, threads: int, scratch: dict) -> SDFG:
    """A nested SDFG of one thread-block map of ``threads`` threads, around a C++ tasklet with ``code`` that reads the
    operands (``a``, ``b``, ``a_index``) and writes ``c`` and the transients in ``scratch``."""
    sdfg = SDFG(f"{node.label}_sdfg")
    for connector, (size, strides, desc) in operands.items():
        if connector not in sdfg.arrays:
            sdfg.add_array(connector, size, desc.dtype, storage=desc.storage, strides=strides)
    for name, (shape, dtype, storage) in scratch.items():
        sdfg.add_transient(name, shape, dtype, storage=storage)
    state = sdfg.add_state(node.label)
    entry, exit_ = state.add_map(
        f"{node.label}_threads", {"tid": f"0:{threads}"}, schedule=dtypes.ScheduleType.GPU_ThreadBlock
    )
    inputs = {"a": "_a", "b": "_b", **({"a_index": "_a_index"} if product.gather else {})}
    outputs = {"c": "_c", **{name.lstrip("_"): name for name in scratch}}
    tasklet = state.add_tasklet(f"{node.label}_product", set(inputs), set(outputs), code, language=dtypes.Language.CPP)
    for connector, name in inputs.items():
        memlet = Memlet.from_array(name, sdfg.arrays[name])
        state.add_memlet_path(state.add_read(name), entry, tasklet, dst_conn=connector, memlet=memlet)
    for connector, name in outputs.items():
        memlet = Memlet.from_array(name, sdfg.arrays[name])
        memlet.dynamic = True
        state.add_memlet_path(tasklet, exit_, state.add_write(name), src_conn=connector, memlet=memlet)
    return sdfg


def _elementwise(product: _Product, threads: int) -> str:
    """C++ code in which every thread computes elements of the product (strided over the block)."""
    acc = "double" if dtypes.float64 in product.types else "float"
    sa, sb, sc = ([sym2cpp(s) for s in strides] for strides in (product.sa, product.sb, product.sc))
    row = "a_index[i]" if product.gather else "i"
    scale = "" if product.alpha == 1 else f"({acc})({sym2cpp(product.alpha)}) * "
    return f"""
for (long long idx = tid; idx < (long long)({sym2cpp(product.m)}) * ({sym2cpp(product.n)}); idx += {threads}) {{
    const long long i = idx / ({sym2cpp(product.n)}), j = idx % ({sym2cpp(product.n)});
    const long long row = {row};
    {acc} sum = 0;
    for (long long kk = 0; kk < {sym2cpp(product.k)}; ++kk)
        sum += ({acc}) a[row * ({sa[0]}) + kk * ({sa[1]})] * ({acc}) b[kk * ({sb[0]}) + j * ({sb[1]})];
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
        return _nested(node, operands, product, code, threads, {})


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
        warps = threads // WARP_SIZE
        n, k = (int(symbolic.evaluate(v, sdfg.constants)) for v in (product.n, product.k))
        m = product.m
        rows = MMA_ROWS
        if not symbolic.issymbolic(m, sdfg.constants):
            rows = min(MMA_ROWS, -(-int(symbolic.evaluate(m, sdfg.constants)) // 16) * 16)
        operand, ctype = product.types[0], product.types[2].ctype
        mma_type = "bf16" if operand == dace.bfloat16 else "f16"
        chunks = MMA_K_STEP // 8
        # Chunks of 16 bytes are swizzled (XOR with the row) so that the eight rows ``ldmatrix`` reads at once, and the
        # rows ``cp.async`` writes, hit distinct banks
        swizzle = "((r) >> 1) & 3" if chunks == 4 else "(r) & 7"
        sa0, sb1 = sym2cpp(product.sa[0]), sym2cpp(product.sb[1])
        sc0, sc1 = (sym2cpp(s) for s in product.sc)
        a_row = "a_index[m0 + r]" if product.gather else "(m0 + r)"
        scale = "" if product.alpha == 1 else f"(float)({sym2cpp(product.alpha)}) * "
        code = f"""
constexpr int MT = {rows}, N = {n}, K = {k}, KS = {MMA_K_STEP}, STAGES = {MMA_STAGES}, CHUNKS = {chunks};
constexpr int THREADS = {threads}, WN = N / {warps}, MF = MT / 16, NF = WN / 8;
const long long M_TOTAL = {sym2cpp(m)};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
const int w = tid / {WARP_SIZE}, lane = tid % {WARP_SIZE};
const uint32_t smem = (uint32_t) __cvta_generic_to_shared(workspace);
#define DACE_TB_SWZ(r, c) ((c) ^ ({swizzle}))
#define DACE_TB_ADDR(s, r, c) (smem + ((((s) * (MT + N) + (r)) * CHUNKS + DACE_TB_SWZ(r, c)) << 4))
constexpr int A_COPIES = (MT * CHUNKS + THREADS - 1) / THREADS, B_COPIES = (N * CHUNKS + THREADS - 1) / THREADS;
for (long long m0 = 0; m0 < M_TOTAL; m0 += MT) {{
    const int nrows = (int) min((long long) MT, M_TOTAL - m0);
    // The chunks of every slice this thread copies: where they start in A (null for rows beyond the product) and B
    const {operand.ctype} *a_src[A_COPIES];
    const {operand.ctype} *b_src[B_COPIES];
    #pragma unroll
    for (int j = 0; j < A_COPIES; ++j) {{
        const int i = tid + j * THREADS, r = i / CHUNKS, c = i % CHUNKS;
        a_src[j] = (i < MT * CHUNKS && r < nrows) ? a + (long long)({a_row}) * ({sa0}) + c * 8 : nullptr;
    }}
    #pragma unroll
    for (int j = 0; j < B_COPIES; ++j) {{
        const int i = tid + j * THREADS, r = i / CHUNKS, c = i % CHUNKS;
        b_src[j] = b + r * ({sb1}) + c * 8;
    }}
    // Copies the slices of A and B starting at k0 into stage s (rows of A beyond the product are zero-filled)
    auto load_stage = [&](int s, int k0) {{
        #pragma unroll
        for (int j = 0; j < A_COPIES; ++j) {{
            const int i = tid + j * THREADS, r = i / CHUNKS, c = i % CHUNKS;
            if (MT * CHUNKS % THREADS == 0 || i < MT * CHUNKS)
                asm volatile("cp.async.cg.shared.global.L2::{MMA_L2_FETCH}B [%0], [%1], 16, %2;\\n"
                             :: "r"(DACE_TB_ADDR(s, r, c)), "l"(a_src[j] ? a_src[j] + k0 : a), "r"(a_src[j] ? 16 : 0));
        }}
        #pragma unroll
        for (int j = 0; j < B_COPIES; ++j) {{
            const int i = tid + j * THREADS, r = i / CHUNKS, c = i % CHUNKS;
            if (N * CHUNKS % THREADS == 0 || i < N * CHUNKS)
                asm volatile("cp.async.cg.shared.global.L2::{MMA_L2_FETCH}B [%0], [%1], 16;\\n"
                             :: "r"(DACE_TB_ADDR(s, MT + r, c)), "l"(b_src[j] + k0));
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
                                 : "+f"(acc[i][j][0]), "+f"(acc[i][j][1]), "+f"(acc[i][j][2]), "+f"(acc[i][j][3])
                                 : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]), "r"(bf[j][0]), "r"(bf[j][1]));
            }}
        }}
    }}
    asm volatile("cp.async.wait_group 0;\\n" ::);

    // Accumulator fragment: rows lane / 4 and lane / 4 + 8, columns 2 * (lane % 4) and the next
    #pragma unroll
    for (int i = 0; i < MF; ++i)
        #pragma unroll
        for (int j = 0; j < NF; ++j)
            #pragma unroll
            for (int h = 0; h < 2; ++h) {{
                const int row = i * 16 + (lane >> 2) + h * 8, col = w * WN + j * 8 + (lane & 3) * 2;
                if (row < nrows) {{
                    {ctype} *out = c + (m0 + row) * ({sc0});
                    out[col * ({sc1})] = ({ctype})({scale}acc[i][j][2 * h]);
                    out[(col + 1) * ({sc1})] = ({ctype})({scale}acc[i][j][2 * h + 1]);
                }}
            }}
}}
#undef DACE_TB_ADDR
#undef DACE_TB_SWZ
#else  // Without the tensor-core instructions
{_elementwise(product, threads)}
#endif
__syncthreads();  // The product is complete for every thread of the block
"""
        workspace = ([MMA_STAGES, rows + n, MMA_K_STEP], operand, dtypes.StorageType.GPU_Shared(dynamic=True))
        return _nested(node, operands, product, code, threads, {"_workspace": workspace})
