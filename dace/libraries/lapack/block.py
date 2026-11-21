# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""In-kernel LAPACK: ONE thread block factors or solves ONE small matrix.

Neither CUDA nor HIP has a block-level solver, so the ``'CUDA (block)'`` expansion of every LAPACK and dense
``linalg`` node is one tasklet calling the unblocked kernels of ``dace/cuda/block_lapack.cuh`` under a lane map
(:func:`~dace.libraries.standard.block_reduce.add_lane_tasklet`). Registering it is what lets the GPU offloader
put the node inside a kernel -- a batch of small factorizations under a parallel map, one per block. Matrices are
read through their strides as the CPU (row-major LAPACKE) expansions read them, not column-major as the vendor GPU
solvers do. Scratch a node needs beyond its operands is static shared memory, so it needs a literal order.
"""

from typing import Any, NamedTuple

import numpy as np

from dace import SDFG, SDFGState, data, dtypes, symbolic
from dace.codegen.common import global_code_id, sym2cpp
from dace.libraries.lapack.environments.block_lapack import BlockLapack
from dace.libraries.standard.block_reduce import BLOCK_COLLECTIVE_THREADS, add_lane_tasklet
from dace.libraries.standard.environments.cuda import CUDA
from dace.libraries.standard.pure_components import operand_view
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation

#: The implementation key of every in-kernel LAPACK expansion.
BLOCK_LAPACK = "CUDA (block)"

#: Static shared memory one node may claim for scratch; under the 48 KiB every GPU grants without opt-in.
SHARED_SCRATCH_BYTES = 32 * 1024

#: Rows a lane keeps in registers in ``getri`` (``block_lapack::ROWS_PER_LANE``).
ROWS_PER_LANE = 4

#: Cyclic Jacobi sweeps before ``syevj`` stops, as the ``pure`` ``Eigh`` expansion.
MAX_JACOBI_SWEEPS = 100

REAL_TYPES = (dtypes.float32, dtypes.float64)


class Operand(NamedTuple):
    """One connector of the node as the block tasklet sees it."""

    desc: data.Data
    shape: list
    strides: list
    is_input: bool

    @property
    def ctype(self) -> str:
        return self.desc.dtype.base_type.ctype


def operands(node: nodes.LibraryNode, state: SDFGState, sdfg: SDFG) -> dict[str, Operand]:
    """The node's connected data connectors, inputs first; an in/out connector name appears once, as an input."""
    found: dict[str, Operand] = {}
    for edge, is_input in [(e, True) for e in state.in_edges(node)] + [(e, False) for e in state.out_edges(node)]:
        conn = edge.dst_conn if is_input else edge.src_conn
        if conn is None or edge.data.is_empty() or conn in found:
            continue
        shape, strides = operand_view(edge, sdfg)
        found[conn] = Operand(sdfg.arrays[edge.data.data], shape, strides, is_input)
    return found


def pointer(conn: str, is_input: bool) -> str:
    """The tasklet connector of ``conn``."""
    return f"__in{conn}" if is_input else f"__out{conn}"


def literal(value: Any) -> int | None:
    """``value`` as an int when it is a literal, else ``None``."""
    return None if symbolic.issymbolic(value) else int(value)


class ExpandLapackBlock(ExpandTransformation):
    """The ``'CUDA (block)'`` expansion: one tasklet under a lane map, its code from :meth:`body`."""

    runs_inside_kernel = True
    environments = [CUDA, BlockLapack]

    #: The matrix connector whose order is the problem's.
    matrix = "_xin"
    #: Whether the code holds ``getri``, whose order the lanes' registers bound.
    inverts = False

    @classmethod
    def body(cls, node: nodes.LibraryNode, ops: dict[str, Operand], uid: str) -> str:
        """The C++ every lane runs."""
        raise NotImplementedError

    @classmethod
    def scratch_bytes(cls, node: nodes.LibraryNode, ops: dict[str, Operand], n: int) -> int:
        """Shared scratch for order ``n``; ``0`` when the operands suffice."""
        return 0

    @classmethod
    def refusal(cls, node: nodes.LibraryNode, state: SDFGState, sdfg: SDFG) -> str | None:
        """Why the block lowering cannot take ``node``, or ``None``."""
        ops = operands(node, state, sdfg)
        if cls.matrix not in ops or len(ops[cls.matrix].shape) != 2:
            return "the matrix is not one 2-D problem per invocation"
        for conn, op in ops.items():
            if op.desc.dtype.veclen != 1:
                return f"{conn} is a vector type"
            if op.desc.dtype.base_type not in REAL_TYPES + (dtypes.int32,):
                return f"{conn} is {op.desc.dtype}, not a real float type"
        n = literal(ops[cls.matrix].shape[0])
        if cls.inverts and n is not None and n > ROWS_PER_LANE * BLOCK_COLLECTIVE_THREADS:
            return f"order {n} exceeds the lanes' registers"
        if cls.scratch_bytes(node, ops, 1):
            if n is None:
                return "its shared scratch needs a literal order"
            if cls.scratch_bytes(node, ops, n) > SHARED_SCRATCH_BYTES:
                return f"order {n} needs more than {SHARED_SCRATCH_BYTES} bytes of shared scratch"
        return None

    @classmethod
    def expansion(cls, node: nodes.LibraryNode, parent_state: SDFGState, parent_sdfg: SDFG, **kwargs: Any) -> SDFG:
        reason = cls.refusal(node, parent_state, parent_sdfg)
        if reason is not None:
            raise NotImplementedError(f"{node.label}: no thread-block lowering, {reason}")
        ops = operands(node, parent_state, parent_sdfg)
        # An output nothing reads (``eigvalsh``) is not computed, so the nested SDFG has no array for it
        for conn in [c for c in node.out_connectors if c not in ops]:
            node.remove_out_connector(conn)
        sdfg = SDFG(node.label + "_block")
        for conn, op in ops.items():
            sdfg.add_array(conn, op.shape, op.desc.dtype, strides=op.strides, storage=op.desc.storage)
        reads = {pointer(c, True): c for c, op in ops.items() if op.is_input}
        writes = {pointer(c, False): c for c, op in ops.items() if not op.is_input or c in node.out_connectors}
        code = cls.body(node, ops, global_code_id(parent_sdfg, parent_state, node))
        add_lane_tasklet(sdfg, node.label, code, reads, writes)
        return sdfg


def mat(ops: dict[str, Operand], conn: str, is_input: bool = False) -> str:
    """``block_lapack::Mat`` over ``conn``: read-only through its input pointer, else through its output pointer."""
    op = ops[conn]
    strides = list(op.strides) + [1]
    qualifier = "const " if is_input else ""
    return (
        f"dace::block_lapack::Mat<{qualifier}{op.ctype}>{{{pointer(conn, is_input)}, {cpp(strides[0])}, "
        f"{cpp(strides[1])}}}"
    )


def cpp(value: Any) -> str:
    """A size or stride as a C++ ``long``."""
    return f"(long)({sym2cpp(value)})"


def order(ops: dict[str, Operand], conn: str) -> str:
    return cpp(ops[conn].shape[0])


def columns(ops: dict[str, Operand], conn: str) -> str:
    """The columns of ``conn``; one for a vector."""
    return cpp(ops[conn].shape[1]) if len(ops[conn].shape) > 1 else "1L"


def copy_code(ops: dict[str, Operand], src: str, dst: str) -> str:
    """Every lane's share of ``dst = src``; nothing when they are one buffer."""
    strides = list(ops[src].strides) + [1]
    return (
        f"dace::block_lapack::copy({pointer(src, True)}, {cpp(strides[0])}, {cpp(strides[1])}, {mat(ops, dst)}, "
        f"{order(ops, src)}, {columns(ops, src)});\n"
    )


def scratch_code(ctype: str, name: str, count: int) -> str:
    """A static shared array of the block."""
    return f"__shared__ {ctype} {name}[{count}];\n"


def info_code(ops: dict[str, Operand], value: str) -> str:
    """Lane 0 stores the info word, when ``_res`` is connected."""
    if "_res" not in ops:
        return f"(void)({value});\n"
    return f"if (threadIdx.x == 0) {pointer('_res', False)}[0] = {value};\n"


def syevj_code(ops: dict[str, Operand], a: str, w: str, v: str, lower: bool, work: str) -> str:
    """Cyclic Jacobi of ``a`` into eigenvalues ``w`` and eigenvectors ``v``, either of which may be unconnected."""
    op = ops[a]
    ctype = op.ctype
    strides = list(op.strides)
    eps = float(np.finfo(op.desc.dtype.base_type.type).eps)
    values = f"{pointer(w, False)}, {cpp(ops[w].strides[0])}" if w in ops else f"({ctype}*)nullptr, 0L"
    vectors = mat(ops, v) if v in ops else f"dace::block_lapack::Mat<{ctype}>{{nullptr, 0L, 0L}}"
    return (
        f"dace::block_lapack::syevj({pointer(a, True)}, {cpp(strides[0])}, {cpp(strides[1])}, "
        f"{'true' if lower else 'false'}, {work}, {order(ops, a)}, {values}, {vectors}, {MAX_JACOBI_SWEEPS}, "
        f"({ctype})({eps!r}));\n"
    )
