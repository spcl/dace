# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A matrix transpose whose only reader is a BLAS operand becomes that operand's transpose flag.

``hpsi = beta @ (dij @ (beta.T @ psi))`` materialized ``beta.T`` (116 MB on kleinman_bylander_nonlocal) for a GEMM that
takes a transposed operand by its ``transA``/``transB`` flag. A transpose several products read folds into each.
"""

from dace import SDFG, data, symbolic
from dace.libraries.blas.nodes.gemm import Gemm
from dace.libraries.blas.nodes.gemv import Gemv
from dace.libraries.blas.nodes.matmul import MatMul
from dace.libraries.linalg.nodes.transpose import Transpose
from dace.libraries.linalg.nodes.ttranspose import TensorTranspose
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.state import SDFGState


def is_matrix_transpose(node: nodes.LibraryNode) -> bool:
    if isinstance(node, Transpose):
        return True
    return isinstance(node, TensorTranspose) and list(node.axes) == [1, 0] and node.alpha == 1 and node.beta == 0


def transpose_flag(node: nodes.Node, connector: str) -> str | None:
    """The name of the flag that transposes ``node``'s operand ``connector``, if ``node`` has one."""
    if isinstance(node, (Gemm, MatMul)) and connector in ("_a", "_b"):
        return "transA" if connector == "_a" else "transB"
    if isinstance(node, Gemv) and connector == "_A":
        return "transA"
    return None


def covers(memlet: Memlet, desc: data.Data) -> bool:
    """Whether ``memlet`` is all of ``desc``, by extent: a view spells its ``total_size`` unlike their product."""
    return len(memlet.subset) == len(desc.shape) and all(
        symbolic.equal_valued(0, b) and symbolic.equal_valued(1, s) and not symbolic.inequal_symbols(e, n - 1)
        for (b, e, s), n in zip(memlet.subset, desc.shape)
    )


def fold_one(sdfg: SDFG, state: SDFGState, transpose: nodes.LibraryNode, access_counts: dict[str, int]) -> bool:
    """Feed ``transpose``'s input straight into the BLAS operands that are the only readers of the transient it
    writes, each with its transpose flag flipped. Returns whether it did."""
    ins, outs = state.in_edges(transpose), state.out_edges(transpose)
    if len(ins) != 1 or len(outs) != 1:
        return False
    src, out = ins[0].src, outs[0].dst
    if not isinstance(src, nodes.AccessNode) or not isinstance(out, nodes.AccessNode):
        return False
    src_desc, out_desc = sdfg.arrays[src.data], sdfg.arrays[out.data]
    if not out_desc.transient or isinstance(out_desc, data.View) or access_counts.get(out.data) != 1:
        return False
    if len(src_desc.shape) != 2 or not covers(ins[0].data, src_desc):
        return False
    readers = state.out_edges(out)
    if state.in_degree(out) != 1 or not readers:
        return False
    for reader in readers:
        if reader.data.is_empty() or transpose_flag(reader.dst, reader.dst_conn) is None:
            return False
        if not covers(reader.data, out_desc):
            return False  # the operand is a window of the transpose, not the flipped matrix
    # One reader is ordered before a later write of the source by whatever ordered the transpose before it; a second
    # reader need not be (``t = A.T; C = t @ B; A = C; D = t @ C``). A view's source is written under another name.
    if len(readers) > 1 and (
        isinstance(src_desc, data.View)
        or any(n.data == src.data and n is not src and state.in_degree(n) > 0 for n in state.data_nodes())
    ):
        return False
    for reader in readers:
        flag = transpose_flag(reader.dst, reader.dst_conn)
        setattr(reader.dst, flag, not getattr(reader.dst, flag))
        state.add_edge(src, ins[0].src_conn, reader.dst, reader.dst_conn, Memlet.from_array(src.data, src_desc))
    state.remove_node(transpose)
    state.remove_node(out)
    sdfg.remove_data(out.data, validate=False)
    return True


def transposes_as_blas_flags(sdfg: SDFG) -> int:
    """Fold every matrix transpose feeding only a BLAS operand into that operand's transpose flag, in ``sdfg`` and its
    nests.

    :returns: how many transposes were folded.
    """
    folded = 0
    for sd in sdfg.all_sdfgs_recursive():
        access_counts: dict[str, int] = {}
        for state in sd.states():
            for node in state.data_nodes():
                access_counts[node.data] = access_counts.get(node.data, 0) + 1
        for state in sd.states():
            for node in [n for n in state.nodes() if isinstance(n, nodes.LibraryNode) and is_matrix_transpose(n)]:
                folded += fold_one(sd, state, node, access_counts)
    return folded
