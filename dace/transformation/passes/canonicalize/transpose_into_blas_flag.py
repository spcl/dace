# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A matrix transpose whose only reader is a BLAS operand becomes that operand's transpose flag.

``hpsi = beta @ (dij @ (beta.T @ psi))`` materialized ``beta.T`` (116 MB on kleinman_bylander_nonlocal) for a GEMM that
takes a transposed operand by its ``transA``/``transB`` flag.
"""

from dace import SDFG, data
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


def flip_transpose_flag(node: nodes.Node, connector: str) -> bool:
    """Flip the flag that transposes ``node``'s operand ``connector``; whether ``node`` has one."""
    if isinstance(node, (Gemm, MatMul)) and connector == "_a":
        node.transA = not node.transA
    elif isinstance(node, (Gemm, MatMul)) and connector == "_b":
        node.transB = not node.transB
    elif isinstance(node, Gemv) and connector == "_A":
        node.transA = not node.transA
    else:
        return False
    return True


def fold_one(sdfg: SDFG, state: SDFGState, transpose: nodes.LibraryNode, access_counts: dict[str, int]) -> bool:
    """Feed ``transpose``'s input straight into the BLAS operand that is the only reader of the transient it writes,
    with that operand's transpose flag flipped. Returns whether it did."""
    ins, outs = state.in_edges(transpose), state.out_edges(transpose)
    if len(ins) != 1 or len(outs) != 1:
        return False
    src, out = ins[0].src, outs[0].dst
    if not isinstance(src, nodes.AccessNode) or not isinstance(out, nodes.AccessNode):
        return False
    src_desc, out_desc = sdfg.arrays[src.data], sdfg.arrays[out.data]
    if not out_desc.transient or isinstance(out_desc, data.View) or access_counts.get(out.data) != 1:
        return False
    if len(src_desc.shape) != 2 or ins[0].data.subset.num_elements() != src_desc.total_size:
        return False
    readers = state.out_edges(out)
    if state.in_degree(out) != 1 or len(readers) != 1:
        return False
    reader = readers[0]
    if not flip_transpose_flag(reader.dst, reader.dst_conn):
        return False
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
