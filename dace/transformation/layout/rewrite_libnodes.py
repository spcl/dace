# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Library-node rewrites that expose an operand's LAYOUT to the layout passes.

A layout change must reach a library node's SEMANTIC index description, not just its memlets:

  * ``transform_einsum(einsum_str, operand_index, perm)`` -- when a layout pass permutes an einsum
    operand's dimensions, reorder that operand's subscript letters so the string still describes the
    same contraction on the new layout (the einsum analog of permute's memlet-subset rewrite).
  * ``GemmToTensorDot`` -- rewrite a layout-opaque ``Gemm`` (``C = A @ B``) into a ``TensorDot``
    (an einsum-syntax node with explicit contracted axes), so the operand layout becomes visible
    and permutable. Only applied when ``alpha == 1``, ``beta == 0`` and there is no ``C`` input
    (``TensorDot`` pins ``alpha=1, beta=0`` and has no accumulator); a scaled/accumulating Gemm is
    left untouched (layout still reaches it through its memlets).

Copy / memset are layout-agnostic: their memlets are renamed/reshaped by the generic passes, so no
node rewrite is needed here.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import dace
from dace import symbolic
from dace.sdfg import nodes as nd
from dace.transformation import pass_pipeline as ppl


def transform_einsum(einsum_str: str, operand_index: int, perm: Tuple[int, ...]) -> str:
    """Permute one einsum operand's subscripts by ``perm`` (``new[i] = old[perm[i]]``).

    ``operand_index`` indexes the operands in the order the ``Einsum`` node binds them -- SORTED
    input-connector name order. The other groups and the output are unchanged: the same letters
    still denote the same logical dimensions, so the contraction is preserved on the new layout.

    ``einsum('ij,jk->ik', ...)`` with ``perm=(1, 0)`` on operand 0 -> ``'ji,jk->ik'``.
    """
    lhs, sep, rhs = einsum_str.partition('->')
    groups = [t.strip() for t in lhs.split(',')]
    if operand_index < 0 or operand_index >= len(groups):
        raise ValueError(f"transform_einsum: operand_index {operand_index} out of range for '{einsum_str}'")
    g = groups[operand_index]
    if len(perm) != len(g):
        raise ValueError(f"transform_einsum: perm {perm} does not match operand '{g}' rank {len(g)}")
    if sorted(perm) != list(range(len(g))):
        raise ValueError(f"transform_einsum: perm {perm} is not a permutation")
    groups[operand_index] = ''.join(g[perm[i]] for i in range(len(perm)))
    out = ','.join(groups)
    return f"{out}{sep}{rhs}" if sep else out


def remap_contracted_axes(axes: List[int], perm: Tuple[int, ...]) -> List[int]:
    """The ``TensorDot`` numeric analog of :func:`transform_einsum` for one permuted operand.

    When an operand is permuted by ``perm`` (``new[i] = old[perm[i]]``), an old contracted axis
    ``a`` moves to new position ``perm.index(a)``. Returns the remapped ``left_axes`` / ``right_axes``
    so the SAME logical modes are still contracted on the new layout. (Reordering only the operand's
    CONTRACTED axes leaves the output modes unchanged; reordering FREE modes additionally requires
    the node's ``permutation`` to compensate -- not handled here, so a caller permuting free modes of
    a multi-free-axis contraction must update ``permutation`` itself.)
    """
    return [perm.index(a) for a in axes]


def permute_reduce(node, perm: Tuple[int, ...]) -> None:
    """Remap a ``Reduce`` node's ``axes`` when its input operand is permuted by ``perm``.

    A reduction is a contraction of its ``axes``; permuting the input moves each reduced axis to a
    new position, so the reduced axes follow (the non-reduced axes -- the output -- reorder to match
    the input, which the output array's own layout carries). A reduce-all (``axes is None``) is
    order-independent and left unchanged.
    """
    if node.axes is None:
        return
    node.axes = remap_contracted_axes(list(node.axes), perm)


def block_scan_stride(node, factor: int) -> None:
    """Bump a ``Scan`` node's ``stride`` by ``factor`` when its 1-D array is interleaved (blocked)
    by ``factor``.

    Blocking a scan array to ``[N/factor, factor]`` and scanning per lane makes the ``factor``
    residue classes mod ``factor`` independent scans over the flattened array -- exactly the
    ``out[i+stride] = out[i] OP in[i]`` recurrence the ``stride`` property expresses.
    """
    node.stride = node.stride * factor


@dataclass
class GemmToTensorDot(ppl.Pass):
    """Rewrite every eligible ``Gemm`` (``alpha==1``, ``beta==0``, no ``C`` input) into a
    ``TensorDot`` so its operand layout is visible to the layout passes. Ineligible Gemms are left
    in place. Runs inside / after ``prepare_for_layout``.

    :param implementation: the ``TensorDot`` implementation to set (``"pure"`` on CPU).
    """

    def __init__(self, implementation: str = "pure"):
        self._impl = implementation

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def _is_eligible(self, node: nd.LibraryNode) -> bool:
        return (symbolic.equal_valued(1, node.alpha) and symbolic.equal_valued(0, node.beta) and not node.alpha_input
                and not node.beta_input and "_cin" not in node.in_connectors)

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Dict[str, Any]) -> int:
        from dace.libraries.blas.nodes.gemm import Gemm
        from dace.libraries.linalg import TensorDot

        count = 0
        for state in sdfg.all_states():
            for node in [n for n in state.nodes() if type(n) is Gemm]:
                if not self._is_eligible(node):
                    continue
                count += self._rewrite(sdfg, state, node, TensorDot)
        return count

    def _rewrite(self, sdfg, state, node, TensorDot) -> int:
        a_edge = next((e for e in state.in_edges(node) if e.dst_conn == "_a"), None)
        b_edge = next((e for e in state.in_edges(node) if e.dst_conn == "_b"), None)
        c_edge = next((e for e in state.out_edges(node) if e.src_conn == "_c"), None)
        if a_edge is None or b_edge is None or c_edge is None:
            return 0

        # C = A @ B, A=[M,K], B=[K,N]: contract A's K with B's K. transA => A=[K,M] (contract axis 0);
        # transB => B=[N,K] (contract axis 1). Output modes = non-contracted-A ++ non-contracted-B.
        left_axes = [0] if node.transA else [1]
        right_axes = [1] if node.transB else [0]

        td = TensorDot(f"{node.label}_tensordot", left_axes=left_axes, right_axes=right_axes)
        td.implementation = self._impl
        state.add_node(td)
        state.add_edge(a_edge.src, a_edge.src_conn, td, "_left_tensor", dace.Memlet.from_memlet(a_edge.data))
        state.add_edge(b_edge.src, b_edge.src_conn, td, "_right_tensor", dace.Memlet.from_memlet(b_edge.data))
        state.add_edge(td, "_out_tensor", c_edge.dst, c_edge.dst_conn, dace.Memlet.from_memlet(c_edge.data))

        for e in (a_edge, b_edge, c_edge):
            state.remove_edge(e)
        state.remove_node(node)
        return 1
