# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Library-node rewrites that expose an operand's layout to the layout passes: reorders einsum subscripts and rewrites ``Gemm``/``CopyLibraryNode`` into ``TensorDot``/``TensorTranspose`` so a layout change reaches node semantics, not just memlets."""
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import dace
from dace import symbolic
from dace.sdfg import nodes as nd
from dace.transformation import pass_pipeline as ppl


def transform_einsum(einsum_str: str, operand_index: int, perm: Tuple[int, ...]) -> str:
    """Permute one einsum operand's subscripts by ``perm`` (``new[i] = old[perm[i]]``); ``operand_index`` follows sorted input-connector order."""
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
    """``TensorDot`` analog of :func:`transform_einsum`: remaps contracted axes for an operand permuted by ``perm``. Free-mode reorders need ``permutation`` updated separately."""
    return [perm.index(a) for a in axes]


def permute_reduce(node, perm: Tuple[int, ...]) -> None:
    """Remap a ``Reduce`` node's ``axes`` for an input permuted by ``perm``; reduce-all (``axes is None``) is left unchanged."""
    if node.axes is None:
        return
    node.axes = remap_contracted_axes(list(node.axes), perm)


def block_scan_stride(node, factor: int) -> None:
    """Bump a ``Scan`` node's ``stride`` by ``factor`` for a blocked (interleaved) 1-D array."""
    node.stride = node.stride * factor


@dataclass
class GemmToTensorDot(ppl.Pass):
    """Rewrite eligible ``Gemm`` (``alpha==1``, ``beta==0``, no ``C`` input) into ``TensorDot`` for layout visibility; ``TensorDot`` has no default implementation, so ``select_layout_lowering`` must run before compile."""

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

        # C=A@B: contract A's K with B's K; transA/transB flip which axis is K.
        left_axes = [0] if node.transA else [1]
        right_axes = [1] if node.transB else [0]

        td = TensorDot(f"{node.label}_tensordot", left_axes=left_axes, right_axes=right_axes)
        state.add_node(td)
        state.add_edge(a_edge.src, a_edge.src_conn, td, "_left_tensor", dace.Memlet.from_memlet(a_edge.data))
        state.add_edge(b_edge.src, b_edge.src_conn, td, "_right_tensor", dace.Memlet.from_memlet(b_edge.data))
        state.add_edge(td, "_out_tensor", c_edge.dst, c_edge.dst_conn, dace.Memlet.from_memlet(c_edge.data))

        for e in (a_edge, b_edge, c_edge):
            state.remove_edge(e)
        state.remove_node(node)
        return 1


def copy_permutation_axes(in_sizes: List, out_sizes: List):
    """Axes ``P`` such that ``out = transpose(in, P)``, or ``None`` if no transpose is needed (same order or reshape). Raises ``NotImplementedError`` on ambiguous (repeated-size) permutations."""

    def same(a, b) -> bool:
        return dace.symbolic.simplify(a - b) == 0

    if len(in_sizes) != len(out_sizes) or all(same(a, b) for a, b in zip(in_sizes, out_sizes)):
        return None  # same order or rank change -> left to the copy node
    if sorted(str(dace.symbolic.simplify(s))
              for s in in_sizes) != sorted(str(dace.symbolic.simplify(s)) for s in out_sizes):
        return None  # different extents -> reshape, not transpose

    axes, used = [], [False] * len(in_sizes)
    for os in out_sizes:
        matches = [k for k in range(len(in_sizes)) if not used[k] and same(in_sizes[k], os)]
        if len(matches) != 1:
            raise NotImplementedError(
                "RewriteCopyForLayout: ambiguous copy permutation (a repeated dim size); the "
                "permutation is not recoverable from the operand shapes alone. Drive the conversion "
                "from the permute pass, which knows the applied permutation.")
        axes.append(matches[0])
        used[matches[0]] = True
    return axes


@dataclass
class RewriteCopyForLayout(ppl.Pass):
    """Rewrite ``CopyLibraryNode``s whose operands ended up with different layouts into ``TensorTranspose``; same-layout copies and reshapes are left untouched. Run after the layout change, with no implementation set -- ``select_layout_lowering`` (or the node default) assigns one at compile time."""

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Dict[str, Any]) -> int:
        from dace.libraries.standard.nodes.copy_node import CopyLibraryNode
        from dace.libraries.linalg import TensorTranspose

        count = 0
        for state in sdfg.all_states():
            for node in [n for n in state.nodes() if type(n) is CopyLibraryNode]:
                count += self._rewrite(state, node, CopyLibraryNode, TensorTranspose)
        return count

    def _rewrite(self, state, node, CopyLibraryNode, TensorTranspose) -> int:
        in_edge = next((e for e in state.in_edges(node) if e.dst_conn == CopyLibraryNode.INPUT_CONNECTOR_NAME), None)
        out_edge = next((e for e in state.out_edges(node) if e.src_conn == CopyLibraryNode.OUTPUT_CONNECTOR_NAME), None)
        if in_edge is None or out_edge is None:
            return 0
        axes = copy_permutation_axes(in_edge.data.subset.size(), out_edge.data.subset.size())
        if axes is None:
            return 0

        tt = TensorTranspose(f"{node.label}_transpose", axes=axes)
        state.add_node(tt)
        state.add_edge(in_edge.src, in_edge.src_conn, tt, "_inp_tensor", dace.Memlet.from_memlet(in_edge.data))
        state.add_edge(tt, "_out_tensor", out_edge.dst, out_edge.dst_conn, dace.Memlet.from_memlet(out_edge.data))
        state.remove_edge(in_edge)
        state.remove_edge(out_edge)
        state.remove_node(node)
        return 1
