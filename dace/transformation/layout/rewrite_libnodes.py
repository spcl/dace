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


def flip_matmul_transpose(node, connector: str) -> None:
    """Toggle the BLAS transpose flag for one ``Gemm``/``MatMul`` operand: ``_a`` flips ``transA``, ``_b`` flips ``transB``. A 2-D layout transpose of the operand is absorbed by the flag -- BLAS reads the raw box and transposes it for free, so no ``Transpose`` node or physical copy is emitted."""
    if connector == "_a":
        node.transA = not node.transA
    elif connector == "_b":
        node.transB = not node.transB
    else:
        raise ValueError(f"flip_matmul_transpose: operand connector must be '_a' or '_b', got '{connector}'")


def flip_operand_transpose(node, connector: str) -> None:
    """Absorb a 2-D transpose of one BLAS operand into the library node's structural flag, so a layout permute stays a single native call with no physical transpose. Dispatches by node type:

    * ``Gemm`` / ``MatMul`` -- ``_a`` toggles ``transA``, ``_b`` toggles ``transB``.
    * ``Syrk`` (``C = A A^T``) -- ``_a`` toggles ``trans`` ``N`` <-> ``T``.
    * ``Symm`` (``A`` symmetric) -- ``_a`` toggles ``uplo`` ``L`` <-> ``U`` (transposing a symmetric matrix swaps its stored triangle); ``_b`` is a general matrix with no transpose flag and is refused.

    ``Syr2k`` carries a single ``trans`` for both ``A`` and ``B``, so a per-operand toggle is unsound (permuting both would double-flip back); it is refused. Every unhandled case raises rather than silently miscompiling."""
    from dace.libraries.blas.nodes.gemm import Gemm
    from dace.libraries.blas.nodes.matmul import MatMul
    from dace.libraries.blas.nodes.syrk import Syrk
    from dace.libraries.blas.nodes.syr2k import Syr2k
    from dace.libraries.blas.nodes.symm import Symm

    def toggle(v, a, b):
        return b if v == a else a

    if isinstance(node, (Gemm, MatMul)):
        flip_matmul_transpose(node, connector)
    elif isinstance(node, Syrk):
        if connector != "_a":
            raise NotImplementedError(f"flip_operand_transpose: Syrk has no transpose flag for operand '{connector}'.")
        node.trans = toggle(node.trans, "N", "T")
    elif isinstance(node, Symm):
        if connector != "_a":
            raise NotImplementedError(
                f"flip_operand_transpose: Symm operand '{connector}' is a general matrix with no transpose flag; "
                f"its transpose cannot be absorbed.")
        node.uplo = toggle(node.uplo, "L", "U")
    elif isinstance(node, Syr2k):
        raise NotImplementedError(
            "flip_operand_transpose: Syr2k shares one 'trans' flag across both operands; a per-operand transpose "
            "cannot be absorbed soundly (permuting both would double-flip). Permute its operands as a physical copy.")
    else:
        raise NotImplementedError(f"flip_operand_transpose: no transpose-flag rule for {type(node).__name__}.")


@dataclass
class FoldTransposeIntoMatMul(ppl.Pass):
    """Fold a ``Transpose`` feeding a ``Gemm``/``MatMul`` operand into the node's transpose flag: ``A --Transpose--> A_T --_a--> MatMul`` becomes ``A --_a--> MatMul(transA flipped)``, dropping the ``Transpose`` node and the dead intermediate. This is the ``Array -> Transpose -> Gemm`` pattern kept as a single Gemm call with the input marked transposed; only whole-operand (no partial subset) consumers are folded."""

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Memlets | ppl.Modifies.AccessNodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Dict[str, Any]) -> int:
        from dace.libraries.blas.nodes.gemm import Gemm
        from dace.libraries.blas.nodes.matmul import MatMul
        from dace.libraries.linalg.nodes.transpose import Transpose

        count = 0
        for state in sdfg.all_states():
            for tnode in [n for n in state.nodes() if type(n) is Transpose]:
                count += self._fold(state, tnode, (Gemm, MatMul))
        return count

    def _fold(self, state, tnode, gemm_types) -> int:
        in_edge = next((e for e in state.in_edges(tnode) if e.dst_conn == "_inp"), None)
        out_edge = next((e for e in state.out_edges(tnode) if e.src_conn == "_out"), None)
        if in_edge is None or out_edge is None or not isinstance(out_edge.dst, nd.AccessNode):
            return 0
        interm = out_edge.dst  # A_T access node

        # every consumer of the intermediate must be a Gemm/MatMul operand reading the whole box
        consumers = state.out_edges(interm)
        if not consumers or any(not isinstance(e.dst, gemm_types) or e.dst_conn not in ("_a", "_b") for e in consumers):
            return 0

        src, src_conn, src_memlet = in_edge.src, in_edge.src_conn, in_edge.data
        for e in list(consumers):
            flip_matmul_transpose(e.dst, e.dst_conn)
            state.add_edge(src, src_conn, e.dst, e.dst_conn, dace.Memlet.from_memlet(src_memlet))
            state.remove_edge(e)

        state.remove_edge(in_edge)
        state.remove_edge(out_edge)
        state.remove_node(tnode)
        if state.degree(interm) == 0:
            state.remove_node(interm)
        return 1


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


@dataclass
class SyrkToTensorDot(ppl.Pass):
    """Fallback for a ``Syrk`` (``C = A A^T``) whose operand layout cannot be absorbed by the ``trans`` flag alone (a blocked or non-``[1,0]`` relaid-out ``A``): rewrite it to the tensor contraction ``ik,jk->ij`` (``trans='T'`` -> ``ki,kj->ij``), which expresses any operand layout. Eligible only for a fresh full output (``alpha==1``, ``beta==0``, no ``_cin``) -- unlike ``Syrk`` the ``TensorDot`` writes the WHOLE symmetric ``C``, so the referenced-triangle-only contract is not preserved. ``TensorDot`` has no default implementation, so ``select_layout_lowering`` must run before compile."""

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def _is_eligible(self, node) -> bool:
        return (symbolic.equal_valued(1, node.alpha) and symbolic.equal_valued(0, node.beta) and not node.alpha_input
                and not node.beta_input and "_cin" not in node.in_connectors)

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Dict[str, Any]) -> int:
        from dace.libraries.blas.nodes.syrk import Syrk
        from dace.libraries.linalg import TensorDot

        count = 0
        for state in sdfg.all_states():
            for node in [n for n in state.nodes() if type(n) is Syrk]:
                if self._is_eligible(node):
                    count += self._rewrite(state, node, TensorDot)
        return count

    def _rewrite(self, state, node, TensorDot) -> int:
        a_edge = next((e for e in state.in_edges(node) if e.dst_conn == "_a"), None)
        c_edge = next((e for e in state.out_edges(node) if e.src_conn == "_c"), None)
        if a_edge is None or c_edge is None:
            return 0

        # C = A A^T (trans='N', A is N x K, contract K = axis 1) or A^T A (trans='T', contract axis 0).
        axis = [1] if node.trans == "N" else [0]
        td = TensorDot(f"{node.label}_tensordot", left_axes=axis, right_axes=axis)
        state.add_node(td)
        state.add_edge(a_edge.src, a_edge.src_conn, td, "_left_tensor", dace.Memlet.from_memlet(a_edge.data))
        state.add_edge(a_edge.src, a_edge.src_conn, td, "_right_tensor", dace.Memlet.from_memlet(a_edge.data))
        state.add_edge(td, "_out_tensor", c_edge.dst, c_edge.dst_conn, dace.Memlet.from_memlet(c_edge.data))
        state.remove_edge(a_edge)
        state.remove_edge(c_edge)
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
