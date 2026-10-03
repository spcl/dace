# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lift a hand-written symmetric rank-k or rank-2k update nest to a ``Syrk`` / ``Syr2k`` BLAS node.

polybench ``syrk`` (``C := alpha*A*A^T + beta*C``) and ``syr2k`` (``C := alpha*A*B^T + alpha*B*A^T + beta*C``)
update one triangle of ``C`` as a per-row triangular accumulation::

    for i in range(N):
        C[i, :i + 1] *= beta[0]
        for k in range(M):
            C[i, :i + 1] += alpha[0] * A[i, k] * A[:i + 1, k]                                   # syrk
            C[i, :i + 1] += A[:i + 1, k] * alpha[0] * B[i, k] + B[:i + 1, k] * alpha[0] * A[i, k]  # syr2k

Both share one skeleton and one dataflow-expression match (see
:mod:`~dace.transformation.passes.canonicalize.rank_k_match`); only the operand pairing differs. One operand read
at ``[i,k]`` and ``[j,k]`` is a rank-k update, two cross-paired operands are a rank-2k update. The BLAS primitive
computes only the referenced triangle, threaded. Any deviation is a clean no-op.
"""
from typing import Any, Dict, List, NamedTuple, Optional

import sympy

from dace import SDFG, memlet as mm
from dace.sdfg.state import ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.canonicalize.rank_k_match import (RankKMatch, beta_and_inner_loop, expressions_equal,
                                                                  full_memlet, internal_writes_contained, loop_extent,
                                                                  loop_invariant, match_beta_state, operand_shape_ok,
                                                                  outer_loop_candidates, replace_loop_with_state,
                                                                  resolve_accumulate, root_sdfg_of, single_body_state,
                                                                  sink_node, square_output_ok)
from dace.transformation.transformation import explicit_cf_compatible

# Stand-in for the triangular slice index while the body expression is resolved.
SLICE_INDEX = sympy.Symbol("__rk_j")


class TriangularNest(NamedTuple):
    scale_state: SDFGState
    inner: LoopRegion
    acc_state: SDFGState
    c: str
    n: object
    k: object


def triangular_nest(root: SDFG, loop: LoopRegion) -> Optional[TriangularNest]:
    """The beta-scale state, inner ``k`` loop and square output ``C`` of a candidate nest."""
    n = loop_extent(loop)
    if n is None:
        return None
    split = beta_and_inner_loop(loop)
    if split is None:
        return None
    scale_state, inner = split
    k = loop_extent(inner)
    if k is None:
        return None
    acc_state = single_body_state(inner)
    if acc_state is None:
        return None
    # The output C is whatever the accumulation state writes (its only sink).
    sink = sink_node(acc_state)
    if sink is None:
        return None
    if not square_output_ok(root, sink.data, n) or not internal_writes_contained(loop, root, sink.data):
        return None
    return TriangularNest(scale_state, inner, acc_state, sink.data, n, k)


def symmetric_operands(roles: Dict[str, Any], c_array: str) -> Optional[List[str]]:
    """``[A]`` (rank-k) or sorted ``[A, B]`` (rank-2k) when every operand is read at both ``[i,k]`` and ``[j,k]``."""
    rows, cols = roles["row"], roles["col"]
    if len(rows) not in (1, 2) or rows.keys() != cols.keys():
        return None
    operands = sorted(rows)
    return None if c_array in operands else operands


def update_operands(root: SDFG, loop: LoopRegion, nest: TriangularNest, value: sympy.Basic, roles: Dict[str, Any],
                    beta: str) -> Optional[List[str]]:
    """The operands when ``value`` is ``C[i,j] + alpha * <symmetric pairing>`` over loop-invariant inputs."""
    operands = symmetric_operands(roles, nest.c)
    if operands is None:
        return None
    alpha, alpha_sym = next(iter(roles["coeffs"].items()))
    if not all(operand_shape_ok(root, name, roles["trans"], nest.n, nest.k) for name in operands):
        return None
    if not loop_invariant(loop, (*operands, alpha, beta)):
        return None
    # A[i,k]*A[j,k] for one operand, A[i,k]*B[j,k] + B[i,k]*A[j,k] for two.
    pairing = sum(roles["row"][p] * roles["col"][q] for p, q in zip(operands, reversed(operands)))
    if not expressions_equal(value, roles["c"] + alpha_sym * pairing):
        return None
    return operands


@explicit_cf_compatible
class LoopToRankKUpdate(ppl.Pass):
    """Lift a hand-written symmetric rank-k / rank-2k update loop nest to a ``Syrk`` / ``Syr2k`` node."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.CFG)

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        count = 0
        for parent, loop in outer_loop_candidates(sdfg):
            if loop not in parent.nodes():
                continue  # already spliced out (defensive)
            match = self.match(parent, loop)
            if match is None:
                continue
            self.replace(parent, loop, match)
            count += 1
        return count or None

    def match(self, parent: ControlFlowRegion, loop: LoopRegion) -> Optional[RankKMatch]:
        root = root_sdfg_of(parent)
        nest = triangular_nest(root, loop)
        if nest is None:
            return None
        i = loop.loop_variable
        beta_match = match_beta_state(nest.scale_state, root, nest.c, i, SLICE_INDEX, nest.n)
        if beta_match is None:
            return None
        beta, uplo = beta_match
        resolved = resolve_accumulate(nest.acc_state, root, nest.c, i, SLICE_INDEX, nest.inner.loop_variable, nest.n)
        # The scale and the accumulation must cover the same triangle.
        if resolved is None or resolved[2] != uplo:
            return None
        roles = resolved[1]
        operands = update_operands(root, loop, nest, resolved[0], roles, beta)
        if operands is None:
            return None
        return RankKMatch(c=nest.c,
                          a=operands[0],
                          b=operands[1] if len(operands) == 2 else None,
                          alpha=next(iter(roles["coeffs"])),
                          beta=beta,
                          uplo=uplo,
                          trans=roles["trans"],
                          n=nest.n,
                          k=nest.k)

    def replace(self, parent: ControlFlowRegion, loop: LoopRegion, match: RankKMatch) -> None:
        # Deferred: the BLAS package imports transformations.
        from dace.libraries.blas.nodes.syr2k import Syr2k
        from dace.libraries.blas.nodes.syrk import Syrk
        node_class, suffix = (Syrk, "_syrk") if match.b is None else (Syr2k, "_syr2k")
        operands = [match.a] if match.b is None else [match.a, match.b]
        root = root_sdfg_of(parent)
        state = replace_loop_with_state(parent, loop, loop.label + suffix)
        node = node_class(loop.label + suffix,
                          uplo=match.uplo,
                          trans=match.trans,
                          alpha=1,
                          beta=1,
                          alpha_input=True,
                          beta_input=True)
        state.add_node(node)
        for conn, name in zip(("_a", "_b"), operands):
            state.add_edge(state.add_read(name), None, node, conn, full_memlet(root, name))
        state.add_edge(state.add_read(match.c), None, node, "_c", full_memlet(root, match.c))
        state.add_edge(state.add_read(match.alpha), None, node, "_alpha", mm.Memlet(f"{match.alpha}[0]"))
        state.add_edge(state.add_read(match.beta), None, node, "_beta", mm.Memlet(f"{match.beta}[0]"))
        state.add_edge(node, "_c", state.add_write(match.c), None, full_memlet(root, match.c))


__all__ = ["LoopToRankKUpdate"]
