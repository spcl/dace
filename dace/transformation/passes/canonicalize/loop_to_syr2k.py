# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lift a hand-written symmetric rank-2k update nest to a ``Syr2k`` BLAS node.

polybench ``syr2k`` computes ``C := alpha*A*B^T + alpha*B*A^T + beta*C`` over one
triangle of ``C``, written by hand as a per-row triangular accumulation rather than
the BLAS ``xSYR2K`` primitive::

    for i in range(N):
        C[i, :i + 1] *= beta[0]
        for k in range(M):
            C[i, :i + 1] += A[:i + 1, k] * alpha[0] * B[i, k] + B[:i + 1, k] * alpha[0] * A[i, k]

Recognising that nest and emitting a :class:`~dace.libraries.blas.nodes.syr2k.Syr2k`
node replaces the sequential in-place accumulation with the optimized primitive
(threaded ``dsyr2k`` / ``cublasDsyr2k``), which also computes only the referenced
triangle -- half the flops of the equivalent pair of ``gemm`` calls.

The sibling of :class:`~dace.transformation.passes.canonicalize.loop_to_syrk.LoopToSyrk`:
same skeleton, same conservative dataflow-expression match (see
:mod:`~dace.transformation.passes.canonicalize.rank_k_match`), but the body must
resolve to exactly ``C[i,j] + alpha[0]*(A[i,k]*B[j,k] + B[i,k]*A[j,k])`` -- the
two-operand pairing that makes the update symmetric.
"""
from typing import Optional

import sympy

from dace import SDFG, memlet as mm
from dace.sdfg.state import ControlFlowRegion, LoopRegion
from dace.subsets import Range
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.canonicalize.rank_k_match import (RankKMatch, beta_and_inner_loop, expressions_equal,
                                                                  internal_writes_contained, loop_extent,
                                                                  loop_invariant, match_beta_state, operand_shape_ok,
                                                                  outer_loop_candidates, replace_loop_with_state,
                                                                  resolve_accumulate, root_sdfg_of, single_body_state,
                                                                  sink_node, square_output_ok)
from dace.transformation.transformation import explicit_cf_compatible

# Stand-in for the triangular slice index while the body expression is resolved.
SLICE_INDEX = sympy.Symbol("__rk_j")


@explicit_cf_compatible
class LoopToSyr2k(ppl.Pass):
    """Lift a hand-written symmetric rank-2k update loop nest to a ``Syr2k`` node."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.CFG)

    def apply_pass(self, sdfg: SDFG, _) -> Optional[int]:
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
        c_array = sink.data
        if not square_output_ok(root, c_array, n):
            return None
        if not internal_writes_contained(loop, root, c_array):
            return None

        i, kvar = loop.loop_variable, inner.loop_variable
        beta_match = match_beta_state(scale_state, root, c_array, i, SLICE_INDEX, n)
        if beta_match is None:
            return None
        beta, scale_uplo = beta_match

        resolved = resolve_accumulate(acc_state, root, c_array, i, SLICE_INDEX, kvar, n)
        if resolved is None:
            return None
        value, roles, acc_uplo = resolved
        if scale_uplo != acc_uplo:
            return None  # the scale and the accumulation must cover the same triangle

        # syr2k reads TWO operands, each at both [i,k] and [j,k]: the cross-pairing
        # A[i,k]*B[j,k] + B[i,k]*A[j,k] is what makes the update symmetric.
        if len(roles["row"]) != 2 or len(roles["col"]) != 2:
            return None
        if set(roles["row"]) != set(roles["col"]):
            return None
        a_array, b_array = sorted(roles["row"])
        if c_array in (a_array, b_array):
            return None
        alpha, alpha_sym = next(iter(roles["coeffs"].items()))
        trans = roles["trans"]
        if not operand_shape_ok(root, a_array, trans, n, k) or not operand_shape_ok(root, b_array, trans, n, k):
            return None
        if not loop_invariant(loop, (a_array, b_array, alpha, beta)):
            return None
        expected = roles["c"] + alpha_sym * (roles["row"][a_array] * roles["col"][b_array] +
                                             roles["row"][b_array] * roles["col"][a_array])
        if not expressions_equal(value, expected):
            return None
        return RankKMatch(c=c_array,
                          a=a_array,
                          b=b_array,
                          alpha=alpha,
                          beta=beta,
                          uplo=scale_uplo,
                          trans=trans,
                          n=n,
                          k=k)

    def replace(self, parent: ControlFlowRegion, loop: LoopRegion, match: RankKMatch) -> None:
        from dace.libraries.blas.nodes.syr2k import Syr2k
        root = root_sdfg_of(parent)
        state = replace_loop_with_state(parent, loop, loop.label + "_syr2k")
        node = Syr2k(loop.label + "_syr2k",
                     uplo=match.uplo,
                     trans=match.trans,
                     alpha=1,
                     beta=1,
                     alpha_input=True,
                     beta_input=True)
        state.add_node(node)

        def full(name: str) -> mm.Memlet:
            # Fresh Range per edge -- DaCe forbids two memlets sharing one subset.
            return mm.Memlet(data=name, subset=Range([(0, s - 1, 1) for s in root.arrays[name].shape]))

        state.add_edge(state.add_read(match.a), None, node, "_a", full(match.a))
        state.add_edge(state.add_read(match.b), None, node, "_b", full(match.b))
        state.add_edge(state.add_read(match.c), None, node, "_c", full(match.c))
        state.add_edge(state.add_read(match.alpha), None, node, "_alpha", mm.Memlet(f"{match.alpha}[0]"))
        state.add_edge(state.add_read(match.beta), None, node, "_beta", mm.Memlet(f"{match.beta}[0]"))
        state.add_edge(node, "_c", state.add_write(match.c), None, full(match.c))


__all__ = ["LoopToSyr2k"]
