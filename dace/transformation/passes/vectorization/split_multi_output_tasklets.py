# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Fission a multi-output tasklet into one single-output tasklet per output.

The vectorization prep phase splits fused compute so that :class:`MapFission` can
separate independent map components and the einsum / reduction lifts can match a
single-output contraction / reduction map. :class:`~dace.transformation.passes.
split_tasklets.SplitTasklets` deliberately *skips* a tasklet with more than one
output connector ("Can't split a tasklet that has >1 outputs"), so a fused body
like gesummv's::

    ot = A[i, j] * x[j]      # -> tmp(+)[i]
    oy = B[i, j] * x[j]      # -> y(+)[i]

stays a single 2-output tasklet -- one map component -- that MapFission cannot
separate and LiftEinsum cannot match (two contractions in one node).

This pass runs first: for each output it backward-slices the statements needed to
compute that output (duplicating any shared read / intermediate) into a fresh
single-output tasklet, wired only to the inputs it reads and that one output. The
original tasklet is removed. Single-output tasklets and non-Python tasklets are
left untouched. After this pass every tasklet has exactly one output, so
``SplitTasklets`` + ``MapFission`` behave as expected.

An output whose statement transitively reads *another* output is NOT split (the
outputs are not independent -- splitting would need to re-order / share mutable
state); such a tasklet is left intact for a later pass or a clean abort.
"""
import ast
import copy
from typing import Dict, List, Optional, Set, Tuple

import dace
from dace.sdfg import SDFG, SDFGState, nodes as nd
from dace.transformation import pass_pipeline as ppl


def _rhs_names(node: ast.AST) -> Set[str]:
    """Every ``Name`` identifier read in an expression AST."""
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


class SplitMultiOutputTasklets(ppl.Pass):
    """Split every >1-output Python tasklet into one single-output tasklet per output
    (backward-slicing shared reads / intermediates into each). See the module
    docstring."""

    CATEGORY: str = "Vectorization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _) -> Optional[int]:
        count = 0
        for state in sdfg.all_states():
            for tasklet in list(state.nodes()):
                if not isinstance(tasklet, nd.Tasklet):
                    continue
                if len(tasklet.out_connectors) <= 1:
                    continue
                if tasklet.code.language != dace.dtypes.Language.Python:
                    continue
                if self._split_one(state, tasklet):
                    count += 1
        return count or None

    def _split_one(self, state: SDFGState, tasklet: nd.Tasklet) -> bool:
        """Fission one multi-output tasklet in ``state``. Returns True if split."""
        try:
            body = ast.parse(tasklet.code.as_string).body
        except (SyntaxError, ValueError):
            return False
        # Only straight-line ``name = expr`` statements are safe to slice by output.
        if not body or not all(
                isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name)
                for s in body):
            return False
        stmts: List[Tuple[str, str, Set[str]]] = [(s.targets[0].id, ast.unparse(s.value), _rhs_names(s.value))
                                                  for s in body]
        out_conns = set(tasklet.out_connectors)
        in_conns = set(tasklet.in_connectors)
        # The last statement assigning each output connector is its defining statement.
        last_def: Dict[str, int] = {}
        for idx, (lhs, _, _) in enumerate(stmts):
            if lhs in out_conns:
                last_def[lhs] = idx
        if any(o not in last_def for o in out_conns):
            return False

        in_edges = {e.dst_conn: e for e in state.in_edges(tasklet) if e.dst_conn is not None}
        out_edges = {e.src_conn: e for e in state.out_edges(tasklet) if e.src_conn is not None}
        if any(o not in out_edges for o in out_conns):
            return False

        plans: List[Tuple[str, List[int], Set[str]]] = []
        for o in out_conns:
            sliced = self._backward_slice(stmts, last_def[o], out_conns - {o})
            if sliced is None:
                return False  # output depends on another output -- leave intact
            used_in = set()
            for i in sliced:
                used_in |= (stmts[i][2] & in_conns)
            plans.append((o, sorted(sliced), used_in))

        # Build the per-output tasklets and rewire edges (after slicing succeeds for
        # every output, so a partial failure never leaves a half-split state).
        for o, sliced, used_in in plans:
            code = "\n".join(f"{stmts[i][0]} = {stmts[i][1]}" for i in sliced)
            new_t = state.add_tasklet(f"{tasklet.label}_out_{o}", set(used_in), {o}, code)
            for ic in used_in:
                e = in_edges[ic]
                state.add_edge(e.src, e.src_conn, new_t, ic, copy.deepcopy(e.data))
            oe = out_edges[o]
            state.add_edge(new_t, o, oe.dst, oe.dst_conn, copy.deepcopy(oe.data))
        state.remove_node(tasklet)
        return True

    def _backward_slice(self, stmts: List[Tuple[str, str, Set[str]]], target_idx: int,
                        other_outputs: Set[str]) -> Optional[Set[int]]:
        """Indices of the statements needed to compute ``stmts[target_idx]``'s LHS.

        Walks the def-use chain backwards: each read name is resolved to its most
        recent prior assignment. Returns ``None`` if the slice reads another output
        connector (the outputs are entangled and cannot be cleanly separated).
        """
        needed: Set[int] = set()
        frontier = [target_idx]
        while frontier:
            idx = frontier.pop()
            if idx in needed:
                continue
            needed.add(idx)
            for name in stmts[idx][2]:
                if name in other_outputs:
                    return None
                producer = next((j for j in range(idx - 1, -1, -1) if stmts[j][0] == name), None)
                if producer is not None:
                    frontier.append(producer)
        return needed
