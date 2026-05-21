# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Re-roll a manually-unrolled lane chain back into a step-1 loop.

A hand-unrolled loop has a step ``S != 1`` and a body that is ``S`` copies of a
single-position body, the copy ``k`` accessing every loop-variable-indexed array
one element past copy ``k - 1`` (``a[i]``, ``a[i + 1]``, ... ``a[i + S - 1]``).
TSVC ``s351`` (dense saxpy) and ``s353`` (indirect/gather saxpy) are the direct
examples. This pass detects that shape, keeps lane 0, drops the other lanes, and
re-rolls the loop to step 1 over the flattened range so each iteration does one
position. ``LoopToMap`` can then parallelize the result (the manually-unrolled
form blocks it because the lane subsets ``S * i + k`` look like a single
strided access).

The match is deliberately conservative: a constant positive step, a single body
state, lanes that index every loop-dependent array at the contiguous offsets
``0 .. S - 1`` with a unit coefficient, and structurally identical lanes.
Anything else is left untouched.
"""

from typing import Dict, List, Optional, Set

import dace
from dace import symbolic
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation.transformation import explicit_cf_compatible
from dace.transformation.passes.analysis import loop_analysis


def _const_int(value) -> Optional[int]:
    """Return ``value`` as a Python ``int`` if it is a constant integer, else ``None``.

    :param value: A symbolic expression or number.
    :returns: The integer value, or ``None`` if not a constant integer.
    """
    try:
        sym = symbolic.pystr_to_symbolic(value) if isinstance(value, str) else symbolic.pystr_to_symbolic(str(value))
        if sym.is_Integer:
            return int(sym)
    except Exception:
        return None
    return None


def _index_offset(subset, loop_var: str) -> Optional[int]:
    """Constant offset ``k`` of a one-dimensional ``loop_var + k`` index access.

    :param subset: The memlet subset to inspect.
    :param loop_var: The loop variable name.
    :returns: The integer offset ``k`` if the subset is a single point
              ``loop_var + k`` (unit coefficient), else ``None``.
    """
    if subset is None or len(subset) != 1:
        return None
    rb, re, rs = subset[0]
    if rb != re:
        return None
    expr = symbolic.pystr_to_symbolic(rb)
    if loop_var not in {str(s) for s in expr.free_symbols}:
        return None
    diff = symbolic.pystr_to_symbolic(expr - symbolic.pystr_to_symbolic(loop_var))
    return _const_int(diff)


@dace.properties.make_properties
@explicit_cf_compatible
class RerollUnrolledLoops(ppl.Pass):
    """Re-roll hand-unrolled lane chains (step ``S``, ``S`` lanes) to step-1 loops."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
        """Re-roll every matching unrolled loop in ``sdfg`` and its nested SDFGs.

        :param sdfg: SDFG to mutate in place.
        :returns: The number of loops re-rolled, or ``None`` if none.
        """
        rerolled = 0
        for sd in sdfg.all_sdfgs_recursive():
            for cfg in list(sd.all_control_flow_regions(recursive=True)):
                if isinstance(cfg, LoopRegion) and self._try_reroll(cfg):
                    rerolled += 1
        return rerolled or None

    def _loop_body_state(self, loop: LoopRegion) -> Optional[SDFGState]:
        """The loop's body when it is exactly one ``SDFGState``, else ``None``.

        :param loop: Loop region to inspect.
        :returns: The single body state or ``None``.
        """
        blocks = list(loop.nodes())
        states = [b for b in blocks if isinstance(b, SDFGState)]
        if len(blocks) != 1 or len(states) != 1:
            return None
        return states[0]

    def _lane_offsets(self, state: SDFGState, loop_var: str) -> Optional[Dict]:
        """Map each loop-variable-indexed boundary edge to its lane offset.

        :param state: The loop body state.
        :param loop_var: The loop variable name.
        :returns: ``{edge: offset}`` for every edge whose subset is
                  ``loop_var + k``, or ``None`` if any loop-dependent edge is not
                  of that unit-coefficient single-point form.
        """
        offsets = {}
        for edge in state.edges():
            if edge.data is None or edge.data.subset is None:
                continue
            if loop_var not in {str(s) for s in edge.data.subset.free_symbols}:
                continue
            off = _index_offset(edge.data.subset, loop_var)
            if off is None:
                return None
            offsets[edge] = off
        return offsets

    def _lane_nodes(self, state: SDFGState, lane_edges: List, shared: Set) -> Set:
        """Internal (non-shared) nodes reachable from a lane's boundary edges.

        :param state: The loop body state.
        :param lane_edges: The boundary edges belonging to the lane.
        :param shared: Access nodes shared across lanes (excluded from the walk).
        :returns: The set of internal nodes that belong only to this lane.
        """
        seen: Set = set()
        frontier = []
        for e in lane_edges:
            for n in (e.src, e.dst):
                if n not in shared:
                    frontier.append(n)
        while frontier:
            n = frontier.pop()
            if n in seen:
                continue
            seen.add(n)
            for e in state.in_edges(n):
                if e.src not in shared:
                    frontier.append(e.src)
            for e in state.out_edges(n):
                if e.dst not in shared:
                    frontier.append(e.dst)
        return seen

    def _try_reroll(self, loop: LoopRegion) -> bool:
        """Attempt to re-roll one loop; return whether it was rerolled.

        :param loop: The candidate loop region.
        :returns: ``True`` if the loop matched and was re-rolled.
        """
        loop_var = loop.loop_variable
        if not loop_var:
            return False
        stride = loop_analysis.get_loop_stride(loop)
        step = _const_int(stride) if stride is not None else None
        if step is None or step < 2:
            return False

        state = self._loop_body_state(loop)
        if state is None:
            return False

        offsets = self._lane_offsets(state, loop_var)
        if not offsets:
            return False
        # Every lane offset must be one of 0 .. step - 1, and all must be present.
        if set(offsets.values()) != set(range(step)):
            return False

        # Shared boundary access nodes are the loop's external arrays/scalars
        # (``a``, ``b``, the ``alpha`` scalar): non-transient containers, plus
        # any pure source/sink (a read-only input such as ``alpha = c[0]``
        # enters the body as an in-degree-0 source even when transient). The
        # frontend may split a read+written array into several per-lane
        # AccessNodes; those that lose all edges once their lane is removed are
        # cleaned up afterwards. The lane traversal stops at the shared nodes so
        # the per-lane transient chains come out disjoint.
        shared = set()
        for n in state.nodes():
            if not isinstance(n, nodes.AccessNode):
                continue
            desc = state.sdfg.arrays.get(n.data)
            if (desc is not None and not desc.transient) or state.in_degree(n) == 0 or state.out_degree(n) == 0:
                shared.add(n)

        # Group the boundary edges by lane and find each lane's internal nodes.
        lane_edges: Dict[int, List] = {k: [] for k in range(step)}
        for edge, off in offsets.items():
            lane_edges[off].append(edge)
        lane_nodes = {k: self._lane_nodes(state, lane_edges[k], shared) for k in range(step)}

        # Lanes must be disjoint and structurally identical (same tasklet-code multiset).
        all_internal: Set = set()
        codes: Dict[int, List[str]] = {}
        for k in range(step):
            nodes_k = lane_nodes[k]
            if all_internal & nodes_k:
                return False
            all_internal |= nodes_k
            codes[k] = sorted(n.code.as_string for n in nodes_k if isinstance(n, nodes.Tasklet))
        if any(codes[k] != codes[0] for k in range(1, step)):
            return False
        if not codes[0]:
            return False

        # Re-roll: drop lanes 1 .. step-1 (their internal nodes + boundary edges),
        # then rewrite the loop to step 1 over the flattened range. Lane 0 keeps
        # its ``loop_var + 0`` subsets, which now sweep every position.
        for k in range(1, step):
            for edge in lane_edges[k]:
                if edge in state.edges():
                    state.remove_edge(edge)
            for n in lane_nodes[k]:
                if n in state.nodes():
                    state.remove_node(n)

        # The frontend often splits a read+written array into per-lane
        # AccessNodes; the copies that fed the removed lanes are now isolated.
        for n in list(state.nodes()):
            if isinstance(n, nodes.AccessNode) and state.degree(n) == 0:
                state.remove_node(n)

        self._rewrite_to_unit_step(loop, loop_var, step)
        return True

    def _rewrite_to_unit_step(self, loop: LoopRegion, loop_var: str, step: int) -> None:
        """Rewrite a step-``S`` loop to step 1 over the flattened position range.

        The original loop attains ``loop_var`` values ``init, init + S, ...,
        end``; the kept lane 0 then sweeps positions ``init .. end + S - 1``. The
        re-rolled loop therefore steps by 1 with the exclusive bound ``end + S``.

        :param loop: The loop region to rewrite.
        :param loop_var: The loop variable name.
        :param step: The original constant step ``S``.
        """
        loop_end = loop_analysis.get_loop_end(loop)
        new_excl = symbolic.pystr_to_symbolic(loop_end) + step
        loop.update_statement = dace.properties.CodeBlock(f"{loop_var} = {loop_var} + 1")
        loop.loop_condition = dace.properties.CodeBlock(f"{loop_var} < ({symbolic.symstr(new_excl)})")
