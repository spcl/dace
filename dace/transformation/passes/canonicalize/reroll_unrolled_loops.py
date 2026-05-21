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

        # The distinct lane offsets must be equally spaced from 0:
        # ``{0, g, 2g, ..., (m-1)g}`` -- ``m`` lanes, spacing ``g``. The re-roll
        # then steps by ``g`` (one position per iteration) instead of ``step``.
        distinct = sorted(set(offsets.values()))
        m = len(distinct)
        if m < 2 or distinct[0] != 0:
            return False
        g = distinct[1]
        if g <= 0 or distinct != [j * g for j in range(m)]:
            return False

        # Position-coverage safety. A loop iteration advances by ``step`` while
        # one lane block spans ``m * g``. ``step > m*g`` leaves gaps that the
        # re-rolled step-``g`` loop would wrongly fill; ``step < m*g`` overlaps,
        # so a position is touched by more than one lane -- safe only when no
        # array is both read and written by the lanes (otherwise the re-roll
        # changes a read-modify-write count). ``step == m*g`` tiles exactly.
        if step > m * g:
            return False
        if step < m * g and self._has_read_modify_write(offsets):
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

        # Group the boundary edges by lane offset and find each lane's nodes.
        lane_edges: Dict[int, List] = {d: [] for d in distinct}
        for edge, off in offsets.items():
            lane_edges[off].append(edge)
        lane_nodes = {d: self._lane_nodes(state, lane_edges[d], shared) for d in distinct}

        # Lanes must be disjoint and structurally identical (same tasklet-code multiset).
        all_internal: Set = set()
        codes: Dict[int, List[str]] = {}
        for d in distinct:
            nodes_d = lane_nodes[d]
            if all_internal & nodes_d:
                return False
            all_internal |= nodes_d
            codes[d] = sorted(n.code.as_string for n in nodes_d if isinstance(n, nodes.Tasklet))
        if any(codes[d] != codes[0] for d in distinct[1:]):
            return False
        if not codes[0]:
            return False

        # Re-roll: drop every lane but offset 0 (their internal nodes + boundary
        # edges), then rewrite the loop to step ``g``. Lane 0 keeps its
        # ``loop_var + 0`` subsets, which now sweep every position.
        for d in distinct[1:]:
            for edge in lane_edges[d]:
                if edge in state.edges():
                    state.remove_edge(edge)
            for n in lane_nodes[d]:
                if n in state.nodes():
                    state.remove_node(n)

        # The frontend often splits a read+written array into per-lane
        # AccessNodes; the copies that fed the removed lanes are now isolated.
        for n in list(state.nodes()):
            if isinstance(n, nodes.AccessNode) and state.degree(n) == 0:
                state.remove_node(n)

        self._rewrite_step(loop, loop_var, g, m)
        return True

    def _has_read_modify_write(self, offsets: Dict) -> bool:
        """Whether any array is both read and written by the lanes.

        :param offsets: ``{edge: offset}`` for the loop-variable-indexed edges.
        :returns: ``True`` if some array appears both as a read source and a
                  write destination among the lane boundary edges.
        """
        reads, writes = set(), set()
        for edge in offsets:
            if isinstance(edge.src, nodes.AccessNode):
                reads.add(edge.src.data)
            if isinstance(edge.dst, nodes.AccessNode):
                writes.add(edge.dst.data)
        return bool(reads & writes)

    def _rewrite_step(self, loop: LoopRegion, loop_var: str, g: int, m: int) -> None:
        """Rewrite a step-``S`` loop to step ``g`` over the flattened range.

        The original loop attains ``loop_var`` values ``init, ..., end`` and the
        kept lane 0 sweeps positions ``init .. end + (m - 1) * g``. The re-rolled
        loop steps by ``g`` with the exclusive bound ``end + m * g``.

        :param loop: The loop region to rewrite.
        :param loop_var: The loop variable name.
        :param g: The lane-offset spacing (the new step).
        :param m: The lane count.
        """
        loop_end = loop_analysis.get_loop_end(loop)
        new_excl = symbolic.pystr_to_symbolic(loop_end) + m * g
        loop.update_statement = dace.properties.CodeBlock(f"{loop_var} = {loop_var} + {g}")
        loop.loop_condition = dace.properties.CodeBlock(f"{loop_var} < ({symbolic.symstr(new_excl)})")
