# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Re-roll a manually-unrolled lane chain back into a step-``g`` loop.

A hand-unrolled loop has a step ``S != 1`` and a body that is ``m`` copies of a
single-position body -- copy ``k`` accessing every loop-variable-indexed array
at an offset ``k * g`` past copy 0 (``a[i]``, ``a[i + g]``, ... ``a[i + (m-1)*g]``).
TSVC ``s351`` (dense saxpy) and ``s353`` (indirect/gather saxpy) are the direct
examples. This pass detects that shape, keeps lane 0, drops the other lanes, and
re-rolls the loop to step ``g`` over the flattened range so each iteration does
one position. ``LoopToMap`` can then parallelize the result -- the unrolled form
blocks it because the lanes look like one strided ``S * i + k`` access.

The match is conservative: a constant positive step, a body made only of states,
``m`` lanes at equally-spaced offsets ``{0, g, ..., (m-1)g}`` (in memlet
subscripts and/or in interstate gather assignments such as ``idx = b[ip[i+k]]``),
structurally identical lanes, and contiguous coverage (``step == m*g``, or
overlapping pure writes). Anything else is left untouched.
"""

import re
from typing import Dict, List, Optional, Set, Tuple

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


def _offset_of_index(index_expr: str, loop_var: str) -> Optional[int]:
    """Constant offset ``k`` of an index expression ``loop_var + k``.

    :param index_expr: The (string) index expression, e.g. ``i + 2``.
    :param loop_var: The loop variable name.
    :returns: The integer offset ``k`` (unit coefficient), else ``None``.
    """
    try:
        expr = symbolic.pystr_to_symbolic(index_expr)
    except Exception:
        return None
    if loop_var not in {str(s) for s in expr.free_symbols}:
        return None
    return _const_int(expr - symbolic.pystr_to_symbolic(loop_var))


def _index_offset(subset, loop_var: str) -> Optional[int]:
    """Constant offset ``k`` of a one-dimensional ``loop_var + k`` subset.

    :param subset: The memlet subset to inspect.
    :param loop_var: The loop variable name.
    :returns: The integer offset ``k`` if the subset is a single point
              ``loop_var + k`` (unit coefficient), else ``None``.
    """
    if subset is None or len(subset) != 1:
        return None
    rb, re_, rs = subset[0]
    if rb != re_:
        return None
    return _offset_of_index(str(rb), loop_var)


def _assignment_offset(rhs: str, loop_var: str) -> Optional[int]:
    """Offset ``k`` of an interstate-assignment RHS ``loop_var + k`` or ``arr[loop_var + k]``.

    The gather form ``ip[i + k]`` (an array access) is handled by extracting the
    index between the brackets; the direct form ``i + k`` is used as-is.

    :param rhs: The assignment right-hand side string.
    :param loop_var: The loop variable name.
    :returns: The integer offset ``k``, or ``None`` if the RHS does not reference
              the loop variable as a single unit-coefficient offset.
    """
    rhs = rhs.strip()
    index_expr = rhs
    if '[' in rhs and rhs.endswith(']'):
        index_expr = rhs[rhs.rindex('[') + 1:-1]
    return _offset_of_index(index_expr, loop_var)


@dace.properties.make_properties
@explicit_cf_compatible
class RerollUnrolledLoops(ppl.Pass):
    """Re-roll hand-unrolled lane chains (step ``S``, ``m`` equally-spaced lanes)."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Nodes | ppl.Modifies.Memlets | ppl.Modifies.Symbols

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

    def _body_states(self, loop: LoopRegion) -> Optional[List[SDFGState]]:
        """The loop's body states when the body is made only of states.

        :param loop: Loop region to inspect.
        :returns: The list of body states, or ``None`` if the body contains a
                  non-state block (a nested region the matcher cannot reason about).
        """
        blocks = list(loop.nodes())
        if not blocks or not all(isinstance(b, SDFGState) for b in blocks):
            return None
        return blocks

    def _interstate_lane_symbols(self, loop: LoopRegion, loop_var: str) -> Optional[Dict[str, int]]:
        """Per-lane symbols defined on the body's interstate edges.

        A gather is unrolled as one interstate assignment per lane, e.g.
        ``ip_index = ip[i]``, ``ip_index_0 = ip[i + 1]``; the assigned symbol
        then carries the lane offset into the next state's memlets.

        :param loop: The loop region.
        :param loop_var: The loop variable name.
        :returns: ``{symbol: offset}`` for the loop-variable-dependent
                  assignments, or ``None`` if any such assignment is not a clean
                  single offset.
        """
        word = re.compile(r'\b%s\b' % re.escape(loop_var))
        sym_offset: Dict[str, int] = {}
        for edge in loop.edges():
            for sym, rhs in edge.data.assignments.items():
                if not word.search(str(rhs)):
                    continue
                off = _assignment_offset(str(rhs), loop_var)
                if off is None:
                    return None
                sym_offset[sym] = off
        return sym_offset

    def _edge_offset(self, subset, loop_var: str, sym_offset: Dict[str, int]) -> Tuple[Optional[int], bool]:
        """Lane offset of a memlet subset, and whether it is a lane edge at all.

        A subset indexed by ``loop_var + k`` has offset ``k``; a subset indexed by
        a per-lane symbol (e.g. ``b[ip_index]``) inherits that symbol's offset.

        :param subset: The memlet subset.
        :param loop_var: The loop variable name.
        :param sym_offset: The per-lane interstate symbols and their offsets.
        :returns: ``(offset, ok)``. ``ok`` is ``False`` only when the subset
                  references the loop variable or multiple lane symbols but cannot
                  be reduced to one clean offset (the caller then refuses). A
                  non-lane (shared/constant) subset returns ``(None, True)``.
        """
        if subset is None:
            return None, True
        names = {str(s) for s in subset.free_symbols}
        if loop_var in names:
            off = _index_offset(subset, loop_var)
            return (off, off is not None)
        lane_syms = names & set(sym_offset)
        if lane_syms:
            if len(lane_syms) != 1:
                return None, False
            return sym_offset[next(iter(lane_syms))], True
        return None, True

    #: Sympy class names of associative binary operations recognised as the
    #: merge tasklet's reduction op. Indexed via ``type(expr).__name__`` so we
    #: stay on the :mod:`dace.symbolic` interface (no direct ``import sympy``).
    _ASSOC_SYMPY_KIND = {'Add': '+', 'Mul': '*', 'Min': 'min', 'Max': 'max'}

    def _associative_op_kind(self, node) -> Optional[str]:
        """If ``node`` is a binary associative-op tasklet over ``__in1, __in2``,
        return its op kind (``'+'``, ``'*'``, ``'min'``, ``'max'``); else ``None``.

        Lifts the tasklet RHS to a :mod:`dace.symbolic` expression -- sympy's
        ``Add`` / ``Mul`` / ``Min`` / ``Max`` are associative by construction,
        so we don't need to enumerate parenthesisation or whitespace variants.
        """
        if not isinstance(node, nodes.Tasklet):
            return None
        code = node.code.as_string.strip()
        if '=' not in code:
            return None
        lhs, rhs = code.split('=', 1)
        if lhs.strip() != '__out':
            return None
        try:
            expr = symbolic.pystr_to_symbolic(rhs.strip())
        except Exception:
            return None
        if {str(s) for s in expr.free_symbols} != {'__in1', '__in2'}:
            return None
        if len(expr.args) != 2:
            return None
        return self._ASSOC_SYMPY_KIND.get(type(expr).__name__)

    def _shared_nodes(self, state: SDFGState) -> Set:
        """Boundary access nodes of a state (external arrays + read-only sources).

        :param state: The body state.
        :returns: The AccessNodes the lane traversal must stop at.
        """
        shared = set()
        for n in state.nodes():
            if not isinstance(n, nodes.AccessNode):
                continue
            desc = state.sdfg.arrays.get(n.data)
            if (desc is not None and not desc.transient) or state.in_degree(n) == 0 or state.out_degree(n) == 0:
                shared.add(n)
        return shared

    def _lane_nodes(self, state: SDFGState, lane_edges: List, shared: Set) -> Set:
        """Internal (non-shared) nodes reachable from a lane's boundary edges.

        :param state: The body state.
        :param lane_edges: The boundary edges (in this state) belonging to the lane.
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

        states = self._body_states(loop)
        if states is None:
            return False
        sym_offset = self._interstate_lane_symbols(loop, loop_var)
        if sym_offset is None:
            return False

        # Lane offset of every loop-dependent boundary edge, per state.
        edge_offsets: Dict[SDFGState, Dict] = {st: {} for st in states}
        for st in states:
            for edge in st.edges():
                if edge.data is None:
                    continue
                off, ok = self._edge_offset(edge.data.subset, loop_var, sym_offset)
                if not ok:
                    return False
                if off is not None:
                    edge_offsets[st][edge] = off

        all_offsets = set(sym_offset.values())
        for st in states:
            all_offsets |= set(edge_offsets[st].values())
        if not all_offsets:
            return False

        # The distinct lane offsets must be equally spaced from 0:
        # ``{0, g, 2g, ..., (m-1)g}`` -- ``m`` lanes, spacing ``g``.
        distinct = sorted(all_offsets)
        m = len(distinct)
        if m < 2 or distinct[0] != 0:
            return False
        g = distinct[1]
        if g <= 0 or distinct != [j * g for j in range(m)]:
            return False

        # Position-coverage safety: ``step > m*g`` leaves gaps the re-rolled
        # step-``g`` loop would wrongly fill; ``step < m*g`` overlaps, so a
        # position is touched by more than one lane -- safe only when no array is
        # both read and written (else the re-roll changes a read-modify-write count).
        if step > m * g:
            return False
        if step < m * g and self._has_read_modify_write(states, edge_offsets):
            return False

        # Per state, walk each lane's component. Two patterns are accepted:
        #
        # 1. **Disjoint lanes** (the classic ``s351``/``s353`` shape): every node
        #    reached from lane ``k``'s boundary edges is reachable from *only*
        #    lane ``k``. Lanes are structurally identical (same tasklet-code
        #    multiset).
        # 2. **Lanes merged through an associative reduction tree** (the
        #    ``s352`` single-expression shape): the lane components overlap only
        #    at binary associative-op tasklets (``+``, ``*``, ``min``, ``max``)
        #    that form a tree combining the ``m`` lane outputs into a single
        #    value. The classifier verifies the shared nodes are uniformly one
        #    associative op; the rewrite then collapses the tree to lane 0.
        shared = {st: self._shared_nodes(st) for st in states}
        lane_nodes: Dict[SDFGState, Dict[int, Set]] = {st: {} for st in states}
        merge_nodes: Dict[SDFGState, Set] = {st: set() for st in states}
        codes: Dict[int, List[str]] = {d: [] for d in distinct}
        for st in states:
            per_lane_edges: Dict[int, List] = {d: [] for d in distinct}
            for edge, off in edge_offsets[st].items():
                per_lane_edges[off].append(edge)
            # Walk each lane independently, recording which lanes reach each node.
            visited_by: Dict = {}
            for d in distinct:
                for n in self._lane_nodes(st, per_lane_edges[d], shared[st]):
                    visited_by.setdefault(n, set()).add(d)
            # Classify: a node reached by exactly one lane is part of that lane's
            # unique component; a node reached by >=2 lanes is a merge candidate
            # and must be an associative binary-op tasklet.
            per_lane_unique: Dict[int, Set] = {d: set() for d in distinct}
            merges: Set = set()
            merge_op: Optional[str] = None
            for n, lanes_visiting in visited_by.items():
                if len(lanes_visiting) == 1:
                    per_lane_unique[next(iter(lanes_visiting))].add(n)
                else:
                    op = self._associative_op_kind(n)
                    if op is None:
                        return False
                    if merge_op is None:
                        merge_op = op
                    elif merge_op != op:
                        return False  # mixed ops in the merge tree
                    merges.add(n)
            lane_nodes[st] = per_lane_unique
            merge_nodes[st] = merges
            for d in distinct:
                codes[d].extend(n.code.as_string for n in per_lane_unique[d] if isinstance(n, nodes.Tasklet))
        codes = {d: sorted(c) for d, c in codes.items()}
        if not codes[0] or any(codes[d] != codes[0] for d in distinct[1:]):
            return False

        # Re-roll: drop every lane but offset 0 -- its unique nodes in each state,
        # the per-lane interstate symbol assignments. Then collapse the merge
        # tree: each merge tasklet has its dropped-lane input gone, so it now
        # has only one live input; splice it (consumers of its output read from
        # that one input directly), and delete the merge tasklet + its output
        # AccessNode. Finally, rewrite the loop to step ``g``.
        drop_offsets = set(distinct[1:])
        drop_syms = {s for s, o in sym_offset.items() if o in drop_offsets}
        for st in states:
            for d in drop_offsets:
                for edge in list(edge_offsets[st]):
                    if edge_offsets[st][edge] == d and edge in st.edges():
                        st.remove_edge(edge)
                for n in lane_nodes[st][d]:
                    if n in st.nodes():
                        st.remove_node(n)
            self._collapse_merge_tree(st, merge_nodes[st])
            # Per-lane AccessNode copies the frontend split out are now isolated.
            for n in list(st.nodes()):
                if isinstance(n, nodes.AccessNode) and st.degree(n) == 0:
                    st.remove_node(n)
        for edge in loop.edges():
            for s in drop_syms:
                edge.data.assignments.pop(s, None)

        self._rewrite_step(loop, loop_var, g, m)
        return True

    def _collapse_merge_tree(self, state: SDFGState, merges: Set) -> None:
        """Splice each surviving merge tasklet to passthrough its one live input.

        After the dropped-lane nodes are removed, each merge tasklet in the
        original reduction tree has lost one of its two input sources (the
        edge from a now-removed dropped-lane producer). The surviving input
        (lane 0's product, or another merge that has already been collapsed)
        flows through. We rewrite consumers of the merge tasklet's output
        AccessNode to read directly from the live input's source, then delete
        the merge tasklet and its output AccessNode.

        :param state: The body state being collapsed.
        :param merges: The set of merge tasklets identified by the classifier;
                       safe to pass merges already deleted by an upstream pass
                       (membership is re-checked).
        """
        # Process merges in topological order (closer to the leaves first) so a
        # later merge sees its already-collapsed predecessors as direct edges.
        try:
            from dace.sdfg.utils import dfs_topological_sort
            order = [n for n in dfs_topological_sort(state) if n in merges]
        except Exception:
            order = list(merges)
        for m in order:
            if m not in state.nodes():
                continue
            in_edges = list(state.in_edges(m))
            out_edges = list(state.out_edges(m))
            # A live merge has exactly one of its 2 inputs still wired (the
            # other lane's chain was just removed). Anything else is unsafe to
            # touch heuristically -- leave it (the caller's structural check
            # already vetted the pattern).
            if len(in_edges) != 1 or len(out_edges) != 1:
                continue
            live_in = in_edges[0]
            out_acc = out_edges[0].dst
            if not isinstance(out_acc, nodes.AccessNode):
                continue
            # Redirect every consumer of ``out_acc`` to read from ``live_in.src``
            # with that source's port. Preserve the original consumer-side
            # connector and memlet (rebased onto the surviving source's data
            # name when needed).
            for ce in list(state.out_edges(out_acc)):
                new_memlet = ce.data
                if (new_memlet is not None and isinstance(live_in.src, nodes.AccessNode)
                        and new_memlet.data == out_acc.data):
                    new_memlet = dace.Memlet(data=live_in.src.data, subset=new_memlet.subset, wcr=new_memlet.wcr)
                state.remove_edge(ce)
                state.add_edge(live_in.src, live_in.src_conn, ce.dst, ce.dst_conn, new_memlet)
            state.remove_node(m)
            if state.degree(out_acc) == 0:
                state.remove_node(out_acc)

    def _has_read_modify_write(self, states: List[SDFGState], edge_offsets: Dict) -> bool:
        """Whether any array is both read and written by the lanes.

        :param states: The body states.
        :param edge_offsets: ``{state: {edge: offset}}`` for the lane edges.
        :returns: ``True`` if some array appears both as a read source and a
                  write destination among the lane boundary edges.
        """
        reads, writes = set(), set()
        for st in states:
            for edge in edge_offsets[st]:
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
