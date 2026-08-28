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

A lane that READS PAST its own position -- TSVC ``s116``, ``a[i+k] = a[i+k+1] *
a[i+k]`` -- does not fit that per-edge classifier: its ``m`` lanes carry ``m + 1``
distinct offsets, so ``step < m*g`` and the read-modify-write refusal fires.
:meth:`RerollUnrolledLoops._try_reroll_chain` handles it on a second pass, keying
a lane on the VALUE COMPONENT it forms and on the offset that component stores
at, with the read-ahead as one more boundary edge at a relative offset every lane
must agree on. It additionally requires the lanes to RUN in ascending offset
order, which is what makes the read-ahead see the same value in both forms.
"""

import re
from typing import Dict, List, Optional, Tuple

import dace
from dace import symbolic
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.state import LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation.transformation import explicit_cf_compatible
from dace.transformation.passes.analysis import loop_analysis


def value_edges(edges) -> List:
    """The edges carrying a VALUE; an empty memlet is an ordering edge and carries none."""
    return [e for e in edges if e.data is not None and not e.data.is_empty()]


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
    if loop_var not in (str(s) for s in expr.free_symbols):
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

        Runs to a fixpoint: re-rolling one loop can expose a sibling or an
        enclosing loop that only became a clean lane chain after the inner
        collapse (a manually multi-level / nested-tiled unroll). Bounded by the
        loop count so it always terminates.

        :param sdfg: SDFG to mutate in place.
        :returns: The number of loops re-rolled, or ``None`` if none.
        """
        total = 0
        max_iters = 1 + sum(1 for sd in sdfg.all_sdfgs_recursive()
                            for r in sd.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion))
        for _ in range(max_iters):
            rerolled = 0
            for sd in sdfg.all_sdfgs_recursive():
                for cfg in list(sd.all_control_flow_regions(recursive=True)):
                    if isinstance(cfg, LoopRegion) and (self._try_reroll(cfg) or self._try_reroll_chain(cfg)
                                                        or self._try_reroll_accumulator_reduction(cfg)):
                        rerolled += 1
            if rerolled == 0:
                break
            total += rerolled
        return total or None

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
        names = dict.fromkeys(str(s) for s in subset.free_symbols)
        if loop_var in names:
            off = _index_offset(subset, loop_var)
            return (off, off is not None)
        lane_syms = dict.fromkeys(n for n in names if n in sym_offset)
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
        """If ``node`` is a binary associative-op tasklet over its two inputs,
        return its op kind (``'+'``, ``'*'``, ``'min'``, ``'max'``); else ``None``.

        Lifts the tasklet RHS to a :mod:`dace.symbolic` expression -- sympy's
        ``Add`` / ``Mul`` / ``Min`` / ``Max`` are associative by construction,
        so we don't need to enumerate parenthesisation or whitespace variants.
        The operands are the tasklet's own connectors rather than a fixed
        ``__in1``/``__in2`` pair: the frontend names a binop's inputs that way
        but a ``min``/``max`` call's ``__in_a``/``__in_b``.
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
        conns = dict.fromkeys(node.in_connectors)
        if len(conns) != 2 or dict.fromkeys(str(s) for s in expr.free_symbols) != conns:
            return None
        if len(expr.args) != 2:
            return None
        return self._ASSOC_SYMPY_KIND.get(type(expr).__name__)

    def _is_transparent_spine(self, state: SDFGState, node) -> bool:
        """Whether ``node`` carries a value through the reduction without itself
        being a lane-unique computation or an associative merge op.

        A manually-unrolled reduction (``acc += f(i) + f(i+g) + ...``; TSVC
        ``s352`` dot, ``s31111`` sum) lowers to an SSA left-fold: associative
        merge tasklets (``__out = __in1 + __in2``) interleaved with two kinds of
        pass-through carriers that are reached by every lane but are NOT merges:

        * a **transient AccessNode** -- the SSA value handed from one merge
          tasklet to the next (``a_slice_b_slice_plus_...``, ``partial_v_k``);
        * a single-input **pass-through tasklet** (``__out = __inp``) -- the
          accumulator copy-back the frontend stages (``dot = __inp``).

        Treating these as a transparent spine (rather than a disqualifying
        non-associative merge) is what lets the chain collapse to lane 0.
        """
        if isinstance(node, nodes.AccessNode):
            desc = state.sdfg.arrays.get(node.data)
            return desc is not None and desc.transient
        if isinstance(node, nodes.Tasklet):
            code = node.code.as_string.strip()
            if '=' not in code:
                return False
            lhs, rhs = code.split('=', 1)
            if lhs.strip() != '__out':
                return False
            try:
                expr = symbolic.pystr_to_symbolic(rhs.strip())
            except Exception:
                return False
            return dict.fromkeys(str(s) for s in expr.free_symbols) == dict.fromkeys(['__inp'])
        return False

    def _carried_accumulator_op(self, state: SDFGState, acc: str) -> Optional[str]:
        """Op kind of a lane's ``acc = acc OP x`` update, or ``None`` if it is not that shape.

        ``acc`` must be read once at the top of ``state``, combined by exactly one associative
        tasklet, and handed to the write through transparent spine carriers only.
        """
        acc_nodes = [n for n in state.nodes() if isinstance(n, nodes.AccessNode) and n.data == acc]
        src = [n for n in acc_nodes if state.in_degree(n) == 0]
        snk = [n for n in acc_nodes if state.out_degree(n) == 0]
        if len(acc_nodes) != 2 or len(src) != 1 or len(snk) != 1 or state.out_degree(src[0]) != 1:
            return None
        node = state.out_edges(src[0])[0].dst
        op = self._associative_op_kind(node)
        if op is None:
            return None
        while node is not snk[0]:
            out = state.out_edges(node)
            if len(out) != 1:
                return None
            node = out[0].dst
            if node is not snk[0] and not (self._is_transparent_spine(state, node) and state.in_degree(node) == 1):
                return None
        return op

    def _accumulators_folded_once(self, loop: LoopRegion, accs: Dict[str, None], op: str) -> bool:
        """Whether the loop's per-lane accumulators are consumed by ONE binary ``op`` tree.

        Sound re-roll of ``s0 OP= f(i); ...; s(m-1) OP= f(i+(m-1)g)`` into lane 0 alone relies on
        the dropped accumulators keeping their pre-loop value and re-entering through that fold,
        so their only use after the loop must be a single ``m``-leaf tree of the SAME op (FP
        policy: reassociation is sanctioned WITHIN a reduction combining op).
        """
        sdfg = loop.sdfg
        body = dict.fromkeys(loop.all_control_flow_blocks(recursive=True))
        if any(a in e.data.free_symbols for e in sdfg.all_interstate_edges(recursive=True) for a in accs):
            return False
        leaves: Dict = {}
        seen_names: Dict[str, None] = {}
        host: Dict[SDFGState, None] = {}
        for st in sdfg.all_states():
            if st in body:
                continue
            for n in st.nodes():
                if not (isinstance(n, nodes.AccessNode) and n.data in accs and st.out_degree(n) > 0):
                    continue
                # An unroll REMAINDER loop accumulating into the same lane with the same op does
                # not consume the folded value, it contributes one more term to it.
                if self._carried_accumulator_op(st, n.data) == op:
                    continue
                if st.out_degree(n) != 1 or n.data in seen_names:
                    return False
                leaves[n] = None
                seen_names[n.data] = None
                host[st] = None
        if len(leaves) != len(accs) or len(host) != 1:
            return False
        state = next(iter(host))
        # Forward closure over the fold: ``op`` tasklets chained through single-consumer scalars.
        # A consumer that is not such a tasklet ends the walk -- it is the tree root leaving the
        # fold -- and the counting check below rejects it if the fold was not complete.
        ops: Dict = {}
        inside = dict.fromkeys(leaves)
        frontier = list(leaves)
        while frontier:
            t = state.out_edges(frontier.pop())[0].dst
            if self._associative_op_kind(t) != op or state.in_degree(t) != 2 or state.out_degree(t) != 1:
                continue
            if t in ops:  # its other operand reached it first
                continue
            ops[t] = None
            nxt = state.out_edges(t)[0].dst
            if not isinstance(nxt, nodes.AccessNode) or state.out_degree(nxt) > 1:
                return False
            inside[nxt] = None
            if state.out_degree(nxt) == 1:
                frontier.append(nxt)
        # ``m - 1`` binary ops over ``m`` single-consumer leaves, every operand internal: one tree.
        return len(ops) == len(accs) - 1 and all(e.src in inside for t in ops for e in state.in_edges(t))

    def _shared_nodes(self, state: SDFGState) -> Dict:
        """Boundary access nodes of a state (external arrays + read-only sources).

        :param state: The body state.
        :returns: The AccessNodes the lane traversal must stop at.
        """
        shared: Dict = {}
        for n in state.nodes():
            if not isinstance(n, nodes.AccessNode):
                continue
            desc = state.sdfg.arrays.get(n.data)
            # Value edges only: an ordering memlet is no producer, so it makes no lane boundary.
            if (desc is not None and
                    not desc.transient) or not value_edges(state.in_edges(n)) or not value_edges(state.out_edges(n)):
                shared[n] = None
        return shared

    def _lane_nodes(self, state: SDFGState, lane_edges: List, shared: Dict) -> Dict:
        """Internal (non-shared) nodes reachable from a lane's boundary edges.

        :param state: The body state.
        :param lane_edges: The boundary edges (in this state) belonging to the lane.
        :param shared: Access nodes shared across lanes (excluded from the walk).
        :returns: The set of internal nodes that belong only to this lane.

        Walks VALUE edges only -- an ordering memlet fixes order, never lane membership.
        """
        seen: Dict = {}
        frontier = []
        for e in lane_edges:
            for n in (e.src, e.dst):
                if n not in shared:
                    frontier.append(n)
        while frontier:
            n = frontier.pop()
            if n in seen:
                continue
            seen[n] = None
            for e in value_edges(state.in_edges(n)):
                if e.src not in shared:
                    frontier.append(e.src)
            for e in value_edges(state.out_edges(n)):
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

        all_offsets = dict.fromkeys(sym_offset.values())
        for st in states:
            all_offsets.update(dict.fromkeys(edge_offsets[st].values()))
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
        lane_nodes: Dict[SDFGState, Dict[int, Dict]] = {st: {} for st in states}
        merge_nodes: Dict[SDFGState, Dict] = {st: {} for st in states}
        codes: Dict[int, List[str]] = {d: [] for d in distinct}
        for st in states:
            per_lane_edges: Dict[int, List] = {d: [] for d in distinct}
            for edge, off in edge_offsets[st].items():
                per_lane_edges[off].append(edge)
            # Walk each lane independently, recording which lanes reach each node.
            visited_by: Dict = {}
            for d in distinct:
                # sorted by node id: ``_lane_nodes`` returns a set of NODE objects (hashed by id(), so its
                # order varies with allocation), and this insertion order decides which node ``merge_op``
                # locks onto below -- i.e. whether the re-roll fires at all.
                for n in sorted(self._lane_nodes(st, per_lane_edges[d], shared[st]), key=st.node_id):
                    visited_by.setdefault(n, {})[d] = None
            # Classify: a node reached by exactly one lane is part of that lane's
            # unique component; a node reached by >=2 lanes is a merge candidate
            # and must be an associative binary-op tasklet.
            per_lane_unique: Dict[int, Dict] = {d: {} for d in distinct}
            merges: Dict = {}
            merge_op: Optional[str] = None
            for n, lanes_visiting in visited_by.items():
                if len(lanes_visiting) == 1:
                    per_lane_unique[next(iter(lanes_visiting))][n] = None
                else:
                    op = self._associative_op_kind(n)
                    if op is None:
                        return False
                    if merge_op is None:
                        merge_op = op
                    elif merge_op != op:
                        return False  # mixed ops in the merge tree
                    merges[n] = None
            lane_nodes[st] = per_lane_unique
            merge_nodes[st] = merges
            for d in distinct:
                # Which arrays a lane reads, not just what its tasklets say. Boundary AccessNodes are
                # excluded from a lane component by construction, so on code alone two lanes reading
                # DIFFERENT arrays look identical and the re-roll drops a real access. ``_lane_sig``
                # below folds in the read names for the same reason.
                codes[d].extend(sorted(e.src.data for e in per_lane_edges[d] if isinstance(e.src, nodes.AccessNode)))
                codes[d].extend(sorted(e.dst.data for e in per_lane_edges[d] if isinstance(e.dst, nodes.AccessNode)))
                codes[d].extend(n.code.as_string for n in per_lane_unique[d] if isinstance(n, nodes.Tasklet))
        codes = {d: sorted(c) for d, c in codes.items()}
        if not codes[0] or any(codes[d] != codes[0] for d in distinct[1:]):
            return False

        # Per-lane carried accumulators -- the ``s0 OP= f(i); s1 OP= f(i+g); ...; s = s0 OP s1 ...``
        # reduction shape. A lane whose private component writes a shared container at a LOOP-
        # INVARIANT subset leaves a carried value live past the loop; dropping the lane moves its
        # accumulation into lane 0, so the value only survives when every such accumulator is
        # folded after the loop by one tree of the same associative op. Without that proof the
        # re-roll silently rewrites what the lanes computed (``c[0] = s0; c[1] = s1``).
        carried: Dict[int, Dict[str, SDFGState]] = {d: {} for d in distinct}
        for st in states:
            for d in distinct:
                for n in lane_nodes[st][d]:
                    for e in st.out_edges(n):
                        if isinstance(e.dst, nodes.AccessNode) and e.dst in shared[st] and e not in edge_offsets[st]:
                            carried[d][e.dst.data] = st
        acc: Dict[int, Tuple[str, SDFGState]] = {}
        for d in distinct:
            if len(carried[d]) > 1:
                return False
            for name, st in carried[d].items():
                acc[d] = (name, st)
        if acc:
            names = dict.fromkeys(name for name, _ in acc.values())
            if len(acc) != m or len(names) != m:
                return False
            ops = dict.fromkeys(self._carried_accumulator_op(st, name) for name, st in acc.values())
            if len(ops) != 1 or next(iter(ops)) is None:
                return False
            if not self._accumulators_folded_once(loop, names, next(iter(ops))):
                return False

        # Re-roll: drop every lane but offset 0 -- its unique nodes in each state,
        # the per-lane interstate symbol assignments. Then collapse the merge
        # tree: each merge tasklet has its dropped-lane input gone, so it now
        # has only one live input; splice it (consumers of its output read from
        # that one input directly), and delete the merge tasklet + its output
        # AccessNode. Finally, rewrite the loop to step ``g``.
        drop_offsets = dict.fromkeys(distinct[1:])  # already sorted -- dict.fromkeys keeps that order
        drop_syms = dict.fromkeys(s for s, o in sym_offset.items() if o in drop_offsets)
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

    def _cut_nodes(self, state: SDFGState) -> Dict:
        """AccessNodes of the loop's VISIBLE containers -- where the value walk stops.

        A lane's private values live in transients; everything a lane reads from or stores to the
        outside world crosses one of these nodes, which is what makes them the component boundary.

        :param state: The body state.
        :returns: The non-transient AccessNodes of ``state``.
        """
        arrays = state.sdfg.arrays
        return dict.fromkeys(n for n in state.nodes()
                             if isinstance(n, nodes.AccessNode) and not (n.data in arrays and arrays[n.data].transient))

    def _cut_side_subset(self, edge, cut_node):
        """The subset of ``edge`` that indexes ``cut_node``'s container.

        :param edge: A boundary edge of a lane component.
        :param cut_node: The boundary AccessNode the edge touches.
        :returns: The memlet subset addressing ``cut_node``'s array, or ``None``.
        """
        mem = edge.data
        if mem is None or mem.data is None:
            return None
        return mem.subset if mem.data == cut_node.data else mem.other_subset

    def _topological_index(self, state: SDFGState) -> Dict[int, int]:
        """Position of every node of ``state`` in a topological order, by node id.

        :param state: The body state.
        :returns: ``{node_id: position}``, unreachable nodes appended in insertion order.
        """
        try:
            order = list(sdutil.dfs_topological_sort(state))
        except Exception:
            order = []
        idx: Dict[int, int] = {}
        for n in order:
            idx.setdefault(state.node_id(n), len(idx))
        for n in state.nodes():
            idx.setdefault(state.node_id(n), len(idx))
        return idx

    def _value_components(self, states: List[SDFGState], cut: List[Dict]) -> List[List[Tuple[int, nodes.Node]]]:
        """Partition the body's non-boundary nodes into connected VALUE components.

        Two nodes are in one component when a value flows between them: along a non-empty memlet
        that stays clear of the boundary, or through two AccessNodes of the same transient (a
        component crosses a state boundary exactly there, which is how the frontend hands a lane's
        read-ahead to the state that consumes it). Empty memlets are ordering edges and carry no
        value, so they never join two components -- in ``s116`` they are precisely what ties one
        lane's operands to the next lane's, and honouring them would fuse the whole body into one.

        :param states: The body states.
        :param cut: Per-state boundary AccessNodes, from :meth:`_cut_nodes`.
        :returns: One list of ``(state index, node)`` pairs per component, deterministically ordered.
        """
        parent: Dict[Tuple[int, int], Tuple[int, int]] = {}

        def find(key):
            while parent[key] != key:
                parent[key] = parent[parent[key]]
                key = parent[key]
            return key

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for si, st in enumerate(states):
            for n in st.nodes():
                if n not in cut[si]:
                    key = (si, st.node_id(n))
                    parent[key] = key
        for si, st in enumerate(states):
            for e in st.edges():
                if e.data is None or e.data.is_empty() or e.src in cut[si] or e.dst in cut[si]:
                    continue
                union((si, st.node_id(e.src)), (si, st.node_id(e.dst)))
        first_seen: Dict[str, Tuple[int, int]] = {}
        for si, st in enumerate(states):
            for n in st.nodes():
                if not isinstance(n, nodes.AccessNode) or n in cut[si]:
                    continue
                key = (si, st.node_id(n))
                union(first_seen.setdefault(n.data, key), key)
        groups: Dict[Tuple[int, int], List[Tuple[int, nodes.Node]]] = {}
        for key in sorted(parent):
            groups.setdefault(find(key), []).append((key[0], states[key[0]].node(key[1])))
        return list(groups.values())

    def _try_reroll_chain(self, loop: LoopRegion) -> bool:
        """Re-roll a lane chain whose lanes READ PAST their own position (TSVC ``s116``).

        :meth:`_try_reroll` keys a lane on the offset an edge carries, so a lane that reads
        ``a[i + k + 1]`` while storing to ``a[i + k]`` contributes TWO offsets and the ``m`` lanes
        look like ``m + 1`` of them -- ``step < m*g`` then meets the read-modify-write refusal and
        the chain is left unrolled. Here a lane is a VALUE COMPONENT of the body instead, and its
        offset is the offset it STORES at; the read-ahead is one more boundary edge of that
        component, at a relative offset every lane must agree on.

        Legality, given ``m`` lanes at ``{0, g, ..., (m-1)g}`` with ``step == m*g`` (every position
        stored exactly once) and one shared set of relative offsets: lane ``k`` reading relative
        ``r > 0`` reads a position lane ``k + r/g`` has not stored yet, and the re-rolled step-``g``
        loop reads that same position ``r/g`` iterations early -- also unstored. For ``r < 0`` both
        forms read a position already stored. That holds only while the lanes RUN in ascending
        offset order, which is checked: in a descending body lane ``k``'s read-ahead would see a
        stored value the ascending re-rolled loop does not.

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
        # A per-lane interstate symbol names the offset of a lane's READ, not of its store; that
        # shape (the ``s353`` gather) is ``_try_reroll``'s and this path does not model it.
        if self._interstate_lane_symbols(loop, loop_var):
            return False

        cut = [self._cut_nodes(st) for st in states]
        comps = self._value_components(states, cut)
        if not comps:
            return False
        owner: Dict[Tuple[int, int], int] = {}
        for ci, comp in enumerate(comps):
            for si, node in comp:
                owner[(si, states[si].node_id(node))] = ci

        bounds: List[List] = [[] for _ in comps]
        for si, st in enumerate(states):
            for e in st.edges():
                if e.data is None or e.data.is_empty():
                    continue
                is_read, stores = e.src in cut[si], e.dst in cut[si]
                if is_read and stores:
                    return False  # a bare container-to-container copy belongs to no lane
                if not is_read and not stores:
                    continue
                inner = e.dst if is_read else e.src
                bounds[owner[(si, st.node_id(inner))]].append((si, e, e.src if is_read else e.dst, is_read))

        topo = [self._topological_index(st) for st in states]
        lanes: Dict[int, int] = {}
        sigs: Dict[int, List[str]] = {}
        order: Dict[int, Tuple[int, int]] = {}
        for ci, comp in enumerate(comps):
            offsets = []
            for si, e, node, is_read in bounds[ci]:
                off, ok = self._edge_offset(self._cut_side_subset(e, node), loop_var, {})
                # A loop-invariant READ (a scalar coefficient) is shared by every lane and stays;
                # a loop-invariant STORE is a carried value the re-roll would fold into one lane.
                if not ok or (off is None and not is_read):
                    return False
                offsets.append((is_read, node.data, off, si, e))
            stored_at = dict.fromkeys(off for is_read, _, off, _, _ in offsets if not is_read)
            if len(stored_at) != 1:
                return False
            d = next(iter(stored_at))
            if d in lanes:
                return False
            lanes[d] = ci
            sigs[d] = sorted('%s|%s|%s' % ('r' if is_read else 'w', name, 'inv' if off is None else off - d)
                             for is_read, name, off, _, _ in offsets)
            sigs[d] += sorted(n.code.as_string for _, n in comp if isinstance(n, nodes.Tasklet))
            order[d] = min((si, topo[si][states[si].node_id(e.dst)]) for is_read, _, _, si, e in offsets if not is_read)

        distinct = sorted(lanes)
        m = len(distinct)
        if m < 2 or distinct[0] != 0:
            return False
        g = distinct[1]
        if g <= 0 or distinct != [j * g for j in range(m)]:
            return False
        # Exact coverage only: a gap would leave positions the step-``g`` loop wrongly fills, an
        # overlap would store a position twice, and both break the read-ahead argument above.
        if step != m * g:
            return False
        if any(sigs[d] != sigs[0] for d in distinct[1:]):
            return False
        if sorted(distinct, key=lambda d: order[d]) != distinct:
            return False

        for d in distinct[1:]:
            for si, node in comps[lanes[d]]:
                st = states[si]
                if node in st.nodes():
                    st.remove_node(node)
        for st in states:
            for n in list(st.nodes()):
                if isinstance(n, nodes.AccessNode) and st.degree(n) == 0:
                    st.remove_node(n)
        self._rewrite_step(loop, loop_var, g, m)
        return True

    def _try_reroll_accumulator_reduction(self, loop: LoopRegion) -> bool:
        """Re-roll a hand-unrolled *reduction* into a single-lane step-``g`` loop.

        The lane-decomposition path (:meth:`_try_reroll`) cannot handle a manually
        unrolled reduction (TSVC ``s352`` dot, ``s31111`` sum): the ``m`` lanes are
        joined by an associative left-fold into one carried scalar accumulator, so
        a bidirectional lane walk reaches every node from every lane (no separable
        lane component). This matcher instead follows the fold dataflow with a
        FORWARD-only reach from each lane's reads:

        * a node reached forward from exactly one offset is that lane's private
          computation (``s352``'s per-lane ``_Mult_`` ``a[i+k]*b[i+k]``);
        * a node reached from >= 2 offsets is a fold node -- it must be an
          associative merge tasklet of the single commutative fold op, or a
          transparent spine carrier (SSA transient / ``__out = __inp`` copy-back).

        Soundness: the loop must read each covered position exactly once
        (``step == m*g``), write exactly one carried scalar accumulator (a pure
        reduction with no other output), reduce with a commutative-associative op,
        and have structurally identical lanes (same read arrays + same private
        non-fold ops per offset). Keeping lane 0 and re-rolling to step ``g`` then
        recomputes the identical reduction over every position. The GC-splice
        rewrite drops the other lanes' reads, deletes their now-broken private ops,
        and collapses each fold tasklet that lost a term to its surviving input.

        :param loop: The candidate loop region.
        :returns: ``True`` if it matched and was re-rolled.
        """
        loop_var = loop.loop_variable
        if not loop_var:
            return False
        stride = loop_analysis.get_loop_stride(loop)
        step = _const_int(stride) if stride is not None else None
        if step is None or step < 2:
            return False
        states = self._body_states(loop)
        if states is None or len(states) != 1:
            return False
        st = states[0]
        # A pure in-state reduction: no per-lane interstate gather symbols.
        if self._interstate_lane_symbols(loop, loop_var):
            return False

        # Lane offsets of the loop-dependent READ edges (src is an AccessNode).
        per_lane_edges: Dict[int, List] = {}
        for edge in st.edges():
            if edge.data is None or not isinstance(edge.src, nodes.AccessNode):
                continue
            off, ok = self._edge_offset(edge.data.subset, loop_var, {})
            if not ok:
                return False
            if off is not None:
                per_lane_edges.setdefault(off, []).append(edge)
        distinct = sorted(per_lane_edges)
        m = len(distinct)
        if m < 2 or distinct[0] != 0:
            return False
        g = distinct[1]
        if g <= 0 or distinct != [j * g for j in range(m)]:
            return False
        # Contiguous, no-overlap coverage: each position read exactly once.
        if step != m * g:
            return False

        # Exactly one carried scalar accumulator: a data name with both a source
        # (in-degree 0) and a sink (out-degree 0) AccessNode in the state, scalar,
        # and the loop's only sink -> the reduction has no other output.
        sinks = [n for n in st.nodes() if isinstance(n, nodes.AccessNode) and st.out_degree(n) == 0]
        sources = dict.fromkeys(n.data for n in st.nodes() if isinstance(n, nodes.AccessNode) and st.in_degree(n) == 0)
        if len(sinks) != 1 or sinks[0].data not in sources:
            return False
        acc_desc = st.sdfg.arrays.get(sinks[0].data)
        if acc_desc is None or acc_desc.total_size != 1:
            return False

        # Forward reach: which offsets reach each node (out-edges only).
        reached_by: Dict = {}
        for d in distinct:
            frontier = [e.dst for e in per_lane_edges[d]]
            seen: Dict = {}
            while frontier:
                n = frontier.pop()
                if n in seen:
                    continue
                seen[n] = None
                reached_by.setdefault(n, {})[d] = None
                for e in st.out_edges(n):
                    frontier.append(e.dst)

        # Classify fold nodes (>=2 offsets): each must be an associative merge of
        # ONE commutative op, or a transparent spine carrier.
        fold_op: Optional[str] = None
        for n, offs in reached_by.items():
            if len(offs) < 2:
                continue
            op = self._associative_op_kind(n)
            if op is None:
                if self._is_transparent_spine(st, n):
                    continue
                return False
            if fold_op is None:
                fold_op = op
            elif fold_op != op:
                return False
        if fold_op is None:
            return False

        # Per-offset lane signature: read arrays + private non-fold tasklet codes.
        def _lane_sig(d: int) -> Tuple:
            reads = sorted(e.src.data for e in per_lane_edges[d])
            priv = sorted(n.code.as_string for n, offs in reached_by.items()
                          if len(offs) == 1 and d in offs and isinstance(n, nodes.Tasklet)
                          and self._associative_op_kind(n) != fold_op and not self._is_transparent_spine(st, n))
            return (tuple(reads), tuple(priv))

        sig0 = _lane_sig(0)
        if not sig0[0] or any(_lane_sig(d) != sig0 for d in distinct[1:]):
            return False

        # Rewrite. Drop the other lanes' read edges (this also unhooks an
        # edge-only lane, e.g. s31111's bare ``a[i + k]`` feeding the shared fold),
        # then delete every node reached ONLY from dropped offsets -- the dropped
        # lanes' private computation (s352's per-lane ``_Mult_`` and its product) --
        # which is sound because such a node contributes to no surviving lane.
        # Finally collapse each fold tasklet that lost a term to its one surviving
        # input. Lane 0's private ops (reached from offset 0) and the carried
        # accumulator spine are untouched.
        drop_offsets = dict.fromkeys(distinct[1:])  # already sorted -- dict.fromkeys keeps that order
        for d in distinct[1:]:
            for edge in per_lane_edges[d]:
                if edge in st.edges():
                    st.remove_edge(edge)
        for n in list(st.nodes()):
            offs = reached_by.get(n)
            if offs and all(o in drop_offsets for o in offs):
                st.remove_node(n)
        self._collapse_merge_tree(st, dict.fromkeys(n for n in st.nodes() if self._associative_op_kind(n) == fold_op))
        for n in list(st.nodes()):
            if isinstance(n, nodes.AccessNode) and st.degree(n) == 0:
                st.remove_node(n)

        self._rewrite_step(loop, loop_var, g, m)
        return True

    def _collapse_merge_tree(self, state: SDFGState, merges: Dict) -> None:
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
            # ``merges`` is a set of NODE objects -- hashed by id(), so raw iteration order varies with
            # allocation (not even PYTHONHASHSEED-stable). Order is load-bearing here (see the comment
            # above: a later merge must see its collapsed predecessors), so fall back to a stable key.
            order = sorted(merges, key=state.node_id)
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
        reads: Dict = {}
        writes: Dict = {}
        for st in states:
            for edge in edge_offsets[st]:
                if isinstance(edge.src, nodes.AccessNode):
                    reads[edge.src.data] = None
                if isinstance(edge.dst, nodes.AccessNode):
                    writes[edge.dst.data] = None
        return any(a in writes for a in reads)

    def _rewrite_step(self, loop: LoopRegion, loop_var: str, g: int, m: int) -> None:
        """Rewrite a step-``S`` loop to step ``g`` over the flattened range.

        The original loop runs ``loop_var = init, init + S, ..., last_i`` where
        ``last_i`` is the LAST iteration value -- which is ``init + S *
        floor((end - init) / S)``, NOT ``end`` itself when the range is not a
        multiple of the step (``get_loop_end`` returns the largest *value* below
        the bound, ignoring step alignment). Lane 0 of that last iteration sweeps
        up to ``last_i + (m - 1) * g``, so the re-rolled step-``g`` loop's
        exclusive bound is ``last_i + m * g``. Using ``end`` directly would
        over-cover the unaligned tail by extra positions the original loop never
        visits -- for a reduction that silently adds spurious terms (TSVC s352:
        ``for i in range(0, LEN_1D - 4, 5)`` skips the final partial group).

        :param loop: The loop region to rewrite.
        :param loop_var: The loop variable name.
        :param g: The lane-offset spacing (the new step).
        :param m: The lane count.
        """
        loop_end = symbolic.pystr_to_symbolic(loop_analysis.get_loop_end(loop))
        init = symbolic.pystr_to_symbolic(loop_analysis.get_init_assignment(loop))
        step = symbolic.pystr_to_symbolic(loop_analysis.get_loop_stride(loop))
        last_i = init + step * symbolic.int_floor(loop_end - init, step)
        new_excl = last_i + m * g
        loop.update_statement = dace.properties.CodeBlock(f"{loop_var} = {loop_var} + {g}")
        loop.loop_condition = dace.properties.CodeBlock(f"{loop_var} < ({symbolic.symstr(new_excl)})")
