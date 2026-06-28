# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Rewrite a scalar-accumulator loop as a parallel Map plus a ``Reduce`` libnode.

A loop whose body is a single read-modify-write of a constant-indexed
accumulator, e.g.::

    for i in range(start, end + 1):
        acc[c] = acc[c] OP g(other_inputs, i)

cannot be turned into a ``Map`` by ``LoopToMap`` because the constant-index
write/read pair conservatively reads as a loop-carried dependence. When ``OP``
is associative the iterations are reorderable, so the loop is equivalent to::

    for i: buf[i - start] = g(other_inputs, i)          # MAP-able
    acc[c] = reduce(OP, buf, identity=None)             # Reduce libnode

The Map computes the per-iteration deltas into a fresh transient ``buf`` indexed
by ``i - start`` (uniquely per iteration, so ``LoopToMap`` parallelizes it). The
``Reduce`` libnode then folds ``buf`` into ``acc[c]`` with ``identity=None``, so
the accumulator's pre-loop value is honoured (it is the seed of the fold).

This differs from :class:`~dace.transformation.passes.loop_to_reduce.LoopToReduce`
in two ways:

- The per-iteration value ``g(other_inputs, i)`` may be an arbitrary tasklet
  expression -- it does not have to be a single ``arr[f(i)]`` read. ``LoopToReduce``
  collapses straight to a ``Reduce`` over the source array, which requires that
  shape.
- The pass tolerates other per-iteration side effects in the loop body: the
  ``Reduce``-only rewrite would silently drop those, but here the rest of the
  body is preserved by the Map's first stage. ``LoopToReduce`` refuses such
  loops; this pass takes them.

Supported associative ``OP``: ``+``, ``*``, ``max``, ``min``, ``&``, ``|``, ``^``.

Loop body shape recognised: a single state, no maps or nested SDFGs, exactly
one non-transient scalar accumulator (a unique pair of read AccessNode +
write AccessNode with matching loop-invariant single-element subsets), and
exactly one *accumulator-update tasklet* found by walking back from the
carried-accumulator write AN through the frontend's copy-out chain
(intermediate transients + identity ``__out = __inp`` tasklets). The
accumulator-update tasklet has the form ``out = lhs OP rhs`` with one input
tracing back to the carried-accumulator read AN. Other (delta-computation)
tasklets are allowed in the body and are left untouched by the rewrite.
"""
import ast
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

from dace import SDFG, data, dtypes, memlet as mm, properties, subsets, symbolic
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis

#: Per-iteration delta buffer name prefix; ``find_new_name`` disambiguates.
_BUF_PREFIX = '_accum_buf_'

#: Map of AST operator type to associative WCR lambda string.
_BINOP_TO_WCR: Dict[type, str] = {
    ast.Add: 'lambda a, b: a + b',
    ast.Mult: 'lambda a, b: a * b',
    ast.BitAnd: 'lambda a, b: a & b',
    ast.BitOr: 'lambda a, b: a | b',
    ast.BitXor: 'lambda a, b: a ^ b',
}
_CALL_TO_WCR: Dict[str, str] = {
    'max': 'lambda a, b: max(a, b)',
    'min': 'lambda a, b: min(a, b)',
}


class _Match(NamedTuple):
    """A successfully matched accumulator loop.

    :param tasklet: The accumulator-update tasklet.
    :param state: The state containing the tasklet (the only state in the loop body).
    :param wcr: The reduction lambda string (key in :data:`_BINOP_TO_WCR` / :data:`_CALL_TO_WCR`).
    :param accum: The carried-accumulator data-descriptor name (the scalar that flows
                  out of the loop). May differ from the tasklet's direct write target
                  when the frontend stages the result through copy-out intermediates.
    :param accum_subset: The accumulator's loop-invariant single-element subset.
    :param accum_in_conn: The tasklet input connector that reads the accumulator.
    :param other_in_conn: The tasklet input connector that reads the per-iteration operand.
    :param out_conn: The tasklet output connector.
    :param accum_read_an: The AccessNode (possibly an intermediate transient) that the
                          tasklet's accumulator-input edge enters; for orphan cleanup.
    :param iter_start: The loop's start expression.
    :param iter_end: The loop's inclusive end expression.
    """
    tasklet: nodes.Tasklet
    state: SDFGState
    wcr: str
    accum: str
    accum_subset: subsets.Range
    accum_in_conn: str
    other_in_conn: str
    out_conn: str
    accum_read_an: nodes.AccessNode
    iter_start: Any
    iter_end: Any


@properties.make_properties
@xf.explicit_cf_compatible
class AccumulatorToMapAndReduce(ppl.Pass):
    """Rewrite scalar-accumulator loops as Map (delta buffer) + Reduce libnode."""

    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Memlets | ppl.Modifies.CFG | ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # The matcher recognises and skips loops whose accumulator output target is one
        # of this pass's own ``_accum_buf_`` transients, so re-application is a no-op.
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Rewrite every matching accumulator pattern in ``sdfg`` (and nested SDFGs).

        Two patterns are recognised:

        - **Sequential loop** -- a ``LoopRegion`` whose body has a scalar
          accumulator RMW (``acc[c] = acc[c] OP delta(...)``); rewritten into a
          buffer-writing Map plus a ``Reduce`` libnode.
        - **Parallel Map with WCR** -- a Map whose body writes to a scalar
          accumulator via a ``wcr=`` edge (typically what
          :class:`~dace.transformation.dataflow.wcr_conversion.AugAssignToWCR`
          plus a permissive lift produces). The same buffer-and-reduce shape
          replaces the WCR edge, which lets the existing Map run without an
          atomic write and downstream consumers see a clean ``Reduce`` libnode.

        :param sdfg: The SDFG to transform in place.
        :param _pipeline_results: Unused; kept for the Pass interface.
        :returns: ``{location_label: accumulator_name}`` for each pattern rewritten,
                  or ``None`` if nothing matched.
        """
        # Eliminate the frontend's ``__out = __inp`` copy tasklets first so the matcher
        # sees the bare RMW shape. ``TrivialTaskletElimination`` is value-preserving.
        from dace.transformation.dataflow.trivial_tasklet_elimination import TrivialTaskletElimination
        from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
        PatternMatchAndApplyRepeated([TrivialTaskletElimination()]).apply_pass(sdfg, {})

        rewritten: Dict[str, str] = {}

        # Pattern 1: sequential accumulator loops.
        loops: List[Tuple[LoopRegion, ControlFlowRegion]] = []
        for sd in sdfg.all_sdfgs_recursive():
            for region in sd.all_control_flow_regions():
                if isinstance(region, LoopRegion) and region.loop_variable:
                    loops.append((region, region.parent_graph))
        for loop, parent in loops:
            match = _match(loop)
            if match is None:
                continue
            _rewrite(parent, loop, match)
            rewritten[loop.label] = match.accum

        # Pattern 2: WCR write chains into scalar accumulators (one or more enclosing maps).
        map_matches: List[Tuple[SDFGState, _MapWCRMatch]] = []
        for sd in sdfg.all_sdfgs_recursive():
            for state in sd.all_states():
                for mmatch in _match_map_wcr(state, sd):
                    map_matches.append((state, mmatch))
        for state, mmatch in map_matches:
            _rewrite_map_wcr(state, mmatch)
            innermost_label = mmatch.map_entries[0].label if mmatch.map_entries else state.label
            rewritten[innermost_label] = mmatch.accum

        return rewritten or None

    def report(self, pass_retval: Any) -> Optional[str]:
        if not pass_retval:
            return None
        return f'AccumulatorToMapAndReduce: rewrote {len(pass_retval)} accumulator pattern(s): {pass_retval}'


def _root_sdfg(region: ControlFlowRegion) -> SDFG:
    cur = region
    while not isinstance(cur, SDFG):
        cur = cur.parent_graph
    return cur


def _one_elem(subset: subsets.Subset) -> Optional[int]:
    """Constant element count of ``subset``, or ``None`` if non-constant."""
    if subset is None:
        return None
    try:
        s = symbolic.simplify(subset.num_elements())
    except Exception:
        return None
    return int(s) if s.is_Integer else None


def _uses_loop_var(subset: subsets.Subset, loop_var) -> bool:
    """``True`` if any bound in ``subset`` mentions ``loop_var``."""
    if subset is None:
        return False
    for fs in subset.free_symbols:
        if symbolic.pystr_to_symbolic(str(fs)) == loop_var:
            return True
    return False


def _scalar_equiv(sdfg: SDFG, a: str, b: str) -> bool:
    """Same descriptor, or two dtype-compatible scalar-equivalents (Scalar or shape-1 Array)."""
    if a == b:
        return True
    da, db = sdfg.arrays.get(a), sdfg.arrays.get(b)
    if da is None or db is None or da.dtype != db.dtype:
        return False

    def scalar_like(d) -> bool:
        return isinstance(d, data.Scalar) or (isinstance(d, data.Array) and all(s == 1 for s in d.shape))

    return scalar_like(da) and scalar_like(db)


def _match(loop: LoopRegion) -> Optional[_Match]:
    """Decide whether ``loop`` is a constant-index scalar accumulator, and return
    the info needed to rewrite it. Conservative: anything off-pattern returns ``None``.

    The body may contain multiple tasklets (the per-iteration delta may be a
    multi-operator expression) -- we only require that exactly one tasklet is the
    *accumulator-update* (``acc_in OP delta_in``) whose output chains to the unique
    carried-accumulator write AccessNode. Upstream tasklets compute the delta and
    are left untouched by the rewrite.

    :param loop: The candidate :class:`~dace.sdfg.state.LoopRegion`.
    :returns: A populated :class:`_Match` on success, ``None`` otherwise.
    """
    if not loop.loop_variable:
        return None
    start = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if start is None or end is None or stride is None or stride != 1:
        return None

    sdfg = _root_sdfg(loop)
    loop_var_sym = symbolic.pystr_to_symbolic(loop.loop_variable)

    # Single-state body. Multiple tasklets are allowed: a multi-operator delta
    # spawns one tasklet per operator (e.g. ``acc + a[i] * 2`` -> ``Mult`` +
    # ``Add``). Only AccessNodes and Tasklets are allowed (refuse maps, nested
    # SDFGs, etc -- the rewrite assumes a flat dataflow body).
    blocks = loop.nodes()
    if len(blocks) != 1 or not isinstance(blocks[0], SDFGState):
        return None
    state = blocks[0]
    for n in state.nodes():
        if not isinstance(n, (nodes.Tasklet, nodes.AccessNode)):
            return None

    # Find the unique carried-accumulator AccessNode chain:
    #   - read AN: in_degree == 0, descriptor scalar-like and non-transient
    #     (the value flowing into the loop iteration)
    #   - write AN: in_degree > 0, same descriptor, same loop-invariant subset
    # Then walk back from the write AN through copy-out intermediates to find
    # the producing tasklet -- the *accumulator-update tasklet*.
    carried_candidates: List[Tuple[str, subsets.Subset, nodes.AccessNode, nodes.AccessNode]] = []
    write_ans_by_data: Dict[str, List[nodes.AccessNode]] = {}
    for n in state.data_nodes():
        if state.in_degree(n) > 0:
            write_ans_by_data.setdefault(n.data, []).append(n)
    for n in state.data_nodes():
        if state.in_degree(n) != 0:
            continue
        desc = sdfg.arrays.get(n.data)
        if desc is None or getattr(desc, 'transient', False):
            continue
        if not (isinstance(desc, data.Scalar) or (isinstance(desc, data.Array) and all(s == 1 for s in desc.shape))):
            continue
        if n.data not in write_ans_by_data or len(write_ans_by_data[n.data]) != 1:
            continue
        write_an = write_ans_by_data[n.data][0]
        read_outs = list(state.out_edges(n))
        write_ins = list(state.in_edges(write_an))
        if len(read_outs) != 1 or len(write_ins) != 1:
            continue
        if read_outs[0].data is None or write_ins[0].data is None:
            continue
        read_sub, write_sub = read_outs[0].data.subset, write_ins[0].data.subset
        if read_sub is None or write_sub is None:
            continue
        if _one_elem(read_sub) != 1 or _one_elem(write_sub) != 1:
            continue
        if _uses_loop_var(read_sub, loop_var_sym) or _uses_loop_var(write_sub, loop_var_sym):
            continue
        if read_sub != write_sub:
            continue
        carried_candidates.append((n.data, read_sub, n, write_an))
    if len(carried_candidates) != 1:
        return None
    carried_accum, carried_sub, accum_read_an, accum_write_an = carried_candidates[0]

    # Walk back from the carried-accum write AN through copy-out intermediates
    # (transient AN with in=1, out=1, plus identity ``__out = __inp`` tasklets)
    # to find the accumulator-update tasklet.
    update_tasklet = _walk_back_to_tasklet(state, accum_write_an)
    if update_tasklet is None or update_tasklet.code.language != dtypes.Language.Python:
        return None

    # Classify the update tasklet's body as ``out = lhs OP rhs``.
    try:
        tree = ast.parse((update_tasklet.code.as_string or '').strip())
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        return None
    rhs = tree.body[0].value
    if isinstance(rhs, ast.BinOp):
        wcr = _BINOP_TO_WCR.get(type(rhs.op))
    elif (isinstance(rhs, ast.Call) and isinstance(rhs.func, ast.Name) and len(rhs.args) == 2):
        wcr = _CALL_TO_WCR.get(rhs.func.id)
    else:
        wcr = None
    if wcr is None:
        return None

    def _has_data(e):
        return e.data is not None and not e.data.is_empty()

    in_edges = [e for e in state.in_edges(update_tasklet) if _has_data(e)]
    out_edges = [e for e in state.out_edges(update_tasklet) if _has_data(e)]
    if len(in_edges) != 2 or len(out_edges) != 1:
        return None

    # Identify which tasklet input is the accumulator read by tracing each input
    # edge back to its source AccessNode (possibly through a copy-wrapped scalar
    # intermediate). The accumulator input traces to the carried-accum READ AN.
    accum_idx: Optional[int] = None
    for idx, e in enumerate(in_edges):
        src_name = _trace_input_to_source(state, e)
        if src_name == carried_accum:
            if accum_idx is not None:
                return None  # ambiguous: both inputs trace to the accumulator
            accum_idx = idx
    if accum_idx is None:
        return None
    other_idx = 1 - accum_idx
    accum_edge = in_edges[accum_idx]
    other_edge = in_edges[other_idx]

    # Refuse a body that also performs a SECOND read-modify-write on a non-transient
    # array: an extra carried dependence we are not lifting.
    other_writes = set()
    other_reads = set()
    for st in loop.all_states():
        for node in st.data_nodes():
            if node.data == carried_accum:
                continue
            desc = sdfg.arrays.get(node.data)
            if desc is None or getattr(desc, 'transient', False):
                continue
            if st.in_degree(node) > 0:
                other_writes.add(node.data)
            if st.out_degree(node) > 0:
                other_reads.add(node.data)
    if other_writes & other_reads:
        return None

    return _Match(
        tasklet=update_tasklet,
        state=state,
        wcr=wcr,
        accum=carried_accum,
        accum_subset=carried_sub,
        accum_in_conn=accum_edge.dst_conn,
        other_in_conn=other_edge.dst_conn,
        out_conn=out_edges[0].src_conn,
        accum_read_an=accum_edge.src if isinstance(accum_edge.src, nodes.AccessNode) else accum_read_an,
        iter_start=start,
        iter_end=end,
    )


def _walk_back_to_tasklet(state: SDFGState, write_an: nodes.AccessNode) -> Optional[nodes.Tasklet]:
    """Walk back from ``write_an`` through copy-out intermediates (transient AN +
    identity tasklet chain) to the producing tasklet. Returns ``None`` if the
    chain is malformed or terminates at something other than a tasklet.
    """
    cur: Any = write_an
    while True:
        if isinstance(cur, nodes.Tasklet):
            return cur
        if not isinstance(cur, nodes.AccessNode):
            return None
        ins = list(state.in_edges(cur))
        if len(ins) != 1:
            return None
        prev = ins[0].src
        if isinstance(prev, nodes.Tasklet):
            # Identity copy-out tasklet ``__out = __inp`` -- step through to its single input.
            try:
                tree = ast.parse((prev.code.as_string or '').strip())
            except SyntaxError:
                return prev
            if (len(tree.body) == 1 and isinstance(tree.body[0], ast.Assign)
                    and isinstance(tree.body[0].value, ast.Name)):
                prev_ins = [ie for ie in state.in_edges(prev) if ie.data is not None and not ie.data.is_empty()]
                if len(prev_ins) == 1 and isinstance(prev_ins[0].src, nodes.AccessNode):
                    cur = prev_ins[0].src
                    continue
            return prev
        cur = prev


def _trace_input_to_source(state: SDFGState, edge) -> Optional[str]:
    """Walk a tasklet input edge back across an ``in=1, out=1`` copy-wrapped
    intermediate transient to identify the upstream source array name. Returns
    ``None`` if the chain does not fit the expected shape.
    """
    src = edge.src
    if not isinstance(src, nodes.AccessNode):
        return None
    desc = state.sdfg.arrays.get(src.data)
    if desc is None:
        return None
    if desc.transient and state.in_degree(src) == 1 and state.out_degree(src) == 1:
        pred = state.in_edges(src)[0]
        if isinstance(pred.src, nodes.AccessNode):
            return pred.src.data
    return src.data


def _rewrite(parent: ControlFlowRegion, loop: LoopRegion, match: _Match):
    """Rewrite ``loop`` into stage-1 (Map writing the per-iteration buffer) + stage-2
    (``Reduce`` libnode folding the buffer into the accumulator).

    Strategy: surgically mutate the original loop's tasklet into ``buf_out =
    other_in`` and retarget the write edge to ``buf[loop_var - iter_start]``;
    splice a new state after the loop that contains the Reduce libnode.

    :param parent: The control-flow region that owns ``loop``.
    :param loop: The matched accumulator loop.
    :param match: The :class:`_Match` produced by :func:`_match`.
    """
    sdfg = _root_sdfg(parent)

    # Allocate the per-iteration buffer. Shape ``(trip,)``, dtype matches the
    # accumulator's element type. Storage default -- the buffer is a transient
    # scratch and its lifetime is the surrounding scope.
    accum_desc = sdfg.arrays[match.accum]
    elem_dtype = accum_desc.dtype
    trip = symbolic.simplify(match.iter_end - match.iter_start + 1)
    buf_name, _ = sdfg.add_transient(_BUF_PREFIX + match.accum, [trip], elem_dtype, find_new_name=True)

    # Stage 1: rewrite the tasklet and its write edge.
    _mutate_to_stage1(loop, match, buf_name)

    # Stage 2: append a state after the loop containing the Reduce libnode.
    _splice_reduce_state(parent, loop, match, buf_name)

    # Refresh CFG bookkeeping so downstream analyses see the new state.
    sdfg.reset_cfg_list()


def _mutate_to_stage1(loop: LoopRegion, match: _Match, buf_name: str):
    """Turn the matched loop body into the per-iteration buffer write.

    Drops the accumulator-read input chain, removes the entire output-side chain
    (any copy-out intermediates plus the carried-accumulator write inside the body),
    retargets the tasklet output to ``buf[i - start]``, and rewrites the tasklet
    body to ``out = other_in``.
    """
    state = match.state
    tasklet = match.tasklet

    _disconnect_accum_read(state, tasklet, match.accum_in_conn, match.accum_read_an)
    tasklet.code.as_string = f'{match.out_conn} = {match.other_in_conn}'
    _strip_output_chain(state, tasklet, match.out_conn)

    buf_an = state.add_write(buf_name)
    idx = symbolic.simplify(symbolic.pystr_to_symbolic(loop.loop_variable) - match.iter_start)
    write_memlet = mm.Memlet(data=buf_name, subset=subsets.Range([(idx, idx, 1)]))
    state.add_edge(tasklet, match.out_conn, buf_an, None, write_memlet)


def _strip_output_chain(state: SDFGState, tasklet: nodes.Tasklet, out_conn: str):
    """Walk forward from ``tasklet[out_conn]`` deleting the entire write chain.

    The frontend stages an RMW as ``tasklet -> tmp -> [assign-tasklet ->] acc``
    -- one or more intermediate transient AccessNodes optionally interleaved with
    identity ``__out = __inp`` tasklets. After we redirect the tasklet's output to
    ``buf``, every node on the old chain is unreachable; remove them so the body
    is clean for downstream LoopToMap.

    Walks via the edge being removed: at each step we strip ``cur``'s outgoing
    edge to ``nxt``, then keep walking from ``nxt`` if it is an intermediate
    transient (AccessNode with no remaining inputs and exactly one output) or an
    identity tasklet (one data out-edge). Anything else terminates the walk.
    """
    cur = tasklet
    cur_conn: Optional[str] = out_conn
    to_remove: List = []
    while True:
        if cur_conn is None:
            out_edges = list(state.out_edges(cur))
        else:
            out_edges = [e for e in state.out_edges(cur) if e.src_conn == cur_conn]
        if len(out_edges) != 1:
            break
        e = out_edges[0]
        nxt = e.dst
        state.remove_edge(e)
        if isinstance(nxt, nodes.AccessNode):
            if state.in_degree(nxt) > 0:
                break
            outs = list(state.out_edges(nxt))
            to_remove.append(nxt)
            if len(outs) != 1:
                break
            cur = nxt
            cur_conn = None
            continue
        if isinstance(nxt, nodes.Tasklet):
            data_outs = [oe for oe in state.out_edges(nxt) if oe.data is not None and not oe.data.is_empty()]
            if len(data_outs) != 1:
                break
            for ie in list(state.in_edges(nxt)):
                state.remove_edge(ie)
            to_remove.append(nxt)
            cur = nxt
            cur_conn = data_outs[0].src_conn
            continue
        break
    for n in to_remove:
        if state.in_degree(n) == 0 and state.out_degree(n) == 0:
            state.remove_node(n)


def _disconnect_accum_read(state: SDFGState, tasklet: nodes.Tasklet, conn: str, anchor: nodes.AccessNode):
    """Drop the tasklet's accumulator-read input, peel back any orphan copy-wrapped
    intermediate, and remove the upstream accumulator-read AccessNode if it ends up
    isolated (the body no longer needs the carried value).
    """
    if conn not in tasklet.in_connectors:
        return
    for e in list(state.in_edges(tasklet)):
        if e.dst_conn != conn:
            continue
        state.remove_edge(e)
    tasklet.remove_in_connector(conn)
    # Peel the copy-wrapped intermediate AccessNode (transient, now in=1 / out=0)
    # plus its upstream edge from the accumulator-read AN.
    upstream_an: Optional[nodes.AccessNode] = None
    if isinstance(anchor, nodes.AccessNode) and state.out_degree(anchor) == 0:
        desc = state.sdfg.arrays.get(anchor.data)
        if desc is not None and getattr(desc, 'transient', False):
            for oe in list(state.in_edges(anchor)):
                if isinstance(oe.src, nodes.AccessNode):
                    upstream_an = oe.src
                state.remove_edge(oe)
            state.remove_node(anchor)
        elif state.in_degree(anchor) == 0:
            upstream_an = anchor  # the anchor IS the read AN; treat below
            state.remove_node(anchor)
    # If the upstream accumulator-read AN is now isolated (its only reader was the
    # intermediate we just removed), drop it -- the body has no remaining use.
    if upstream_an is not None and upstream_an in state.nodes() and state.degree(upstream_an) == 0:
        state.remove_node(upstream_an)


def _splice_reduce_state(parent: ControlFlowRegion, loop: LoopRegion, match: _Match, buf_name: str):
    """Append a state after ``loop`` containing a ``Reduce`` libnode that folds
    ``buf`` into ``acc[match.accum_subset]`` with the original associative ``OP``.

    ``identity=None`` so the pre-loop value of ``acc`` seeds the fold.
    """
    import dace
    out_edges = list(parent.out_edges(loop))
    red_state = parent.add_state(loop.label + '_reduce')
    parent.add_edge(loop, red_state, dace.InterstateEdge())
    for e in out_edges:
        parent.remove_edge(e)
        parent.add_edge(red_state, e.dst, e.data)

    sdfg = _root_sdfg(parent)
    buf_desc = sdfg.arrays[buf_name]
    buf_an = red_state.add_read(buf_name)
    acc_an = red_state.add_write(match.accum)
    red = red_state.add_reduce(match.wcr, axes=list(range(len(buf_desc.shape))), identity=None)
    red_state.add_edge(buf_an, None, red, None, mm.Memlet(data=buf_name, subset=subsets.Range.from_array(buf_desc)))
    red_state.add_edge(red, None, acc_an, None, mm.Memlet(data=match.accum, subset=match.accum_subset))


# ----------------------------------------------------------------------------
# Pattern 2: Map with a scalar WCR write.
# ----------------------------------------------------------------------------


class _MapWCRMatch(NamedTuple):
    """A successfully matched WCR write chain into a scalar accumulator.

    Two input shapes are accepted:

    - ``tasklet --[wcr]--> MapExit (+ outer MapExits) --> AccessNode``
    - ``tasklet --> intermediate_AN --[wcr]--> MapExit (+ outer MapExits) --> AccessNode``

    The chain may pass through one or more nested ``MapExit`` nodes (one per
    enclosing map); the buffer's shape is the product of all enclosing map
    iteration ranges, indexed by all their parameters.

    :param map_entries: The enclosing ``MapEntry`` nodes, innermost-first (paired with
        ``map_exits``). The innermost map's body holds the tasklet (and the optional
        intermediate AccessNode).
    :param map_exits: The matching ``MapExit`` nodes, innermost-first.
    :param wcr_edge: The single edge carrying ``wcr=`` (either tasklet→MapExit or
        intermediate-AN→MapExit; never internal to the body otherwise).
    :param chain_edges: All edges along the chain from the WCR edge to ``accum_an``,
        in walk order (the WCR edge is ``chain_edges[0]``). Each gets re-pointed at
        the buffer during rewrite.
    :param accum_an: The destination AccessNode (after walking out all MapExits).
    :param accum: The accumulator descriptor name (``accum_an.data``).
    :param accum_subset: The accumulator's loop-invariant single-element subset.
    :param wcr: Normalised reduction lambda string (one of :data:`_BINOP_TO_WCR` /
        :data:`_CALL_TO_WCR` values).
    """
    map_entries: List[nodes.MapEntry]
    map_exits: List[nodes.MapExit]
    wcr_edge: Any
    chain_edges: List[Any]
    accum_an: nodes.AccessNode
    accum: str
    accum_subset: subsets.Range
    wcr: str


#: Map of WCR lambda body to a (recognised, normalised) reduction lambda. Only
#: associative ops are taken; anything else returns ``None`` and the matcher refuses.
def _normalise_wcr(wcr_str: str) -> Optional[str]:
    """Parse a WCR lambda and return one of :data:`_BINOP_TO_WCR` / :data:`_CALL_TO_WCR`'s
    canonical strings if its body is one of the supported associative ops. Else ``None``."""
    if not wcr_str:
        return None
    try:
        tree = ast.parse(wcr_str, mode='eval').body
    except (SyntaxError, ValueError):
        return None
    if not isinstance(tree, ast.Lambda):
        return None
    body = tree.body
    if isinstance(body, ast.BinOp):
        return _BINOP_TO_WCR.get(type(body.op))
    if isinstance(body, ast.Call) and isinstance(body.func, ast.Name) and len(body.args) == 2:
        return _CALL_TO_WCR.get(body.func.id)
    return None


def _match_map_wcr(state: SDFGState, sdfg: SDFG) -> List[_MapWCRMatch]:
    """Find every WCR write chain into a scalar-like accumulator in ``state``.

    Supports two input shapes (and any composition through one or more
    enclosing MapExits):

    - ``tasklet --[wcr]--> MapExit --> ... --> AccessNode``
    - ``tasklet --> intermediate_AN --[wcr]--> MapExit --> ... --> AccessNode``

    Each match represents one chain: the unique WCR edge plus the MapExits
    walked forward to a single-element-subset write to a non-transient
    scalar-like AccessNode. The reduction op must be associative.

    Returns a list because a single state can contain several independent
    WCR-write chains; the caller rewrites each in order.
    """
    matches: List[_MapWCRMatch] = []
    seen_wcr_ids: Set[int] = set()
    for edge in state.edges():
        if edge.data is None or edge.data.wcr is None:
            continue
        if id(edge) in seen_wcr_ids:
            continue
        seen_wcr_ids.add(id(edge))
        wcr_norm = _normalise_wcr(edge.data.wcr)
        if wcr_norm is None:
            continue
        chain = _trace_wcr_chain(state, edge, sdfg)
        if chain is None:
            continue
        accum_an, accum_subset, map_exits, chain_edges = chain
        # Identify the matching MapEntry nodes (innermost-first) so the rewrite knows
        # which iteration parameters drive the per-iteration buffer index.
        map_entries = [state.entry_node(mx) for mx in map_exits]
        if any(me is None for me in map_entries):
            continue
        matches.append(
            _MapWCRMatch(
                map_entries=map_entries,
                map_exits=map_exits,
                wcr_edge=edge,
                chain_edges=chain_edges,
                accum_an=accum_an,
                accum=accum_an.data,
                accum_subset=accum_subset,
                wcr=wcr_norm,
            ))
    return matches


def _trace_wcr_chain(state: SDFGState, wcr_edge,
                     sdfg: SDFG) -> Optional[Tuple[nodes.AccessNode, subsets.Range, List[nodes.MapExit], List[Any]]]:
    """Walk forward from ``wcr_edge`` through MapExits to a final AccessNode.

    The walker allows the WCR edge to land on a MapExit directly, or on an
    intermediate transient AccessNode that itself feeds a MapExit (one extra
    hop, the ``tasklet → AN → wcr → MapExit`` shape). After the chain exits
    its enclosing maps, the final destination must be a non-transient
    scalar-like AccessNode at a single-element subset that doesn't reference
    any of the enclosing map parameters.

    :returns: ``(accum_an, accum_subset, map_exits_innermost_first, chain_edges)``
              on success; ``None`` if the chain is malformed or violates any of
              the constraints above.
    """
    chain_edges: List[Any] = [wcr_edge]
    map_exits: List[nodes.MapExit] = []

    cur_node = wcr_edge.dst
    cur_conn = wcr_edge.dst_conn

    # Allow one ``intermediate transient AN`` hop *if* the WCR edge points at a
    # transient AccessNode whose single non-WCR outgoing edge already targets a
    # MapExit. That's the ``tasklet → AN → wcr-on-AN-out → MapExit`` shape.
    # If instead the WCR lands directly on a MapExit, no hop is needed.
    if isinstance(cur_node, nodes.AccessNode):
        desc = sdfg.arrays.get(cur_node.data)
        if desc is None or not getattr(desc, 'transient', False):
            return None
        if state.in_degree(cur_node) != 1 or state.out_degree(cur_node) != 1:
            return None
        nxt_edge = state.out_edges(cur_node)[0]
        if not isinstance(nxt_edge.dst, nodes.MapExit):
            return None
        chain_edges.append(nxt_edge)
        cur_node = nxt_edge.dst
        cur_conn = nxt_edge.dst_conn

    # Now walk MapExit → MapExit → ... → AccessNode. At each step the MapExit
    # has matching ``IN_x``/``OUT_x`` connectors; follow the unique out-edge.
    while isinstance(cur_node, nodes.MapExit):
        map_exits.append(cur_node)
        if not cur_conn or not cur_conn.startswith('IN_'):
            return None
        out_conn = 'OUT_' + cur_conn[3:]
        outer = [e for e in state.out_edges(cur_node) if e.src_conn == out_conn]
        if len(outer) != 1:
            return None
        nxt_edge = outer[0]
        chain_edges.append(nxt_edge)
        cur_node = nxt_edge.dst
        cur_conn = nxt_edge.dst_conn

    if not isinstance(cur_node, nodes.AccessNode):
        return None
    accum = cur_node.data
    desc = sdfg.arrays.get(accum)
    # Never re-process our own buffers, but transients ARE valid accumulators: a
    # reduction into a fresh transient row-slot (e.g. matvec ``tmp[i]`` inside an
    # i-loop, ``tmp = define_local(...)``) is exactly the dot/reduce we want to lift
    # to a Reduce node. The single-element write check below still gates the shape.
    if desc is None or accum.startswith(_BUF_PREFIX):
        return None
    if not isinstance(desc, (data.Scalar, data.Array)):
        return None

    # The accumulator WRITE must be a single element (``tmp[i]`` / ``acc[c]`` / a
    # scalar) -- that is what makes the chain a reduction folding into one slot. The
    # slot may be indexed by an enclosing LOOP symbol (matvec ``tmp[i]``); it must
    # NOT be indexed by an enclosing MAP parameter (that would be a per-lane store,
    # not a reduction). The array itself may be larger than one element.
    final_subset = chain_edges[-1].data.subset
    if final_subset is None or _one_elem(final_subset) != 1:
        return None
    map_params = []
    for mx in map_exits:
        me = state.entry_node(mx)
        if me is None:
            return None
        map_params.extend(me.map.params)
    map_param_syms = {symbolic.pystr_to_symbolic(p) for p in map_params}
    final_free = {symbolic.pystr_to_symbolic(str(s)) for s in final_subset.free_symbols}
    if final_free & map_param_syms:
        return None

    return cur_node, final_subset, map_exits, chain_edges


def _accum_is_fresh(sdfg: SDFG, accum: str) -> bool:
    """True iff ``accum`` is written ONLY by WCR edges anywhere in the SDFG (no plain
    seed write) -- i.e. a fresh reduction accumulator that must fold from the identity
    rather than from its (uninitialized) pre-existing value."""
    saw_wcr = False
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nodes.AccessNode) and node.data == accum:
                for e in state.in_edges(node):
                    if e.data is None or e.data.data is None:
                        continue
                    if e.data.wcr is None:
                        return False
                    saw_wcr = True
    return saw_wcr


def _rewrite_map_wcr(state: SDFGState, match: _MapWCRMatch):
    """Replace a WCR write chain into a scalar with a per-iteration buffer write +
    a post-state ``Reduce`` libnode folding the buffer back into the accumulator.

    Buffer shape = product of all enclosing map iteration extents (innermost-first
    order matches ``match.map_entries``). Each map-exit's outgoing chain edge has
    its subset re-projected onto the corresponding buffer axis(es); the WCR flag
    is cleared everywhere along the chain. The original accumulator AccessNode is
    replaced by a fresh buffer write; if it ends up isolated, removed.
    """
    import dace
    sdfg = state.sdfg

    # Capture accumulator freshness NOW, before the chain re-pointing below removes
    # the accumulator's WCR write (which would make ``_accum_is_fresh`` see no WCR and
    # wrongly report not-fresh). A fresh accumulator gets the reduction identity.
    accum_fresh = _accum_is_fresh(sdfg, match.accum)

    # Compute per-axis (start, end, trip) for each enclosing map, innermost-first.
    axis_specs: List[Tuple[Any, Any, Any]] = []
    for me in match.map_entries:
        rng = me.map.range.ranges
        # Refuse multi-dim Maps for v1 -- one axis per enclosing Map keeps the buffer
        # 1:1 with surrounding iteration domains; a multi-dim Map can be handled later.
        if len(rng) != 1 or rng[0][2] != 1:
            return
        rb, re_, _ = rng[0]
        axis_specs.append((rb, re_, symbolic.simplify(re_ - rb + 1)))

    buf_shape = [trip for _, _, trip in axis_specs]
    buf_name, _ = sdfg.add_transient(_BUF_PREFIX + match.accum,
                                     buf_shape,
                                     sdfg.arrays[match.accum].dtype,
                                     find_new_name=True)
    buf_desc = sdfg.arrays[buf_name]

    # The per-iteration write index uses each enclosing map's parameter offset to
    # its own start: ``buf[i - i_start, j - j_start, ...]``. For the outer/MapExit
    # edges, the corresponding axis collapses to the full extent of that axis.
    point_subset = subsets.Range([((symbolic.pystr_to_symbolic(me.map.params[0]) - axis_specs[idx][0], ) * 2 + (1, ))
                                  for idx, me in enumerate(match.map_entries)])

    # Rewrite the chain edges (innermost-first walk order: wcr_edge, then each
    # MapExit-out edge). After ``axis_idx`` MapExits the corresponding axis collapses
    # from a point to its full ``0:trip`` range.
    chain = match.chain_edges
    # 0..len(map_exits) chain edges; ``chain[0]`` is the WCR edge, then one out-edge
    # per MapExit walked. Subsets at index ``k`` cover the first ``k`` innermost axes
    # as full ranges and the remaining axes as point.
    for k, edge in enumerate(chain):
        new_sub = []
        for axis_idx in range(len(axis_specs)):
            _start, _end, trip = axis_specs[axis_idx]
            if axis_idx < k:
                new_sub.append((0, trip - 1, 1))
            else:
                p_start = (symbolic.pystr_to_symbolic(match.map_entries[axis_idx].map.params[0]) -
                           axis_specs[axis_idx][0])
                new_sub.append((p_start, p_start, 1))
        new_memlet = mm.Memlet(data=buf_name, subset=subsets.Range(new_sub))
        new_memlet.wcr = None
        state.remove_edge(edge)
        state.add_edge(edge.src, edge.src_conn, edge.dst, edge.dst_conn, new_memlet)

    # The chain's final edge now targets the original accum AccessNode but writes
    # ``buf`` data; swap that to a fresh ``buf`` AccessNode. The original accum AN,
    # if isolated, is removed.
    final_edge = chain[-1]
    # Re-find the just-added final edge (the one we appended in the previous loop).
    refreshed_final = next(e for e in state.in_edges(match.accum_an)
                           if e.src is final_edge.src and e.src_conn == final_edge.src_conn and e.data.data == buf_name)
    buf_write = state.add_write(buf_name)
    state.remove_edge(refreshed_final)
    state.add_edge(refreshed_final.src, refreshed_final.src_conn, buf_write, None, refreshed_final.data)

    # Choose the Reduce identity. A *fresh* accumulator -- one written ONLY by WCR
    # (no plain seed write anywhere) -- must fold from the reduction identity, NOT
    # from the slot's pre-existing value: a fresh transient is uninitialized, so
    # seeding from it (``identity=None``) would read garbage (the original WCR relied
    # on the slot already holding the identity). A non-fresh accumulator (e.g.
    # ``C = beta*C`` then ``+=``) keeps ``identity=None`` so its seed is honoured.
    identity = None
    if accum_fresh:
        from dace.frontend import operations as _ops
        redtype = _ops.detect_reduction_type(match.wcr)
        if redtype != dtypes.ReductionType.Custom:
            try:
                identity = dtypes.reduction_identity(sdfg.arrays[match.accum].dtype, redtype)
            except Exception:
                identity = None

    # Place the Reduce libnode that folds buf into the accumulator.
    #
    # If the accumulator AccessNode still has consumers IN THIS STATE (e.g. atax's
    # ``compute_y`` reads ``tmp[i]`` produced by the just-lifted ``compute_tmp``),
    # the Reduce MUST stay in-state so dataflow ordering holds (``buf -> Reduce ->
    # tmp -> consumer``). Appending it as a post-state would let the consumer read the
    # accumulator before the Reduce produced it. When the accumulator is consumed only
    # in a LATER state (the scalar case), a post-state Reduce keeps the buffer-fill map
    # and the reduce in separate states (the original, tested behaviour).
    has_in_state_consumer = state.degree(match.accum_an) > 0
    buf_full = mm.Memlet(data=buf_name, subset=subsets.Range.from_array(buf_desc))
    acc_memlet = mm.Memlet(data=match.accum, subset=match.accum_subset)
    if has_in_state_consumer:
        red = state.add_reduce(match.wcr, axes=list(range(len(buf_desc.shape))), identity=identity)
        state.add_edge(buf_write, None, red, None, buf_full)
        state.add_edge(red, None, match.accum_an, None, acc_memlet)
    else:
        if state.degree(match.accum_an) == 0:
            state.remove_node(match.accum_an)
        parent = state.parent_graph
        out_edges = list(parent.out_edges(state))
        red_state = parent.add_state(state.label + '_reduce')
        parent.add_edge(state, red_state, dace.InterstateEdge())
        for e in out_edges:
            parent.remove_edge(e)
            parent.add_edge(red_state, e.dst, e.data)
        buf_read = red_state.add_read(buf_name)
        acc_write = red_state.add_write(match.accum)
        red = red_state.add_reduce(match.wcr, axes=list(range(len(buf_desc.shape))), identity=identity)
        red_state.add_edge(buf_read, None, red, None, buf_full)
        red_state.add_edge(red, None, acc_write, None, acc_memlet)

    sdfg.reset_cfg_list()
