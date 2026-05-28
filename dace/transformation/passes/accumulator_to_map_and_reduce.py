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
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

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
        """Rewrite every matching accumulator loop in ``sdfg`` (and nested SDFGs).

        :param sdfg: The SDFG to transform in place.
        :param _pipeline_results: Unused; kept for the Pass interface.
        :returns: ``{loop_label: accumulator_name}`` for each loop rewritten, or
                  ``None`` if no loop matched.
        """
        # Eliminate the frontend's ``__out = __inp`` copy tasklets first so the matcher
        # sees the bare RMW shape. ``TrivialTaskletElimination`` is value-preserving.
        from dace.transformation.dataflow.trivial_tasklet_elimination import TrivialTaskletElimination
        from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
        PatternMatchAndApplyRepeated([TrivialTaskletElimination()]).apply_pass(sdfg, {})

        rewritten: Dict[str, str] = {}
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
        return rewritten or None

    def report(self, pass_retval: Any) -> Optional[str]:
        if not pass_retval:
            return None
        return f'AccumulatorToMapAndReduce: rewrote {len(pass_retval)} accumulator loop(s): {pass_retval}'


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
    import dace  # avoid an import cycle at module scope
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
    red_state.add_edge(buf_an, None, red, None,
                       mm.Memlet(data=buf_name, subset=subsets.Range.from_array(buf_desc)))
    red_state.add_edge(red, None, acc_an, None, mm.Memlet(data=match.accum, subset=match.accum_subset))
