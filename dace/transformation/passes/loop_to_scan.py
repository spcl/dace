# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lift a sequential prefix-scan loop to a ``Scan`` library node.

A loop body shaped like ::

    for i in range(start, end + 1):
        out[i + 1, jl, ...] = out[i, jl, ...] OP delta[i + d, jl, ...]

is the textbook inclusive prefix scan along the ``i`` axis: ``out[i+1]`` is the
running reduction of ``delta[start+d .. i+d]`` combined with the seed ``out[start]``.
This pass detects that shape and replaces the loop with three sibling states:

1. **delta-build** -- a ``Map`` over the iteration range that copies ``delta[i+d, ...]``
   into a fresh 1-D transient ``_scan_in_<out>`` (size = trip count).
2. **scan** -- a :class:`~dace.libraries.standard.nodes.scan.Scan` libnode that
   computes ``_scan_out_<out>`` from ``_scan_in_<out>`` (CPU expansion = OpenMP 5.0
   parallel scan; CUDA expansion = ``cub::DeviceScan``).
3. **seed-add** -- a ``Map`` that writes ``out[i+1, jl, ...] = seed + _scan_out_<out>[i]``
   where ``seed = out[start, jl, ...]`` (the pre-loop value at the read end of the chain).

The body's per-iteration delta is captured *as the second tasklet input* in v1 -- a
clean array slice ``delta[i + d, ...]``. Multi-tasklet body shapes whose delta is a
computed expression (e.g. ``out[i+1] = out[i] + a[i] * b[i] + c[i]``) are out of scope
for v1 and stay as a follow-up; the matcher refuses those.

Compatibility with :class:`~dace.transformation.passes.loop_to_reduce.LoopToReduce`:
``LoopToReduce``'s tasklet matcher refuses any loop whose write subset depends on the
loop variable (its check at the ``_uses(write_subset, loop_var_sym)`` line). The scan
shape *requires* that dependence (``out[i+1]``), so the two pass matchers do not
overlap -- ``LoopToReduce`` declines, ``LoopToScan`` claims. Run order: ``LoopToReduce``
first, ``LoopToScan`` second, then ``LoopToMap``.

Constraint inherited from :class:`~dace.transformation.passes.promote_constant_index_access.\
PromoteConstantIndexAccess`: the rewrite is sound only when no extra per-iteration
state needs ``lastprivate`` semantics to be observable post-loop. The single-tasklet
v1 shape satisfies this trivially -- the only carry is the scan recurrence itself,
captured by the Scan libnode -- but the matcher checks the body explicitly and refuses
on any other carried writes to non-transient arrays.
"""
import ast
import copy
from typing import Any, List, NamedTuple, Optional

from dace import SDFG, data, dtypes, properties, subsets, symbolic
from dace import memlet as mm
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis

# Re-export the supported associative ops via :class:`ScanOp`; the matcher recognises
# the same four ops the libnode expansions cover.
from dace.libraries.standard.nodes.scan import (Scan, ScanOp, INPUT_CONNECTOR_NAME as _SCAN_IN,
                                                OUTPUT_CONNECTOR_NAME as _SCAN_OUT)


#: Map AST BinOp class -> ScanOp.
_BINOP_TO_SCAN_OP = {
    ast.Add: ScanOp.SUM,
    ast.Mult: ScanOp.PRODUCT,
}

#: Map ``Call(Name(...))`` callee -> ScanOp (for ``max`` / ``min``).
_CALL_TO_SCAN_OP = {
    'max': ScanOp.MAX,
    'min': ScanOp.MIN,
}

#: Prefix for the per-iteration transient buffers the rewrite allocates.
_DELTA_BUF_PREFIX = '_scan_in_'
_SCAN_BUF_PREFIX = '_scan_out_'


class _Scan(NamedTuple):
    """A successfully matched scan loop.

    :param op: The associative reduction op (one of :class:`ScanOp`).
    :param out_name: The scan-output (and carried-input) array's name.
    :param scan_axis: Index of the dimension carrying the scan recurrence.
    :param k_w: Write-side scan-axis offset (``out[i + k_w, ...]``).
    :param k_r: Read-side scan-axis offset (``out[i + k_r, ...]``). Always
        equal to ``k_w - 1``.
    :param other_indices: List of ``(axis, sympy_expr)`` for non-scan axes of
        ``out`` (must be loop-invariant). The same indices are used to slice
        the seed and the seed-add output.
    :param iter_start: The loop's start expression (symbolic or constant).
    :param iter_end: The loop's inclusive end expression.
    :param body_state: The unique state inside the loop body (single-state
        constraint from the matcher).
    :param scan_update_tasklet: The tasklet that performs the scan recurrence
        (``out[i+1] = out[i] OP delta`` -- one carry input, one delta input).
    :param carry_in_conn: The scan-update tasklet's carry-input connector
        (the one reading ``out[i]`` through the slice chain).
    :param delta_in_conn: The scan-update tasklet's delta-input connector.
    :param out_conn: The scan-update tasklet's output connector.
    :param carry_anchor: The AccessNode the carry-in edge enters in the body
        (the slice-copy intermediate, or the direct source AN). Used during
        orphan cleanup.
    """
    op: ScanOp
    out_name: str
    scan_axis: int
    k_w: Any
    k_r: Any
    other_indices: List[Any]
    iter_start: Any
    iter_end: Any
    body_state: SDFGState
    scan_update_tasklet: nodes.Tasklet
    carry_in_conn: str
    delta_in_conn: str
    out_conn: str
    carry_anchor: nodes.AccessNode


@properties.make_properties
@xf.explicit_cf_compatible
class LoopToScan(ppl.Pass):
    """Lift prefix-scan loops to a :class:`Scan` libnode.

    Pattern: a ``LoopRegion`` with a unit-stride loop variable whose single-state body
    holds a single tasklet ``out[i+1, ...] = out[i, ...] OP delta[i+d, ...]`` for one
    of the associative ops ``+``, ``*``, ``max``, ``min``. The write and the carried
    read must address the same array on the same scan axis, the read offset must be
    exactly one less than the write offset, and the non-scan-axis indices must match
    exactly between read and write and must not depend on the loop variable.

    The body must not write any other non-transient array (any such write would
    require ``lastprivate``-style preservation that the rewrite doesn't support).
    """

    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Descriptors | ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results) -> Optional[int]:
        # Strip the frontend's identity ``__out = __inp`` copy tasklets so the matcher
        # sees the bare ``out[i+1] = out[i] + delta[i]`` shape. Without this, the carry
        # is hidden behind an ``assign_NN`` copy node on the write side. Same idea as
        # ``AccumulatorToMapAndReduce``'s TTE preprocess.
        from dace.transformation.dataflow.trivial_tasklet_elimination import TrivialTaskletElimination
        from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
        PatternMatchAndApplyRepeated([TrivialTaskletElimination()]).apply_pass(sdfg, {})

        count = 0
        for loop, parent in _collect_loops(sdfg):
            info = _match(loop, sdfg)
            if info is None:
                continue
            _rewrite(parent, loop, info, sdfg)
            count += 1
        return count or None


def _collect_loops(sdfg: SDFG):
    out: List = []
    for sd in sdfg.all_sdfgs_recursive():
        for region in sd.all_control_flow_regions():
            if isinstance(region, LoopRegion) and region.loop_variable:
                out.append((region, region.parent_graph))
    return out


def _match(loop: LoopRegion, sdfg: SDFG) -> Optional[_Scan]:
    start = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if start is None or end is None or stride is None or stride != 1:
        return None

    # Accept either a single-state body (v1) or a multi-state body whose only
    # *content* state is a single SDFGState (the cloudsc shape: a trivial pre-
    # state with an iedge assignment + the actual body state + a trivial post-
    # state with an iedge advancing the loop iterator). Empty wrapper states
    # are fine because they contribute no dataflow; their iedge assignments stay
    # in place (they bind loop-iteration-symbol shorthands like
    # ``kfdia_plus_1 = kfdia + 1``, not the carry).
    blocks = loop.nodes()
    if not all(isinstance(b, SDFGState) for b in blocks):
        return None
    content_states = [b for b in blocks if len(b.nodes()) > 0]
    if len(content_states) != 1:
        return None
    state = content_states[0]

    # Body must contain only tasklets and AccessNodes (no Map scopes / nested SDFGs).
    for n in state.nodes():
        if not isinstance(n, (nodes.Tasklet, nodes.AccessNode)):
            return None

    # Identify the carried-array write target. The carried array is the unique
    # non-transient Array with both a read and a write incident in this state.
    out_name = _find_carried_array(state, sdfg, loop.loop_variable)
    if out_name is None:
        return None
    out_desc = sdfg.arrays[out_name]

    # The unique write edge into ``out`` (incident on the unique write AccessNode).
    write_edge = _find_unique_write_edge(state, out_name)
    if write_edge is None:
        return None
    write_axis, k_w, write_others = _classify_subset(write_edge.data.subset, loop.loop_variable)
    if write_axis is None:
        return None

    # Find the scan-update tasklet: among all tasklets in the body, the one whose
    # single output (eventually) reaches the ``out`` write AND one of whose inputs
    # resolves to ``out`` at offset ``k_w - 1`` on the scan axis. For v1 that's the
    # only tasklet in the body; for v2 there may be downstream tasklets (extending
    # the delta computation) and the matcher has to find the right one.
    candidate = _find_scan_update_tasklet(state, sdfg, out_name, loop.loop_variable, write_axis, write_others, k_w)
    if candidate is None:
        return None
    tasklet, carry_edge, delta_edge, op, carry_anchor = candidate

    out_edges_t = [e for e in state.out_edges(tasklet)
                   if e.data is not None and not e.data.is_empty()]
    if len(out_edges_t) != 1:
        return None
    k_r = symbolic.simplify(k_w - 1)

    # Refuse any second non-transient write anywhere in the loop body. The body may
    # contain arbitrarily many upstream *transient* nodes feeding the delta (v2), but
    # the only externally observable write must be the carry write to ``out``.
    for st in loop.all_states():
        for node in st.data_nodes():
            if st.in_degree(node) == 0:
                continue
            desc = sdfg.arrays.get(node.data)
            if desc is None or getattr(desc, 'transient', False):
                continue
            if node.data != out_name:
                return None

    return _Scan(
        op=op,
        out_name=out_name,
        scan_axis=write_axis,
        k_w=k_w,
        k_r=k_r,
        other_indices=write_others,
        iter_start=start,
        iter_end=end,
        body_state=state,
        scan_update_tasklet=tasklet,
        carry_in_conn=carry_edge.dst_conn,
        delta_in_conn=delta_edge.dst_conn,
        out_conn=out_edges_t[0].src_conn,
        carry_anchor=carry_anchor,
    )


def _find_carried_array(state: SDFGState, sdfg: SDFG, loop_var: str) -> Optional[str]:
    """Locate the unique non-transient array that has both a write *and* a read incident
    in ``state`` with subsets that depend on ``loop_var``. Returns its name, or ``None``."""
    reads: set = set()
    writes: set = set()
    for n in state.data_nodes():
        desc = sdfg.arrays.get(n.data)
        if desc is None or getattr(desc, 'transient', False):
            continue
        for e in state.in_edges(n):
            if e.data is not None and e.data.data == n.data and e.data.subset is not None:
                if _subset_uses(e.data.subset, loop_var):
                    writes.add(n.data)
        for e in state.out_edges(n):
            if e.data is not None and e.data.data == n.data and e.data.subset is not None:
                if _subset_uses(e.data.subset, loop_var):
                    reads.add(n.data)
    intersect = reads & writes
    if len(intersect) != 1:
        return None
    return next(iter(intersect))


def _find_unique_write_edge(state: SDFGState, name: str):
    """Locate the unique in-edge to a non-transient AccessNode of ``name``. Returns the
    edge, or ``None`` on ambiguity.
    """
    found = None
    for n in state.data_nodes():
        if n.data != name:
            continue
        ins = list(state.in_edges(n))
        if not ins:
            continue
        if len(ins) > 1 or found is not None:
            return None
        found = ins[0]
    return found


def _find_scan_update_tasklet(state: SDFGState, sdfg: SDFG, out_name: str, loop_var: str,
                              scan_axis: int, write_others, k_w):
    """Search the body for the scan-update tasklet -- the unique tasklet whose body is
    ``__out = a OP b`` for an associative OP and whose two inputs are (a) the carry
    (resolves to ``out`` at offset ``k_w - 1`` on the scan axis, matching non-scan
    indices) and (b) the delta (anything else). Returns
    ``(tasklet, carry_edge, delta_edge, op, carry_anchor)`` or ``None``.

    The body may contain additional downstream tasklets (v2: extending the per-iteration
    delta computation after the scan-update). The scan-update tasklet is the *first one*
    along the carry-write path whose carry-side input resolves to ``out`` directly;
    anything past that is the delta-extension chain and stays in place after the rewrite.
    """
    for node in state.nodes():
        if not isinstance(node, nodes.Tasklet) or node.code.language != dtypes.Language.Python:
            continue
        try:
            tree = ast.parse((node.code.as_string or '').strip())
        except SyntaxError:
            continue
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
            continue
        rhs = tree.body[0].value
        if isinstance(rhs, ast.BinOp):
            op = _BINOP_TO_SCAN_OP.get(type(rhs.op))
        elif (isinstance(rhs, ast.Call) and isinstance(rhs.func, ast.Name) and len(rhs.args) == 2):
            op = _CALL_TO_SCAN_OP.get(rhs.func.id)
        else:
            op = None
        if op is None:
            continue
        in_edges = [e for e in state.in_edges(node)
                    if e.data is not None and not e.data.is_empty()]
        out_edges = [e for e in state.out_edges(node)
                     if e.data is not None and not e.data.is_empty()]
        if len(in_edges) != 2 or len(out_edges) != 1:
            continue
        carry_edge = None
        delta_edge = None
        carry_anchor = None
        ambiguous = False
        for e in in_edges:
            src_name, src_subset = _resolve_input(state, e)
            if src_name == out_name and src_subset is not None:
                r_axis, k_r_cand, r_others = _classify_subset(src_subset, loop_var)
                if (r_axis == scan_axis and _same_other_indices(r_others, write_others)
                        and symbolic.simplify(k_w - k_r_cand) == 1):
                    if carry_edge is not None:
                        ambiguous = True
                        break
                    carry_edge = e
                    carry_anchor = e.src
                    continue
            if delta_edge is not None:
                # Two non-carry inputs -- this isn't the scan-update tasklet.
                delta_edge = None
                break
            delta_edge = e
        if ambiguous or carry_edge is None or delta_edge is None:
            continue
        return node, carry_edge, delta_edge, op, carry_anchor
    return None


def _trace_back_to_tasklet(state: SDFGState, node) -> Optional[nodes.Tasklet]:
    """Walk back through a chain of in=1 transient AccessNodes (slice/copy holders)
    to the upstream tasklet. Returns the tasklet, or ``None`` if the chain doesn't
    terminate at one.
    """
    cur = node
    while isinstance(cur, nodes.AccessNode):
        desc = state.sdfg.arrays.get(cur.data)
        if desc is None or not getattr(desc, 'transient', False):
            return None
        ins = list(state.in_edges(cur))
        if len(ins) != 1:
            return None
        cur = ins[0].src
    if isinstance(cur, nodes.Tasklet):
        return cur
    return None


def _resolve_input(state: SDFGState, edge):
    """Walk back from a tasklet input edge through a one-hop intermediate AccessNode
    (the frontend's slice-copy ``arr -> arr_index -> tasklet``) to the *source*
    AccessNode of ``arr``. Returns ``(arr_name, arr-side subset)`` or ``(None, None)``.
    """
    src = edge.src
    if not isinstance(src, nodes.AccessNode):
        return None, None
    desc = state.sdfg.arrays.get(src.data)
    if desc is None:
        return None, None
    if not getattr(desc, 'transient', False):
        # Direct ``arr -> tasklet`` -- subset on the in-edge is ``arr``'s subset.
        return src.data, edge.data.subset
    # Transient intermediate: walk one hop back. The upstream edge's memlet carries
    # the real source's subset (``arr`` side).
    if state.in_degree(src) != 1 or state.out_degree(src) != 1:
        return None, None
    pred = state.in_edges(src)[0]
    if not isinstance(pred.src, nodes.AccessNode):
        return None, None
    if pred.data is None or pred.data.subset is None:
        return None, None
    return pred.src.data, pred.data.subset


def _subset_uses(subset: subsets.Subset, loop_var: str) -> bool:
    """``True`` if any bound of ``subset`` mentions ``loop_var``."""
    if subset is None:
        return False
    loop_var_sym = symbolic.pystr_to_symbolic(loop_var)
    for fs in subset.free_symbols:
        if symbolic.pystr_to_symbolic(str(fs)) == loop_var_sym:
            return True
    return False


def _classify_subset(subset: subsets.Subset, loop_var: str):
    """Return ``(scan_axis, offset, non_scan_indices)`` for ``subset``, or
    ``(None, None, None)`` if the subset doesn't fit the v1 shape.

    The "v1 shape": every dimension is a single point (``lo == hi``, stride 1),
    *exactly one* axis depends on ``loop_var`` (linearly with constant offset),
    all other axes are loop-invariant.
    """
    if not isinstance(subset, subsets.Range):
        return None, None, None
    loop_var_sym = symbolic.pystr_to_symbolic(loop_var)
    scan_axis = None
    offset = None
    others: List[Any] = []
    for axis_idx, (lo, hi, st) in enumerate(subset.ranges):
        if lo != hi or st != 1:
            return None, None, None
        lo_sym = symbolic.pystr_to_symbolic(str(lo))
        if loop_var_sym in lo_sym.free_symbols:
            if scan_axis is not None:
                return None, None, None
            try:
                off = symbolic.simplify(lo_sym - loop_var_sym)
            except Exception:
                return None, None, None
            if loop_var_sym in off.free_symbols:
                return None, None, None
            scan_axis = axis_idx
            offset = off
        else:
            others.append((axis_idx, lo_sym))
    return scan_axis, offset, others


def _same_other_indices(a, b) -> bool:
    """Compare two ``[(axis, expr), ...]`` lists for exact symbolic equality."""
    if len(a) != len(b):
        return False
    for (ax_a, ex_a), (ax_b, ex_b) in zip(a, b):
        if ax_a != ax_b or symbolic.simplify(ex_a - ex_b) != 0:
            return False
    return True


def _rewrite(parent: ControlFlowRegion, loop: LoopRegion, info: _Scan, sdfg: SDFG):
    """Rewrite ``loop`` into a 3-part chain via in-place body mutation.

    1. **Modified loop**: the scan-update tasklet's carry input is severed and its
       output is re-routed to write the per-iteration *delta* into a transient
       ``_scan_in[loop_var - start]``. The tasklet's body collapses to
       ``__o = __delta``, so the rest of the body's delta-computation subgraph
       (whatever produces the delta input -- a direct array slice in v1, an
       arbitrary upstream subgraph in v2) is kept intact and continues to run
       per iteration. After this mutation the body's only write is uniquely
       indexed by ``loop_var``, so a subsequent ``LoopToMap`` lifts it cleanly.
    2. **Scan state**: a single ``Scan`` libnode reads ``_scan_in`` and writes
       ``_scan_out`` (same shape ``[trip]``, same dtype, op matches the matched
       tasklet's combiner).
    3. **Seed-add Map**: writes ``out[start + k_w + _i, ...] = seed OP _scan_out[_i]``
       in parallel, where ``seed = out[start + k_r, ...]`` is the pre-loop value
       read by the original loop's first carry-read.
    """
    import dace
    out_desc = sdfg.arrays[info.out_name]
    trip = symbolic.simplify(info.iter_end - info.iter_start + 1)
    delta_buf, _ = sdfg.add_array(f'{_DELTA_BUF_PREFIX}{info.out_name}', [trip], out_desc.dtype, transient=True,
                                  find_new_name=True)
    scan_buf, _ = sdfg.add_array(f'{_SCAN_BUF_PREFIX}{info.out_name}', [trip], out_desc.dtype, transient=True,
                                 find_new_name=True)

    # Mutate the loop body: scan-update tasklet becomes a passthrough writing to
    # ``_scan_in[loop_var - start]``; the carry-input chain is stripped.
    _mutate_body_to_delta_buffer(info, delta_buf)

    # Splice the post-loop chain into the parent CFG: loop -> scan_state -> apply_state -> [original successors].
    out_edges = list(parent.out_edges(loop))
    s_scan = parent.add_state(loop.label + '_scan')
    s_apply = parent.add_state(loop.label + '_scan_apply')
    parent.add_edge(loop, s_scan, dace.InterstateEdge())
    parent.add_edge(s_scan, s_apply, dace.InterstateEdge())
    for e in out_edges:
        parent.remove_edge(e)
        parent.add_edge(s_apply, e.dst, e.data)

    _emit_scan(s_scan, sdfg, info, delta_buf, scan_buf, trip)
    _emit_seed_add(s_apply, sdfg, info, scan_buf, trip)
    sdfg.reset_cfg_list()


def _mutate_body_to_delta_buffer(info: _Scan, delta_buf: str):
    """In-place: sever the scan-update tasklet's carry input, collapse the tasklet's
    body to a passthrough of its delta input, and re-route the body's *final* write
    to ``out`` so it lands in ``delta_buf[loop_var - iter_start]`` instead. Anything
    between the scan-update tasklet and that final write -- additional tasklets that
    extend the per-iteration delta computation (the v2 case) -- is kept verbatim.
    """
    state = info.body_state
    tasklet = info.scan_update_tasklet

    # 1. Sever the carry input chain (orphan transients pruned).
    _disconnect_carry_chain(state, tasklet, info.carry_in_conn, info.carry_anchor)

    # 2. The scan-update tasklet becomes a passthrough of its delta input. The downstream
    #    chain (whatever tasklets continue the delta computation in v2) now propagates the
    #    delta value verbatim instead of the carry+delta sum.
    tasklet.code.as_string = f'{info.out_conn} = {info.delta_in_conn}'

    # 3. Locate the body's final write edge to ``out`` (the unique in-edge to its write
    #    AccessNode in the state) and re-route it to ``delta_buf[loop_var - iter_start]``.
    write_an = _find_carried_write_an(state, info.out_name)
    if write_an is None:
        return  # Nothing to re-route (defensive; matcher already established the write).
    final_write_edges = list(state.in_edges(write_an))
    if len(final_write_edges) != 1:
        return
    final_edge = final_write_edges[0]
    state.remove_edge(final_edge)
    loop_var = info.body_state.parent_graph.loop_variable
    idx_expr = symbolic.simplify(symbolic.pystr_to_symbolic(loop_var) - info.iter_start)
    buf_an = state.add_write(delta_buf)
    state.add_edge(final_edge.src, final_edge.src_conn, buf_an, None,
                   mm.Memlet(data=delta_buf, subset=subsets.Range([(idx_expr, idx_expr, 1)])))
    if state.degree(write_an) == 0:
        state.remove_node(write_an)


def _find_carried_write_an(state: SDFGState, name: str) -> Optional[nodes.AccessNode]:
    """Return the unique AccessNode of ``name`` with at least one in-edge (the body's
    write target). ``None`` on ambiguity or absence.
    """
    found = None
    for n in state.data_nodes():
        if n.data != name or state.in_degree(n) == 0:
            continue
        if found is not None:
            return None
        found = n
    return found


def _disconnect_carry_chain(state: SDFGState, tasklet: nodes.Tasklet, conn: str,
                            anchor: nodes.AccessNode):
    """Remove the tasklet's carry-input edge and the slice-copy intermediate chain
    that fed it. Transient intermediates that become isolated are dropped. The
    original carry-source ``out`` AccessNode in the body state is ALSO dropped if
    it ends up isolated -- the post-loop seed-add reads from ``out`` in its own
    state, so the body-state read AN is no longer needed.
    """
    if conn in tasklet.in_connectors:
        for e in list(state.in_edges(tasklet)):
            if e.dst_conn == conn:
                state.remove_edge(e)
        tasklet.remove_in_connector(conn)
    # Walk backward from ``anchor`` (the AN the carry edge entered), pruning every
    # ancestor that becomes isolated (in+out degree == 0). The walk stops once it
    # finds a node with remaining incident edges (still in use elsewhere).
    cur = anchor
    while isinstance(cur, nodes.AccessNode) and cur in state.nodes():
        if state.in_degree(cur) + state.out_degree(cur) != 0:
            # Still in use (e.g. some other in-loop reader of ``out``); stop here.
            break
        upstream = None
        ins = list(state.in_edges(cur))
        if len(ins) == 1 and isinstance(ins[0].src, nodes.AccessNode):
            upstream = ins[0].src
        for ie in ins:
            state.remove_edge(ie)
        state.remove_node(cur)
        if upstream is None:
            break
        cur = upstream


def _collect_output_chain(state: SDFGState, tasklet: nodes.Tasklet, out_conn: str
                          ) -> List[nodes.AccessNode]:
    """Walk forward from ``tasklet[out_conn]`` collecting any slice-copy intermediate
    transient AccessNodes (in=1, out=1) until the final write AN of the carried array.
    The final AN is INCLUDED in the returned list (it gets pruned along with the
    intermediates -- the rewrite re-routes the tasklet's output to ``delta_buf``).
    """
    chain: List[nodes.AccessNode] = []
    cur = tasklet
    cur_conn = out_conn
    while True:
        out_edges = [e for e in state.out_edges(cur) if e.src_conn == cur_conn]
        if len(out_edges) != 1:
            break
        e = out_edges[0]
        nxt = e.dst
        if not isinstance(nxt, nodes.AccessNode):
            break
        chain.append(nxt)
        desc = state.sdfg.arrays.get(nxt.data)
        if desc is not None and not getattr(desc, 'transient', False):
            break  # non-transient = the original ``out`` write target; stop after appending
        # Continue through transient intermediates.
        cur = nxt
        cur_conn = None
    return chain


def _emit_scan(state: SDFGState, sdfg: SDFG, info: _Scan, delta_buf: str, scan_buf: str, trip: Any):
    """Scan(delta_buf) -> scan_buf via the libnode."""
    r = state.add_read(delta_buf)
    w = state.add_write(scan_buf)
    node = Scan(name=f'{state.label}_op', op=info.op, exclusive=False)
    state.add_node(node)
    state.add_edge(r, None, node, _SCAN_IN, mm.Memlet(data=delta_buf, subset=subsets.Range([(0, trip - 1, 1)])))
    state.add_edge(node, _SCAN_OUT, w, None, mm.Memlet(data=scan_buf, subset=subsets.Range([(0, trip - 1, 1)])))


def _emit_seed_add(state: SDFGState, sdfg: SDFG, info: _Scan, scan_buf: str, trip: Any):
    """Map over ``_i`` writing ``out[start + k_w + _i, ...] = seed + scan_buf[_i]`` where
    ``seed = out[start + k_r, ...]`` (broadcast over every iteration of the map).
    """
    out_desc = sdfg.arrays[info.out_name]
    seed_axis_expr = symbolic.simplify(info.iter_start + info.k_r)
    seed_subset = _build_subset(out_desc, info.scan_axis, seed_axis_expr, info.other_indices)
    write_axis_expr = symbolic.simplify(info.iter_start + info.k_w) + symbolic.pystr_to_symbolic('_i')
    write_subset = _build_subset(out_desc, info.scan_axis, write_axis_expr, info.other_indices)

    op_expr = {
        ScanOp.SUM: '__seed + __v',
        ScanOp.PRODUCT: '__seed * __v',
        ScanOp.MIN: 'min(__seed, __v)',
        ScanOp.MAX: 'max(__seed, __v)',
    }[info.op]
    state.add_mapped_tasklet(
        f'{state.label}_tasklet',
        {'_i': f'0:{trip}'},
        {
            '__seed': mm.Memlet(data=info.out_name, subset=seed_subset),
            '__v': mm.Memlet(data=scan_buf, subset=subsets.Range([('_i', '_i', 1)])),
        },
        f'__o = {op_expr}',
        {'__o': mm.Memlet(data=info.out_name, subset=write_subset)},
        external_edges=True,
    )


def _build_subset(desc: data.Array, scan_axis: int, scan_expr, other_indices: List[Any]) -> subsets.Range:
    """Synthesize an N-D single-point subset on ``desc`` with ``scan_expr`` on ``scan_axis``
    and the loop-invariant exprs from ``other_indices`` on the rest. Used both for the
    delta read in the build map and the seed/output writes in the apply map.
    """
    other_map = {axis: expr for axis, expr in other_indices}
    rng = []
    for axis_idx in range(len(desc.shape)):
        if axis_idx == scan_axis:
            rng.append((scan_expr, scan_expr, 1))
        else:
            ex = other_map[axis_idx]
            rng.append((ex, ex, 1))
    return subsets.Range(rng)
