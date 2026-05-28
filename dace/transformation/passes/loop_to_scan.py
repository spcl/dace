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
        equal to ``k_w - 1`` in v1.
    :param other_indices: List of ``(axis, sympy_expr)`` for non-scan axes of
        ``out`` (must be loop-invariant). The same indices are used to slice
        the seed and the seed-add output.
    :param delta_name: The per-iteration delta source array's name.
    :param d: The scan-axis offset on the delta read (``delta[i + d, ...]``).
    :param delta_other_indices: ``(axis, sympy_expr)`` for non-scan axes of
        ``delta`` (must be loop-invariant).
    :param iter_start: The loop's start expression (symbolic or constant).
    :param iter_end: The loop's inclusive end expression.
    """
    op: ScanOp
    out_name: str
    scan_axis: int
    k_w: Any
    k_r: Any
    other_indices: List[Any]
    delta_name: str
    d: Any
    delta_other_indices: List[Any]
    iter_start: Any
    iter_end: Any


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

    blocks = loop.nodes()
    if len(blocks) != 1 or not isinstance(blocks[0], SDFGState):
        return None
    state = blocks[0]

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

    # Walk back from the write AccessNode through any chain of single-edge intermediate
    # transients (the frontend's slice-copy form ``tmp -> ... -> out``) to the producing
    # tasklet that actually does the OP. ``TrivialTaskletElimination`` ran in
    # ``apply_pass`` already, so the chain only contains transient AccessNodes.
    tasklet = _trace_back_to_tasklet(state, write_edge.src)
    if tasklet is None or tasklet.code.language != dtypes.Language.Python:
        return None

    # Classify the tasklet body as a single ``out = a OP b`` assignment.
    try:
        tree = ast.parse((tasklet.code.as_string or '').strip())
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        return None
    rhs = tree.body[0].value
    if isinstance(rhs, ast.BinOp):
        op = _BINOP_TO_SCAN_OP.get(type(rhs.op))
    elif (isinstance(rhs, ast.Call) and isinstance(rhs.func, ast.Name) and len(rhs.args) == 2):
        op = _CALL_TO_SCAN_OP.get(rhs.func.id)
    else:
        op = None
    if op is None:
        return None

    # Two data inputs.
    def _has_data(e):
        return e.data is not None and not e.data.is_empty()

    in_edges = [e for e in state.in_edges(tasklet) if _has_data(e)]
    out_edges_t = [e for e in state.out_edges(tasklet) if _has_data(e)]
    if len(in_edges) != 2 or len(out_edges_t) != 1:
        return None

    # Resolve each tasklet input through any one-hop slice transient to the source
    # AccessNode + the source-side subset (memlet.data is the source array; the
    # subset on the in-edge is the source slice copied into the tasklet's connector).
    carry_edge = None
    carry_subset = None
    delta_src = None
    delta_subset = None
    for e in in_edges:
        src_name, src_subset = _resolve_input(state, e)
        if src_name is None:
            return None
        if src_name == out_name:
            if carry_edge is not None:
                return None  # ambiguous carry
            carry_edge = e
            carry_subset = src_subset
        else:
            if delta_src is not None:
                return None
            delta_src = src_name
            delta_subset = src_subset

    if carry_edge is None or delta_src is None:
        return None

    # Carry-side subset must agree with the write on every non-scan axis and offset
    # by exactly +1 on the scan axis.
    r_axis, k_r, r_others = _classify_subset(carry_subset, loop.loop_variable)
    if r_axis is None or r_axis != write_axis:
        return None
    if not _same_other_indices(r_others, write_others):
        return None
    if symbolic.simplify(k_w - k_r) != 1:
        return None

    # Delta-side subset: single scan-axis dependence, non-scan dims loop-invariant.
    delta_axis, d, delta_others = _classify_subset(delta_subset, loop.loop_variable)
    if delta_axis is None:
        return None
    if delta_src == out_name:
        # ``out[i+1] = out[i] + out[i]`` -- delta aliases the carry; refused.
        return None

    # Refuse any second non-transient write anywhere in the loop body.
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
        delta_name=delta_src,
        d=d,
        delta_other_indices=delta_others,
        iter_start=start,
        iter_end=end,
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
    """Replace ``loop`` with three sibling states: delta-build, scan, seed-add."""
    import dace
    out_desc = sdfg.arrays[info.out_name]
    trip = symbolic.simplify(info.iter_end - info.iter_start + 1)
    delta_buf, _ = sdfg.add_array(f'{_DELTA_BUF_PREFIX}{info.out_name}', [trip], out_desc.dtype, transient=True,
                                  find_new_name=True)
    scan_buf, _ = sdfg.add_array(f'{_SCAN_BUF_PREFIX}{info.out_name}', [trip], out_desc.dtype, transient=True,
                                 find_new_name=True)

    was_start = (parent.start_block is loop)
    in_edges = list(parent.in_edges(loop))
    out_edges = list(parent.out_edges(loop))

    s_build = parent.add_state(loop.label + '_scan_build', is_start_block=was_start)
    s_scan = parent.add_state(loop.label + '_scan')
    s_apply = parent.add_state(loop.label + '_scan_apply')
    parent.add_edge(s_build, s_scan, dace.InterstateEdge())
    parent.add_edge(s_scan, s_apply, dace.InterstateEdge())
    for e in in_edges:
        parent.add_edge(e.src, s_build, e.data)
    for e in out_edges:
        parent.add_edge(s_apply, e.dst, e.data)
    parent.remove_node(loop)

    _emit_delta_build(s_build, sdfg, info, delta_buf, trip)
    _emit_scan(s_scan, sdfg, info, delta_buf, scan_buf, trip)
    _emit_seed_add(s_apply, sdfg, info, scan_buf, trip)


def _emit_delta_build(state: SDFGState, sdfg: SDFG, info: _Scan, delta_buf: str, trip: Any):
    """Map over ``_i in [0, trip-1]`` writing ``delta[iter_start + d + _i, ...]`` to ``delta_buf[_i]``."""
    delta_desc = sdfg.arrays[info.delta_name]
    delta_axis_expr = symbolic.simplify(info.iter_start + info.d) + symbolic.pystr_to_symbolic('_i')
    delta_subset = _build_subset(delta_desc, info.scan_axis, delta_axis_expr, info.delta_other_indices)
    state.add_mapped_tasklet(
        f'{state.label}_tasklet',
        {'_i': f'0:{trip}'},
        {'__d': mm.Memlet(data=info.delta_name, subset=delta_subset)},
        '__o = __d',
        {'__o': mm.Memlet(data=delta_buf, subset=subsets.Range([('_i', '_i', 1)]))},
        external_edges=True,
    )


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
