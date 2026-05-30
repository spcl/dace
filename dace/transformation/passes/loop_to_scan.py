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
                                                OUTPUT_CONNECTOR_NAME as _SCAN_OUT,
                                                INIT_CONNECTOR_NAME as _SCAN_INIT)


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
_SEED_SCALAR_PREFIX = '_scan_seed_'


class _Scan(NamedTuple):
    """A successfully matched scan loop.

    :param op: The associative reduction op (one of :class:`ScanOp`).
    :param out_name: The scan-output (and carried-input) array's name.
    :param scan_axis: Index of the dimension carrying the scan recurrence.
    :param k_w: Write-side scan-axis offset (``out[i + k_w, ...]``).
    :param k_r: Read-side scan-axis offset (``out[i + k_r, ...]``). Differs
        from ``k_w`` by a positive ``scan_stride`` (``= 1`` for the contiguous
        scan; ``> 1`` for stride-S residue-class scans like TSVC s1221).
    :param scan_stride: ``k_w - k_r`` -- the recurrence's stride. Passed to
        the ``Scan`` libnode's ``stride`` property; values ``> 1`` run the
        ``S`` independent residue-class scans in parallel.
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
    :param literal_delta: For the v3 literal-augmented-carry shape (TSVC
        s242: ``a[i] = a[i-1] + 0.5 + ...``), the numeric literal on the
        non-carry side of the scan-update tasklet's binop. ``None`` for the
        v1/v2 shape where the delta arrives via a data edge. When set, the
        scan-update tasklet's rewritten code emits the literal directly;
        the carry connector is severed and the tasklet has zero in-edges.
    """
    op: ScanOp
    out_name: str
    scan_axis: int
    k_w: Any
    k_r: Any
    scan_stride: Any
    other_indices: List[Any]
    iter_start: Any
    iter_end: Any
    body_state: SDFGState
    scan_update_tasklet: nodes.Tasklet
    carry_in_conn: str
    delta_in_conn: Optional[str]
    out_conn: str
    carry_anchor: nodes.AccessNode
    literal_delta: Optional[str] = None
    # Nested-body extension: when set, the body is wrapped in this inner LoopRegion
    # (data-parallel column loop in the cloudsc ``for_1133`` shape). The rewrite
    # emits a vector Scan -- a Map over the inner loop variable wrapping the
    # 1-D Scan along the outer (scan) axis. ``None`` for the flat v1-v5 case.
    inner_loop: Optional[LoopRegion] = None


class _ScalarCarryScan(NamedTuple):
    """A successfully matched scalar-carry prefix-scan loop (TSVC s3112 family).

    Shape (after ``TrivialTaskletElimination``)::

        acc(read, scalar) ──[acc[0]]──> Tasklet(__out = __acc OP __delta)
                                              ↑
                                       delta source AN reading delta[i + delta_offset, ...]
        Tasklet ──[interm[0]]──> interm(scalar transient) ──[acc[0]]──> acc(write)
        acc(write) ──[out[i + out_offset, ...]]──> out  (per-iter prefix output)

    The carry is the SCALAR ``acc``; the per-iter output ``out`` is read
    from the post-RMW value of ``acc`` (inclusive scan semantics). Each
    iteration computes ``acc_i = acc_{i-1} OP delta[i + delta_offset]`` and
    emits ``out[i + out_offset, ...] = acc_i``.

    Distinct from :class:`_Scan` (array-carry, where the same array name is
    both the carrier and the per-iter output written at offset
    ``out[i + k_w, ...]`` while reading ``out[i + k_r, ...]``). See the
    array-carry tuple's docstring for that companion shape.

    :param op: The associative reduction op (one of :class:`ScanOp`).
    :param acc_name: The scalar accumulator's data name.
    :param out_name: The per-iter output array's data name.
    :param out_scan_axis: Which dimension of ``out`` carries the loop iter.
    :param out_offset: Constant offset on ``out_scan_axis`` (the per-iter
        write subset is ``out[loop_var + out_offset, ...]``).
    :param out_other_indices: ``(axis, sympy_expr)`` for non-iter axes of
        ``out`` (must be loop-invariant), same convention as
        :class:`_Scan.other_indices`.
    :param delta_name: The per-iter delta-gather array's data name.
    :param delta_scan_axis: Which dimension of ``delta`` carries the loop iter.
    :param delta_offset: Constant offset on ``delta_scan_axis``.
    :param delta_other_indices: Non-iter axes of ``delta`` (loop-invariant).
    :param iter_start: Loop start expr.
    :param iter_end: Loop inclusive end expr.
    :param body_state: The single content state inside the loop body.
    :param scan_update_tasklet: Tasklet whose body is ``__out = __acc OP __delta``.
    :param acc_in_conn: The tasklet's accumulator-input connector.
    :param delta_in_conn: The tasklet's delta-input connector.
    :param out_conn: The tasklet's output connector.
    :param acc_read_an: The accumulator's READ AccessNode in the body.
    :param acc_write_an: The accumulator's WRITE AccessNode in the body
        (whose outgoing edge feeds the per-iter ``out`` write).
    :param acc_used_post_loop: ``True`` if any state OUTSIDE the loop body
        reads ``acc`` -- in which case the rewrite emits a post-loop
        writeback ``acc[0] = scan_out[trip - 1]``.
    """
    op: ScanOp
    acc_name: str
    out_name: str
    out_scan_axis: int
    out_offset: Any
    out_other_indices: List[Any]
    delta_name: str
    delta_scan_axis: int
    delta_offset: Any
    delta_other_indices: List[Any]
    iter_start: Any
    iter_end: Any
    body_state: SDFGState
    scan_update_tasklet: nodes.Tasklet
    acc_in_conn: str
    delta_in_conn: str
    out_conn: str
    acc_read_an: nodes.AccessNode
    acc_write_an: nodes.AccessNode
    acc_used_post_loop: bool


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
        # Whole-SDFG preprocess: strip frontend ``__out = __inp`` copy tasklets so the
        # matcher sees the bare ``out[i+1] = out[i] + delta[i]`` shape. Without this the
        # carry hides behind an ``assign_NN`` copy node on the write side.
        from dace.transformation.dataflow.trivial_tasklet_elimination import TrivialTaskletElimination
        from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
        PatternMatchAndApplyRepeated([TrivialTaskletElimination()]).apply_pass(sdfg, {})

        # Normalise backward-iterating loops (``range(N, 0, -1)`` shape; cloudsc
        # ``for_1079`` is the canonical case) to forward iteration. ``LoopToScan``'s
        # matcher only handles ``stride == 1``; rather than build sign-flip handling
        # into every gate, delegate to the dedicated canonicalisation pass once up
        # front so subsequent analysis sees only positive-stride loops.
        # ``NormalizeNegativeStride`` rebinds the old iterator on the body entry
        # iedge (``jm = N - _loop_pos_X``) rather than rewriting body memlets;
        # follow up with ``SymbolPropagation`` so the body's subsets are expressed
        # in the new positive-stride iterator and the matcher recognises them.
        from dace.transformation.passes.canonicalize.normalize_negative_stride import NormalizeNegativeStride
        from dace.transformation.passes.symbol_propagation import SymbolPropagation
        if NormalizeNegativeStride().apply_pass(sdfg, {}):
            SymbolPropagation().apply_pass(sdfg, {})

        # Per-loop preprocess: fold adjacent content SDFGStates inside the body when the
        # iedge between them is trivial (v5 -- the cloudsc ``pfsqrf`` shape). Whole-SDFG
        # ``StateFusion`` doesn't reach into LoopRegion bodies via ``MatchPatterns``, so
        # do a targeted body-local merge.
        for loop, _ in _collect_loops(sdfg):
            _fuse_body_states(loop)

        count = 0
        for loop, parent in _collect_loops(sdfg):
            infos = _match_all(loop, sdfg)
            if infos:
                for info in infos:
                    _rewrite(parent, loop, info, sdfg)
                    count += 1
                continue
            # No array-carry match; try scalar-carry (TSVC s3112: scalar accumulator
            # whose post-RMW value is written to a per-iter array slot). Mutually
            # exclusive with array-carry by construction -- array-carry needs the
            # carrier read+written at offset (i + k_r) / (i + k_w), scalar-carry
            # needs a SCALAR carrier read+written at constant subset [0].
            sc = _match_scalar_carry(loop, sdfg)
            if sc is not None:
                _rewrite_scalar_carry(parent, loop, sc, sdfg)
                count += 1
        if count > 0:
            # Narrow the freshly-emitted state-level memlets on the new
            # ``Scan`` + seed-add states. The rewrite uses array-extent memlets
            # for the buffer wires; running whole-SDFG propagation collapses
            # them to the actual ``[scan_axis, *non_scan]`` ranges so
            # downstream consumers (RedundantArrayCopying, codegen) see the
            # tight subset rather than the conservative one.
            from dace.sdfg.propagation import propagate_memlets_sdfg
            propagate_memlets_sdfg(sdfg)
        return count or None


def _collect_loops(sdfg: SDFG):
    out: List = []
    for sd in sdfg.all_sdfgs_recursive():
        for region in sd.all_control_flow_regions():
            if isinstance(region, LoopRegion) and region.loop_variable:
                out.append((region, region.parent_graph))
    return out


def _descend_to_content_state(loop: LoopRegion):
    """Find one candidate flat content state inside ``loop``'s body. Convenience
    wrapper around :func:`_descend_to_content_state_candidates` -- returns the
    first candidate, kept for callers that don't need to enumerate.
    """
    cands = _descend_to_content_state_candidates(loop)
    return cands[0] if cands else (None, None)


def _descend_to_content_state_candidates(loop: LoopRegion):
    """Enumerate all flat content states inside ``loop``'s body that could host
    the scan-update tasklet.

    Returns a list of ``(state, inner_loop)`` tuples (``inner_loop`` is ``None``
    when the candidate is a state at the outer body level). The body may have
    MULTIPLE sibling inner LoopRegions and sibling content states (cloudsc
    ``for_1133`` shape: slice-prep inner loop next to scan-update inner loop
    plus a content state). Each candidate is returned; ``_match_all`` then tries
    each until one yields a successful match. Downstream gates
    (``_find_carried_arrays`` / per-loop "only-carriers-written" sweep) reject
    candidates that aren't the scan-update region.

    Backward-compatibility: returns the same shape as the original
    single-candidate descent for the v5 flat case and the v6 single-nested case.
    """
    blocks = loop.nodes()
    inner_loop_regions = [b for b in blocks if isinstance(b, LoopRegion)]
    states = [b for b in blocks if isinstance(b, SDFGState)]
    others = [b for b in blocks if not isinstance(b, (LoopRegion, SDFGState))]
    if others:
        return []
    content_states = [s for s in states if len(s.nodes()) > 0]
    candidates = []
    # Flat candidates at the outer body level: every content state is itself
    # a possible scan-update host (v5 case).
    if not inner_loop_regions:
        if len(content_states) == 1:
            candidates.append((content_states[0], None))
    else:
        # Each sibling inner LoopRegion is a candidate; recurse into each.
        for inner in inner_loop_regions:
            if not inner.loop_variable:
                continue
            inner_state, deeper = _descend_to_content_state(inner)
            if inner_state is None or deeper is not None:
                continue
            candidates.append((inner_state, inner))
    return candidates


def _fuse_body_states(loop: LoopRegion) -> int:
    """Body-local state fusion: merge adjacent SDFGStates inside ``loop`` when the
    iedge between them is trivial (no assignments, ``condition`` unconditional).

    Whole-SDFG ``StateFusion`` doesn't reach into LoopRegion bodies via
    ``match_patterns`` (it's a top-level ``MultiStateTransformation``); this
    helper does the same merge directly on the body. Required for the v5 case --
    the cloudsc ``pfsqrf`` inner loop's body has two content states joined by a
    no-op iedge that the single-content-state matcher otherwise refuses.

    Per-iteration semantics: the iedge orders state 1's writes before state 2's
    reads. After fusion the two states become one, and the merging step glues
    each pair of access nodes with the same ``data`` (state 1's write node ->
    state 2's read node, etc.) so the dataflow read-after-write order is
    preserved by an explicit edge.

    Conservatively refuses when:

    * any iedge in the body carries an assignment or a condition,
    * any of the body's blocks is not an SDFGState (refuse on nested LoopRegion /
      ConditionalBlock).

    :param loop: The LoopRegion to mutate in place.
    :returns: The number of fusions performed.
    """
    n_fused = 0
    while True:
        blocks = loop.nodes()
        if not all(isinstance(b, SDFGState) for b in blocks):
            return n_fused
        # Find a fusion candidate: a pair (s1, s2) with one s1->s2 edge, both states
        # non-empty (we already pass-through empty wrappers in the matcher), and a
        # trivial iedge between them.
        cand = None
        for s1 in blocks:
            outs = loop.out_edges(s1)
            if len(outs) != 1:
                continue
            e = outs[0]
            if e.data is None or e.data.assignments or not e.data.is_unconditional():
                continue
            s2 = e.dst
            if not isinstance(s2, SDFGState):
                continue
            if len(loop.in_edges(s2)) != 1:
                continue
            if s1.is_empty() or s2.is_empty():
                continue
            cand = (s1, s2, e)
            break
        if cand is None:
            return n_fused
        s1, s2, iedge = cand
        _merge_state_into(s1, s2)
        # Reroute s2's out-edges to s1, then drop s2.
        for oe in list(loop.out_edges(s2)):
            loop.add_edge(s1, oe.dst, oe.data)
            loop.remove_edge(oe)
        loop.remove_edge(iedge)
        loop.remove_node(s2)
        n_fused += 1


def _merge_state_into(s1: SDFGState, s2: SDFGState):
    """Move all of ``s2``'s nodes/edges into ``s1``, gluing same-``data`` AccessNodes
    so dataflow read-after-write order is preserved.

    For each AccessNode in ``s2`` that names data written in ``s1``, the s2 node is
    aliased to the s1 write node (so s2's read sees s1's write). All other s2 nodes
    move to s1 unchanged. Edges are then re-added on s1 using the alias-mapped
    endpoints, and ``s2`` is left empty (the caller removes it from the parent CFG).
    """
    # data -> s1's write-side AccessNode (the one with incoming edges).
    s1_writes = {n.data: n for n in s1.data_nodes() if s1.in_degree(n) > 0}
    # Snapshot before mutating: s2's nodes and edges, plus the per-node classification.
    s2_nodes = list(s2.nodes())
    s2_edges = list(s2.edges())
    relocate = {}
    for n in s2_nodes:
        if (isinstance(n, nodes.AccessNode) and n.data in s1_writes
                and s2.out_degree(n) > 0 and s2.in_degree(n) == 0):
            relocate[n] = s1_writes[n.data]
    # Drop s2's edges first (so the upcoming node moves don't leave stale incidences).
    for e in s2_edges:
        s2.remove_edge(e)
    # Move every non-aliased s2 node to s1.
    for n in s2_nodes:
        if n in relocate:
            s2.remove_node(n)
            continue
        s2.remove_node(n)
        s1.add_node(n)
    # Re-add the edges on s1, mapping endpoints through ``relocate``.
    for e in s2_edges:
        src = relocate.get(e.src, e.src)
        dst = relocate.get(e.dst, e.dst)
        s1.add_edge(src, e.src_conn, dst, e.dst_conn, e.data)


def _match_all(loop: LoopRegion, sdfg: SDFG) -> List[_Scan]:
    """Match every independent scan recurrence in ``loop``.

    v4 -- multi-array bodies: a single loop may carry several independent prefix
    scans (cloudsc ``pfsqrf`` carries five: ``pfsqif`` / ``pfsqrf`` / ``pfsqlf`` /
    ``pfsqsf`` / ``pfcqlng``). Each carry has its own scan-update tasklet, its own
    delta input, and its own seed; the rewrite emits one delta-build Map + Scan
    libnode + seed-add Map *per carry*, all in front of (resp. behind) the
    surviving loop. The single-array (v1/v2) case is the special case of
    returning a one-element list.

    A loop is fully accepted only when **every** non-transient array with a
    loop-variable-dependent read+write matches as a scan -- partial scanning is
    not safe (a refused carrier would still iterate sequentially, and the other
    carriers would no longer be in the loop's body to schedule around it).

    :param loop: The LoopRegion to inspect.
    :param sdfg: The owning SDFG.
    :returns: Per-carrier ``_Scan`` infos in deterministic order, or an empty
              list if the loop fails any pre-condition or any single carrier
              fails to match.
    """
    start = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if start is None or end is None or stride is None or stride != 1:
        return []

    # Body shape: either (a) a single flat content state (the v1-v5 case --
    # only tasklets and AccessNodes), or (b) one or more nested ``LoopRegion``
    # candidates that themselves wrap a flat content state (the cloudsc
    # ``for_1133`` case -- outer scan over levels containing an inner data-
    # parallel column loop, possibly alongside slice-prep inner loops or
    # extra content states). Empty wrapper states are tolerated -- see
    # ``apply_pass``. Each candidate ``(state, inner_loop)`` is tried; the
    # first one whose state has carriers + matchable scan-update wins.
    candidates = _descend_to_content_state_candidates(loop)
    if not candidates:
        return []
    matched: Optional[List[_Scan]] = None
    for state, inner_loop in candidates:
        bad_node = False
        for n in state.nodes():
            if not isinstance(n, (nodes.Tasklet, nodes.AccessNode)):
                bad_node = True
                break
        if bad_node:
            continue
        carriers = _find_carried_arrays(state, sdfg, loop.loop_variable)
        if not carriers:
            continue
        cand_matched: List[_Scan] = []
        cand_failed = False
        for out_name in carriers:
            info = _match_one_carrier(loop, sdfg, state, out_name, start, end)
            if info is None:
                cand_failed = True
                break
            if inner_loop is not None:
                info = info._replace(inner_loop=inner_loop)
                if not _other_indices_match_inner(info.other_indices, inner_loop.loop_variable):
                    cand_failed = True
                    break
            cand_matched.append(info)
        if cand_failed or not cand_matched:
            continue
        matched = cand_matched
        break
    if matched is None:
        return []

    # Body may write only the carriers we matched. Any other non-transient write
    # would need ``lastprivate``-style preservation that the rewrite doesn't model.
    carrier_set = {s.out_name for s in matched}
    for st in loop.all_states():
        for node in st.data_nodes():
            if st.in_degree(node) == 0:
                continue
            desc = sdfg.arrays.get(node.data)
            if desc is None or getattr(desc, 'transient', False):
                continue
            if node.data not in carrier_set:
                return []
    return matched


def _other_indices_match_inner(other_indices: List[Any], inner_var: str) -> bool:
    """The non-scan-axis entries must consist of exactly ONE axis whose index is the
    inner loop's iterator symbol, plus any number of axes whose index does NOT
    reference ``inner_var`` (loop-invariant constants or enclosing-scope symbols
    -- e.g. ``arr[species, jk, jl]`` where ``species`` is a constant fixed
    outside the scan and ``jl`` is the inner-loop var). The vector-scan rewrite
    uses the inner-var axis as the Map dimension; the loop-invariant axes are
    threaded through verbatim and contribute no Map iter dim.
    """
    inner_sym = symbolic.pystr_to_symbolic(inner_var)
    inner_count = 0
    for _axis, expr in other_indices:
        try:
            e_sym = symbolic.pystr_to_symbolic(str(expr))
        except Exception:
            return False
        try:
            is_inner = bool(symbolic.simplify(e_sym - inner_sym) == 0)
        except Exception:
            is_inner = False
        if is_inner:
            inner_count += 1
            continue
        # Axis is non-inner: must NOT reference inner_var at all (else it's a
        # composite axis that we can't safely separate from the Map dim).
        if inner_var in {str(s) for s in e_sym.free_symbols}:
            return False
    return inner_count == 1


def _match_one_carrier(loop: LoopRegion, sdfg: SDFG, state: SDFGState, out_name: str,
                       iter_start: Any, iter_end: Any) -> Optional[_Scan]:
    """Match the scan recurrence for a single carrier ``out_name``.

    When the body has multiple write AccessNodes for ``out_name`` (the v5 fused-body
    case, where each pre-fuse state contributed a write to a different subset), tries
    each write edge in turn and picks the one whose subset + scan-update tasklet form
    a valid recurrence. Returns the matched ``_Scan`` info or ``None`` if no write
    edge yields a successful match.
    """
    candidates = []
    for write_edge in _iter_write_edges(state, out_name):
        if write_edge.data is None or write_edge.data.subset is None:
            continue
        write_axis, k_w, write_others = _classify_subset(write_edge.data.subset, loop.loop_variable)
        if write_axis is None:
            continue
        cand = _find_scan_update_tasklet(state, sdfg, out_name, loop.loop_variable,
                                         write_axis, write_others, k_w)
        if cand is None:
            continue
        tasklet, carry_edge, delta_edge, op, carry_anchor, scan_stride, literal_delta = cand
        out_edges_t = [e for e in state.out_edges(tasklet)
                       if e.data is not None and not e.data.is_empty()]
        if len(out_edges_t) != 1:
            continue
        k_r = symbolic.simplify(k_w - scan_stride)
        candidates.append(_Scan(
            op=op,
            out_name=out_name,
            scan_axis=write_axis,
            k_w=k_w,
            k_r=k_r,
            scan_stride=scan_stride,
            other_indices=write_others,
            iter_start=iter_start,
            iter_end=iter_end,
            body_state=state,
            scan_update_tasklet=tasklet,
            carry_in_conn=carry_edge.dst_conn,
            delta_in_conn=delta_edge.dst_conn if delta_edge is not None else None,
            out_conn=out_edges_t[0].src_conn,
            carry_anchor=carry_anchor,
            literal_delta=literal_delta,
        ))
    # Exactly one write-edge classification must yield a valid scan match. More than
    # one would mean two recurrences are written to ``out_name`` in the same body
    # (we cannot pick between them); fewer means no scan shape.
    if len(candidates) != 1:
        return None
    return candidates[0]


def _match(loop: LoopRegion, sdfg: SDFG) -> Optional[_Scan]:
    """Single-carrier match (kept for callers that handled one carry at a time)."""
    infos = _match_all(loop, sdfg)
    if len(infos) != 1:
        return None
    return infos[0]


def _match_scalar_carry(loop: LoopRegion, sdfg: SDFG) -> Optional[_ScalarCarryScan]:
    """Match the scalar-carry prefix-scan shape (TSVC s3112).

    Shape::

        acc(read) → Tasklet(__out = __acc OP __delta) → acc(write) → out[i + d, ...]
                              ↑
                       delta source AN at delta[i + d', ...]

    Where:

    * ``acc`` is a transient ``Scalar`` (or length-1 ``Array``) read+written in
      the body via a single RMW tasklet.
    * The post-RMW value of ``acc`` (its write AN) feeds a SINGLE per-iter
      write of one non-transient array ``out`` at ``out[loop_var + out_offset,
      ...]`` (possibly via TrivialTaskletElimination-folded AN-AN copies).
    * The op is one of ``+`` / ``*`` / ``max`` / ``min``.
    * The delta arrives via a non-transient array ``delta[loop_var + d', ...]``
      (gathered into a transient slice; same shape as the array-carry matcher's
      ``_resolve_input``).
    * The body must not write any other non-transient array.

    Refusals:

    * Loop stride != 1 (mirrors array-carry matcher).
    * Body has any block other than the single content state (no nested
      LoopRegion / ConditionalBlock).
    * ``acc`` not a transient scalar / length-1 array.
    * Multiple writes to ``acc`` in the body (only single-RMW chains).
    * Multiple writes from ``acc`` to per-iter outputs (more than one prefix
      emitter is out of scope for v1).
    * The non-RMW reader of ``acc`` is anything other than the per-iter
      output write -- e.g. ``c[i] = sum * 2`` would refuse.
    """
    start = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if start is None or end is None or stride is None or stride != 1:
        return None

    state, inner_loop = _descend_to_content_state(loop)
    if state is None or inner_loop is not None:
        return None  # v1 keeps narrow on flat bodies; nested-scalar-carry is a follow-up.
    for n in state.nodes():
        if not isinstance(n, (nodes.Tasklet, nodes.AccessNode)):
            return None

    loop_var = loop.loop_variable

    # 1. Find a transient scalar accumulator with the RMW shape.
    candidates: List[_ScalarCarryScan] = []
    for acc_name in _find_scalar_carry_candidates(state, sdfg):
        m = _match_one_scalar_carry(loop, sdfg, state, acc_name, start, end, loop_var)
        if m is not None:
            candidates.append(m)
    if len(candidates) != 1:
        return None
    info = candidates[0]

    # 2. Body must not write any non-transient array other than ``info.out_name``.
    for n in state.data_nodes():
        if state.in_degree(n) == 0:
            continue
        desc = sdfg.arrays.get(n.data)
        if desc is None or getattr(desc, 'transient', False):
            continue
        if n.data != info.out_name:
            return None
    return info


def _find_scalar_carry_candidates(state: SDFGState, sdfg: SDFG) -> List[str]:
    """Names of transient scalars (or length-1 arrays) read AND written in
    ``state`` -- the candidate accumulators. Sorted for determinism.
    """
    candidates: set = set()
    reads: set = set()
    writes: set = set()
    for n in state.data_nodes():
        desc = sdfg.arrays.get(n.data)
        if desc is None or not getattr(desc, 'transient', False):
            continue
        is_scalar = isinstance(desc, data.Scalar) or (isinstance(desc, data.Array)
                                                     and tuple(desc.shape) == (1,))
        if not is_scalar:
            continue
        candidates.add(n.data)
        if state.in_degree(n) > 0:
            writes.add(n.data)
        if state.out_degree(n) > 0:
            reads.add(n.data)
    return sorted(candidates & reads & writes)


def _match_one_scalar_carry(loop: LoopRegion, sdfg: SDFG, state: SDFGState, acc_name: str,
                             iter_start: Any, iter_end: Any,
                             loop_var: str) -> Optional[_ScalarCarryScan]:
    """Match the scalar-carry shape for one candidate accumulator ``acc_name``.

    Returns ``None`` on any refusal.
    """
    # Locate the unique READ AN (in_degree == 0, out_degree > 0) and unique WRITE
    # AN (in_degree > 0, out_degree > 0 -- the write AN also feeds the per-iter
    # output, so out_degree must be > 0; pure sink ANs are not the target shape).
    read_an: Optional[nodes.AccessNode] = None
    write_an: Optional[nodes.AccessNode] = None
    for n in state.data_nodes():
        if n.data != acc_name:
            continue
        in_d, out_d = state.in_degree(n), state.out_degree(n)
        if in_d == 0 and out_d > 0:
            if read_an is not None:
                return None
            read_an = n
        elif in_d > 0 and out_d > 0:
            if write_an is not None:
                return None
            write_an = n
        elif in_d > 0 and out_d == 0:
            # Pure sink AN -- the post-RMW value doesn't feed an output, so this
            # isn't a scalar-carry scan (could be a plain reduction; LoopToReduce's
            # job). Refuse.
            return None
        # in_d == 0 and out_d == 0 -- isolated AN, ignore.
    if read_an is None or write_an is None:
        return None

    # Walk back from the WRITE AN through copy chain to find the RMW tasklet.
    tasklet, acc_in_conn, delta_in_conn, out_conn, op = _trace_back_to_rmw_tasklet(
        state, write_an, acc_name)
    if tasklet is None:
        return None

    # The acc_in_conn's source must trace back (through copies) to the READ AN.
    acc_in_edges = [e for e in state.in_edges(tasklet) if e.dst_conn == acc_in_conn]
    if len(acc_in_edges) != 1:
        return None
    if not _input_traces_to(state, acc_in_edges[0].src, read_an):
        return None

    # The delta_in_conn's source must trace back to a per-iter gather of a
    # non-transient array at offset ``delta[loop_var + d', ...]``.
    delta_in_edges = [e for e in state.in_edges(tasklet) if e.dst_conn == delta_in_conn]
    if len(delta_in_edges) != 1:
        return None
    delta_info = _resolve_per_iter_gather(state, delta_in_edges[0], sdfg, loop_var)
    if delta_info is None:
        return None
    delta_name, delta_axis, delta_offset, delta_others = delta_info

    # The WRITE AN's out-edges feed exactly one per-iter array write.
    write_out_edges = list(state.out_edges(write_an))
    if len(write_out_edges) != 1:
        return None
    out_info = _resolve_per_iter_write(state, write_out_edges[0], sdfg, loop_var, acc_name)
    if out_info is None:
        return None
    out_name, out_axis, out_offset, out_others = out_info

    # Check if ``acc`` is read outside this loop's body (any other state of any
    # nested SDFG, or any interstate edge condition/assignments). When true the
    # rewrite emits a post-loop writeback to preserve the final value.
    acc_used_post_loop = _acc_used_outside_body(sdfg, acc_name, state)

    return _ScalarCarryScan(
        op=op,
        acc_name=acc_name,
        out_name=out_name,
        out_scan_axis=out_axis,
        out_offset=out_offset,
        out_other_indices=out_others,
        delta_name=delta_name,
        delta_scan_axis=delta_axis,
        delta_offset=delta_offset,
        delta_other_indices=delta_others,
        iter_start=iter_start,
        iter_end=iter_end,
        body_state=state,
        scan_update_tasklet=tasklet,
        acc_in_conn=acc_in_conn,
        delta_in_conn=delta_in_conn,
        out_conn=out_conn,
        acc_read_an=read_an,
        acc_write_an=write_an,
        acc_used_post_loop=acc_used_post_loop,
    )


def _trace_back_to_rmw_tasklet(state: SDFGState, write_an: nodes.AccessNode,
                                acc_name: str):
    """Walk back from ``write_an`` through at most one transient single-use
    intermediate scalar AN to the RMW tasklet. Returns
    ``(tasklet, acc_in_conn, delta_in_conn, out_conn, op)`` on success or
    ``(None, None, None, None, None)`` on any refusal.
    """
    ins = list(state.in_edges(write_an))
    if len(ins) != 1:
        return None, None, None, None, None
    src = ins[0].src
    # Optional one-hop intermediate transient scalar AN.
    if isinstance(src, nodes.AccessNode):
        sub_ins = list(state.in_edges(src))
        if len(sub_ins) != 1 or state.out_degree(src) != 1:
            return None, None, None, None, None
        src = sub_ins[0].src
    if not isinstance(src, nodes.Tasklet):
        return None, None, None, None, None
    tasklet = src

    # Parse the tasklet body: ``__out = __acc OP __delta`` (or ``op(a, b)`` call).
    try:
        tree = ast.parse((tasklet.code.as_string or '').strip())
    except SyntaxError:
        return None, None, None, None, None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        return None, None, None, None, None
    assign = tree.body[0]
    if len(assign.targets) != 1 or not isinstance(assign.targets[0], ast.Name):
        return None, None, None, None, None
    out_conn = assign.targets[0].id
    rhs = assign.value
    op = None
    name_a, name_b = None, None
    if isinstance(rhs, ast.BinOp):
        op = _BINOP_TO_SCAN_OP.get(type(rhs.op))
        if op is None:
            return None, None, None, None, None
        if not (isinstance(rhs.left, ast.Name) and isinstance(rhs.right, ast.Name)):
            return None, None, None, None, None
        name_a, name_b = rhs.left.id, rhs.right.id
    elif isinstance(rhs, ast.Call) and isinstance(rhs.func, ast.Name) and len(rhs.args) == 2:
        op = _CALL_TO_SCAN_OP.get(rhs.func.id)
        if op is None or not all(isinstance(a, ast.Name) for a in rhs.args):
            return None, None, None, None, None
        name_a, name_b = rhs.args[0].id, rhs.args[1].id
    else:
        return None, None, None, None, None

    # Identify which input connector is the accumulator by tracing each input
    # edge's source back: the one whose source AN is ``acc_name`` is __acc.
    in_data_edges = [e for e in state.in_edges(tasklet)
                     if e.data is not None and not e.data.is_empty()]
    if len(in_data_edges) != 2:
        return None, None, None, None, None
    acc_conn: Optional[str] = None
    delta_conn: Optional[str] = None
    for e in in_data_edges:
        if _source_is_acc(state, e.src, acc_name):
            if acc_conn is not None:
                return None, None, None, None, None
            acc_conn = e.dst_conn
        else:
            delta_conn = e.dst_conn
    if acc_conn is None or delta_conn is None:
        return None, None, None, None, None
    if {acc_conn, delta_conn} != {name_a, name_b}:
        return None, None, None, None, None
    return tasklet, acc_conn, delta_conn, out_conn, op


def _source_is_acc(state: SDFGState, src, acc_name: str) -> bool:
    """``True`` iff ``src`` is, or traces back through identity transient ANs to,
    an AccessNode of ``acc_name``."""
    cur = src
    seen = set()
    while cur not in seen:
        seen.add(cur)
        if isinstance(cur, nodes.AccessNode):
            if cur.data == acc_name:
                return True
            ins = list(state.in_edges(cur))
            if len(ins) != 1:
                return False
            cur = ins[0].src
            continue
        return False
    return False


def _input_traces_to(state: SDFGState, src, target_an: nodes.AccessNode) -> bool:
    """``True`` iff ``src`` is ``target_an`` directly, or transitively via
    transient AN passthroughs (in=1, out=1) lands at ``target_an``."""
    cur = src
    seen = set()
    while cur not in seen:
        seen.add(cur)
        if cur is target_an:
            return True
        if not isinstance(cur, nodes.AccessNode):
            return False
        ins = list(state.in_edges(cur))
        if len(ins) != 1:
            return False
        cur = ins[0].src
    return False


def _resolve_per_iter_gather(state: SDFGState, edge, sdfg: SDFG, loop_var: str):
    """Trace ``edge`` (a tasklet input edge) back through transient ANs to the
    source non-transient array and the gather subset. Returns
    ``(arr_name, scan_axis, scan_offset, other_indices)`` or ``None`` if the
    subset doesn't classify as a loop-variable affine-axis-with-offset gather.

    The gather's array-level subset lives on the edge connecting the source
    non-transient AN to the first transient intermediate (memlet
    ``data=arr, subset=arr[...]``), not on any later edge in the chain --
    transient-to-tasklet edges carry ``data=transient, subset=[0]`` after
    ``TrivialTaskletElimination`` folds the copy tasklets.
    """
    cur = edge.src
    last_edge = edge
    seen = set()
    while cur not in seen:
        seen.add(cur)
        if not isinstance(cur, nodes.AccessNode):
            return None
        desc = sdfg.arrays.get(cur.data)
        if desc is None:
            return None
        if not getattr(desc, 'transient', False):
            # Reached the non-transient source. The array-level gather subset
            # is on ``last_edge`` (the most recent edge we traversed: it goes
            # from this source AN to the first transient intermediate; its
            # memlet's ``data`` is the source array's name).
            sub = last_edge.data.subset if last_edge.data is not None else None
            if sub is None or last_edge.data.data != cur.data:
                # Fall back: check the AN's out-edge (chain start) for the
                # case where the chain has zero transient hops -- the tasklet
                # reads directly from the non-transient.
                sub = last_edge.data.subset if last_edge.data is not None else None
                if sub is None:
                    return None
            scan_axis, offset, others = _classify_subset(sub, loop_var)
            if scan_axis is None:
                return None
            return cur.data, scan_axis, offset, others
        ins = list(state.in_edges(cur))
        if len(ins) != 1:
            return None
        last_edge = ins[0]
        cur = last_edge.src
    return None


def _resolve_per_iter_write(state: SDFGState, edge, sdfg: SDFG, loop_var: str, acc_name: str):
    """Trace ``edge`` (an out-edge of the acc-write AN) forward through any
    transient pass-through ANs to the destination non-transient array, and
    classify the write subset on the loop iterator. Returns
    ``(arr_name, scan_axis, offset, other_indices)`` or ``None`` if no
    well-defined per-iter write reaches a non-transient destination.

    Refuses when:

    * the destination's subset on ``loop_var`` doesn't have the affine
      single-point shape ``(loop_var + const, ...)`` on exactly one axis;
    * the chain ever splits / hits a tasklet (anything other than direct
      AN-AN memlets to passthrough transients).
    """
    cur_edge = edge
    seen = set()
    while True:
        dst = cur_edge.dst
        if dst in seen:
            return None
        seen.add(dst)
        if isinstance(dst, nodes.AccessNode):
            desc = sdfg.arrays.get(dst.data)
            if desc is None:
                return None
            if not getattr(desc, 'transient', False):
                sub = cur_edge.data.subset
                if sub is None:
                    return None
                scan_axis, offset, others = _classify_subset(sub, loop_var)
                if scan_axis is None:
                    return None
                return dst.data, scan_axis, offset, others
            # passthrough transient: continue along the unique out-edge.
            if state.out_degree(dst) != 1:
                return None
            cur_edge = list(state.out_edges(dst))[0]
            continue
        return None


def _acc_used_outside_body(sdfg: SDFG, acc_name: str, body_state: SDFGState) -> bool:
    """``True`` iff ``acc_name`` is READ in any state other than ``body_state``,
    or appears in any interstate edge's condition / assignment RHS.

    Pre-state WRITES of ``acc`` (e.g. the seed ``sum = 0.0``) do NOT count as
    a use -- the rewrite preserves the pre-state untouched and the Scan reads
    its result as the seed. Only DOWNSTREAM reads (states whose value of
    ``acc`` would observe the post-loop running total) force the post-loop
    writeback.
    """
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.all_states():
            if state is body_state:
                continue
            for n in state.data_nodes():
                if n.data == acc_name and state.out_degree(n) > 0:
                    # ``acc`` is read in this state -- a downstream consumer.
                    return True
        for ise in sd.all_interstate_edges():
            for s in ise.data.free_symbols:
                if str(s) == acc_name:
                    return True
            for _lhs, rhs in (ise.data.assignments or {}).items():
                if acc_name in str(rhs):
                    return True
    return False


def _find_carried_arrays(state: SDFGState, sdfg: SDFG, loop_var: str) -> List[str]:
    """All non-transient arrays read AND written in ``state`` with subsets that depend
    on ``loop_var`` -- the candidate scan carriers. Sorted by name for determinism.

    Multi-carrier loops (cloudsc ``pfsqrf``: 5 parallel prefix sums in one body)
    return a multi-element list; v1/v2 single-carrier loops return a one-element list.

    Scans ALL edges in the state -- top-level edges plus any inside Map scopes -- for
    memlets whose data is a non-transient and whose subset depends on ``loop_var``.
    Walking edges rather than only AccessNode-incident memlets catches carriers whose
    state-level memlets have been widened by Map-exit propagation (defensive: the
    Map-wrapped path isn't reached by the nested-LoopRegion rewrite, but the edge-walk
    is also what makes the v1-v5 flat case robust to the slice-copy intermediates).
    """
    arrays_with_carrier_subset: set = set()
    for e in state.edges():
        m = e.data
        if m is None or m.data is None or m.subset is None:
            continue
        desc = sdfg.arrays.get(m.data)
        if desc is None or getattr(desc, 'transient', False):
            continue
        if _subset_uses(m.subset, loop_var):
            arrays_with_carrier_subset.add(m.data)

    reads: set = set()
    writes: set = set()
    for n in state.data_nodes():
        desc = sdfg.arrays.get(n.data)
        if desc is None or getattr(desc, 'transient', False):
            continue
        if n.data not in arrays_with_carrier_subset:
            continue
        if state.in_degree(n) > 0:
            writes.add(n.data)
        if state.out_degree(n) > 0:
            reads.add(n.data)
    return sorted(reads & writes)


def _find_carried_array(state: SDFGState, sdfg: SDFG, loop_var: str) -> Optional[str]:
    """Single-carrier convenience wrapper (kept for callers that demand exactly one)."""
    carriers = _find_carried_arrays(state, sdfg, loop_var)
    if len(carriers) != 1:
        return None
    return carriers[0]


def _find_unique_write_edge(state: SDFGState, name: str):
    """Locate the unique in-edge to a non-transient AccessNode of ``name``.

    Returns the unique write edge, or ``None`` on ambiguity. Use ``_iter_write_edges``
    when ambiguity must be disambiguated downstream (e.g. distinguishing a scan-carry
    write from a side-effect write to a different subset).
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


def _iter_write_edges(state: SDFGState, name: str) -> List[Any]:
    """Yield every in-edge to any non-transient AccessNode of ``name``.

    Used by the multi-write disambiguator: when the body-state-fusion preprocess
    merges two states each writing the same carrier at a *different* subset
    (one side-effect ``out[i]`` and the scan-carry ``out[i+1]``), both writes
    survive as separate AccessNodes. The scan match tries each in turn and
    picks the one whose subset + scan-update tasklet match the recurrence shape.
    """
    edges: List[Any] = []
    for n in state.data_nodes():
        if n.data != name:
            continue
        edges.extend(state.in_edges(n))
    return edges


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
        if len(out_edges) != 1:
            continue
        # v1/v2: ``carry + delta`` (two data inputs). v3: ``carry + literal``
        # (one data input + a numeric constant on the other side of the binop;
        # e.g. TSVC s242 ``a[i] = a[i-1] + 0.5 + ...``). The Call (``max``/``min``)
        # form keeps the two-data-input requirement -- literal arms to ``max``
        # would degenerate to ``out[i] = max(out[start], literal)`` which is
        # better lifted as a plain reduction, not a scan.
        if len(in_edges) not in (1, 2):
            continue
        literal_delta = None
        if len(in_edges) == 1:
            if not isinstance(rhs, ast.BinOp):
                continue
            literal_delta = _binop_literal_operand(rhs)
            if literal_delta is None:
                continue
        carry_edge = None
        delta_edge = None
        carry_anchor = None
        scan_stride = None
        ambiguous = False
        for e in in_edges:
            src_name, src_subset = _resolve_input(state, e)
            if src_name == out_name and src_subset is not None:
                r_axis, k_r_cand, r_others = _classify_subset(src_subset, loop_var)
                # Accept any *positive integer* offset ``k_w - k_r_cand``: ``1``
                # is the contiguous scan; ``S > 1`` is the residue-class scan
                # that the ``Scan`` libnode's ``stride`` property handles
                # natively (TSVC s1221 ``b[i] = b[i-4] + a[i]``). For ``S > 1``
                # the seed-add Map fans the ``S`` pre-loop seeds out by
                # ``_i mod S`` (see ``_emit_seed_add``).
                if r_axis == scan_axis and _same_other_indices(r_others, write_others):
                    try:
                        diff = symbolic.simplify(k_w - k_r_cand)
                    except Exception:
                        diff = None
                    if (diff is not None and getattr(diff, 'is_number', False)
                            and getattr(diff, 'is_Integer', False) and int(diff) >= 1):
                        if carry_edge is not None:
                            ambiguous = True
                            break
                        carry_edge = e
                        carry_anchor = e.src
                        scan_stride = int(diff)
                        continue
            if delta_edge is not None:
                # Two non-carry inputs -- this isn't the scan-update tasklet.
                delta_edge = None
                break
            # An edge that reads the carry array ``out`` but did NOT match the
            # carry shape is an extra read of the carry at a different position
            # (e.g. TSVC s2111: ``aa[j, i] = (aa[j, i-1] + aa[j-1, i]) / 1.9``
            # -- ``aa[j-1, i]`` reads ``aa`` at a different ROW). The body's
            # recurrence is then *not* a pure scan ``out[i] = out[i-1] OP
            # delta(non-out arrays)``; lifting it as one corrupts the result.
            if src_name == out_name:
                ambiguous = True
                break
            delta_edge = e
        if ambiguous or carry_edge is None or scan_stride is None:
            continue
        # v1/v2 must have a data delta edge; v3 has a literal delta instead.
        if delta_edge is None and literal_delta is None:
            continue
        return node, carry_edge, delta_edge, op, carry_anchor, scan_stride, literal_delta
    return None


def _binop_literal_operand(rhs: ast.BinOp) -> Optional[str]:
    """If exactly one operand of ``rhs`` is a numeric literal (``ast.Constant``
    of int/float, optionally negated via ``ast.UnaryOp(USub, Constant)``),
    return its source-text form. Otherwise return ``None``.

    Symmetric: ``carry + 0.5`` and ``0.5 + carry`` both match. The associative
    ops we support (``+``, ``*``, ``min``, ``max``) all commute, so operand
    order is irrelevant for the scan semantics.
    """
    def _as_literal_str(n) -> Optional[str]:
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return repr(n.value)
        if (isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub)
                and isinstance(n.operand, ast.Constant)
                and isinstance(n.operand.value, (int, float))
                and not isinstance(n.operand.value, bool)):
            return f'-{n.operand.value!r}'
        return None

    left_lit = _as_literal_str(rhs.left)
    right_lit = _as_literal_str(rhs.right)
    if left_lit is not None and right_lit is None:
        return left_lit
    if right_lit is not None and left_lit is None:
        return right_lit
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
    """Walk back from a tasklet input edge to its source AccessNode, traversing both
    1-hop slice-copy transient intermediates (``arr -> arr_index -> tasklet``) and
    Map scope boundaries (``arr -> MapEntry -> tasklet``). Returns
    ``(arr_name, real_subset)`` or ``(None, None)``.

    The "real" subset is the one taken on the edge nearest the carrier AccessNode
    on the interior side of any propagation-widening boundary -- i.e. the inside-Map
    edge for the Map case (the outside edge is widened by Map-exit propagation), and
    the upstream-of-transient edge for the slice-copy case (the downstream slice edge
    typically holds a scalar dereference subset).
    """
    cur = edge
    real_subset = cur.data.subset if cur.data is not None else None
    seen = set()
    while True:
        if id(cur) in seen:
            return None, None
        seen.add(id(cur))
        src = cur.src
        if isinstance(src, nodes.MapEntry):
            src_conn = cur.src_conn
            if not src_conn or not src_conn.startswith('OUT_'):
                return None, None
            in_conn = 'IN_' + src_conn[len('OUT_'):]
            outer_edges = [e for e in state.in_edges(src) if e.dst_conn == in_conn]
            if len(outer_edges) != 1:
                return None, None
            cur = outer_edges[0]
            continue
        if not isinstance(src, nodes.AccessNode):
            return None, None
        desc = state.sdfg.arrays.get(src.data)
        if desc is None:
            return None, None
        if not getattr(desc, 'transient', False):
            return src.data, real_subset
        if state.in_degree(src) != 1 or state.out_degree(src) != 1:
            return None, None
        pred = state.in_edges(src)[0]
        if pred.data is None or pred.data.subset is None:
            return None, None
        real_subset = pred.data.subset
        cur = pred


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
    """Rewrite ``loop`` into a chain via in-place body mutation.

    The body is always mutated to write the per-iteration delta to a 1-D transient
    (``_scan_in[loop_var - start]``); the post-loop chain follows one of two shapes:

    * **1-D direct-write** (``out`` is 1-D and the matched write has no other-axis
      indices). One state runs the ``Scan`` libnode with the optional ``_scan_init``
      connector wired to ``out[start + k_r]``; the libnode's inclusive-with-init
      semantics fold the seed in, so the scan output is written directly to
      ``out[start + k_w : start + k_w + trip]`` and no seed-add Map is emitted.
    * **General path** (multi-dim ``out`` or non-empty other-axis indices). Two
      states: a Scan into a 1-D transient ``_scan_out``, then a seed-add Map that
      writes ``out[start + k_w + _i, ...] = seed OP _scan_out[_i]``.
    * **Nested case** (``info.inner_loop is not None``): the outer scan body
      wraps an inner data-parallel loop. The delta buffer becomes 2-D
      ``[trip, inner_size]``; the post-loop emits a ``Map`` over the inner loop's
      iterator wrapping the 1-D ``Scan`` along the outer axis (vector scan).

    :param parent: CFG owning ``loop``.
    :param loop: The matched scan loop.
    :param info: Matcher output describing the scan-update tasklet, axes, and offsets.
    :param sdfg: SDFG owning ``loop``.
    """
    if info.inner_loop is not None:
        _rewrite_nested(parent, loop, info, sdfg)
        return
    import dace
    out_desc = sdfg.arrays[info.out_name]
    trip = symbolic.simplify(info.iter_end - info.iter_start + 1)
    delta_buf, _ = sdfg.add_array(f'{_DELTA_BUF_PREFIX}{info.out_name}', [trip], out_desc.dtype, transient=True,
                                  find_new_name=True)

    _mutate_body_to_delta_buffer(info, delta_buf)

    out_edges = list(parent.out_edges(loop))
    s_scan = parent.add_state(loop.label + '_scan')
    parent.add_edge(loop, s_scan, dace.InterstateEdge())

    if _can_emit_direct_write(info, out_desc):
        for e in out_edges:
            parent.remove_edge(e)
            parent.add_edge(s_scan, e.dst, e.data)
        _emit_scan_with_init_direct(s_scan, sdfg, info, delta_buf, trip)
    else:
        scan_buf, _ = sdfg.add_array(f'{_SCAN_BUF_PREFIX}{info.out_name}', [trip], out_desc.dtype, transient=True,
                                     find_new_name=True)
        s_apply = parent.add_state(loop.label + '_scan_apply')
        parent.add_edge(s_scan, s_apply, dace.InterstateEdge())
        for e in out_edges:
            parent.remove_edge(e)
            parent.add_edge(s_apply, e.dst, e.data)
        _emit_scan(s_scan, sdfg, info, delta_buf, scan_buf, trip)
        _emit_seed_add(s_apply, sdfg, info, scan_buf, trip)
    sdfg.reset_cfg_list()


def _rewrite_nested(parent: ControlFlowRegion, loop: LoopRegion, info: _Scan, sdfg: SDFG):
    """Vector-scan rewrite: outer scan over the carrier's scan axis, inner
    data-parallel loop over the carrier's non-scan axis.

    Allocates ``delta_buf`` and ``scan_buf`` with shape ``[trip, inner_size]``.
    The body mutation re-routes the carrier write to ``delta_buf[outer_iter -
    iter_start, inner_var]``; the inner LoopRegion is left intact (it now writes
    only ``delta_buf`` and has no loop-carried dependence, so a follow-up
    ``LoopToMap`` can lift it). Post-loop emits a ``Map`` over the inner
    iterator wrapping a 1-D ``Scan`` along the outer axis, followed by a
    seed-add ``Map`` over the (outer-iter, inner-iter) product.
    """
    import dace
    inner = info.inner_loop
    inner_var = inner.loop_variable
    inner_start = loop_analysis.get_init_assignment(inner)
    inner_end = loop_analysis.get_loop_end(inner)
    inner_stride = loop_analysis.get_loop_stride(inner)
    if inner_start is None or inner_end is None or inner_stride is None or inner_stride != 1:
        return
    inner_size = symbolic.simplify(inner_end - inner_start + 1)

    out_desc = sdfg.arrays[info.out_name]
    trip = symbolic.simplify(info.iter_end - info.iter_start + 1)
    # Layout: ``[trip, inner_size]`` -- the natural carrier-matching shape (outer
    # axis first, inner axis second). The single ``Scan`` libnode handles all
    # ``inner_size`` independent per-column prefix sums via its ``stride`` property
    # (``stride = inner_size``, residue-class scan): elements ``buf[i, j]`` and
    # ``buf[i + 1, j]`` are exactly ``inner_size`` positions apart in memory, so
    # the libnode treats them as adjacent in the j-th stream of an
    # ``inner_size``-way interleaved scan. No transpose needed.
    delta_buf, _ = sdfg.add_array(f'{_DELTA_BUF_PREFIX}{info.out_name}', [trip, inner_size], out_desc.dtype,
                                  transient=True, find_new_name=True)
    scan_buf, _ = sdfg.add_array(f'{_SCAN_BUF_PREFIX}{info.out_name}', [trip, inner_size], out_desc.dtype,
                                 transient=True, find_new_name=True)

    _mutate_body_to_delta_buffer_nested(info, delta_buf, inner_start)

    out_edges = list(parent.out_edges(loop))
    s_scan = parent.add_state(loop.label + '_scan')
    s_apply = parent.add_state(loop.label + '_scan_apply')
    parent.add_edge(loop, s_scan, dace.InterstateEdge())
    parent.add_edge(s_scan, s_apply, dace.InterstateEdge())
    for e in out_edges:
        parent.remove_edge(e)
        parent.add_edge(s_apply, e.dst, e.data)

    _emit_scan_nested(s_scan, sdfg, info, delta_buf, scan_buf, trip, inner_var, inner_start, inner_end)
    _emit_seed_add_nested(s_apply, sdfg, info, scan_buf, trip, inner_var, inner_start, inner_end)
    sdfg.reset_cfg_list()


def _nested_axis_kinds(info: _Scan, inner_var: str):
    """Classify each non-scan axis of the carrier as either the inner-Map axis
    (whose subset expression equals ``inner_var``) or a constant / loop-invariant
    axis (passed through verbatim). Returns ``(inner_axis_idx, const_axes)``
    where ``const_axes`` is a list of ``(axis_idx, const_expr)``.
    """
    inner_sym = symbolic.pystr_to_symbolic(inner_var)
    inner_axis_idx = None
    const_axes = []
    for axis_idx, expr in info.other_indices:
        try:
            is_inner = bool(symbolic.simplify(symbolic.pystr_to_symbolic(str(expr)) - inner_sym) == 0)
        except Exception:
            is_inner = False
        if is_inner:
            inner_axis_idx = axis_idx
        else:
            const_axes.append((axis_idx, expr))
    return inner_axis_idx, const_axes


def _build_nested_carrier_subset(desc: data.Array, info: _Scan, scan_expr,
                                 inner_var: str, inner_axis_expr) -> subsets.Range:
    """Build a full N-dim subset for the carrier: ``scan_expr`` on the scan axis,
    ``inner_axis_expr`` on the inner-var axis, and the per-axis constant expression
    on every other axis.
    """
    inner_axis_idx, const_axes = _nested_axis_kinds(info, inner_var)
    const_map = {axis: expr for axis, expr in const_axes}
    rng = []
    for axis_idx in range(len(desc.shape)):
        if axis_idx == info.scan_axis:
            rng.append((scan_expr, scan_expr, 1))
        elif axis_idx == inner_axis_idx:
            rng.append((inner_axis_expr, inner_axis_expr, 1))
        else:
            ex = const_map[axis_idx]
            rng.append((ex, ex, 1))
    return subsets.Range(rng)


def _build_nested_carrier_range_subset(desc: data.Array, info: _Scan, scan_lo, scan_hi,
                                       inner_var: str, inner_start, inner_end) -> subsets.Range:
    """Multi-dim range subset for the carrier with a [scan_lo, scan_hi] range on the
    scan axis and [inner_start, inner_end] on the inner-var axis. Constants pass
    through verbatim.
    """
    inner_axis_idx, const_axes = _nested_axis_kinds(info, inner_var)
    const_map = {axis: expr for axis, expr in const_axes}
    rng = []
    for axis_idx in range(len(desc.shape)):
        if axis_idx == info.scan_axis:
            rng.append((scan_lo, scan_hi, 1))
        elif axis_idx == inner_axis_idx:
            rng.append((inner_start, inner_end, 1))
        else:
            ex = const_map[axis_idx]
            rng.append((ex, ex, 1))
    return subsets.Range(rng)


def _mutate_body_to_delta_buffer_nested(info: _Scan, delta_buf: str, inner_start: Any):
    """Like :func:`_mutate_body_to_delta_buffer` but routes the final write to a 2-D
    ``delta_buf[outer_iter - outer_start, inner_var - inner_start]``. The
    body_state is the innermost flat state; the outer loop variable is the
    parent of the parent-graph.
    """
    state = info.body_state
    tasklet = info.scan_update_tasklet
    inner_region = state.parent_graph
    outer_region = inner_region.parent_graph
    outer_var = outer_region.loop_variable
    inner_var = inner_region.loop_variable

    _disconnect_carry_chain(state, tasklet, info.carry_in_conn, info.carry_anchor)
    if info.literal_delta is not None:
        tasklet.code.as_string = f'{info.out_conn} = {info.literal_delta}'
    else:
        tasklet.code.as_string = f'{info.out_conn} = {info.delta_in_conn}'

    write_an = _find_carried_write_an(state, info.out_name)
    if write_an is None:
        return
    final_write_edges = list(state.in_edges(write_an))
    if len(final_write_edges) != 1:
        return
    final_edge = final_write_edges[0]
    state.remove_edge(final_edge)
    outer_idx = symbolic.simplify(symbolic.pystr_to_symbolic(outer_var) - info.iter_start)
    inner_idx = symbolic.simplify(symbolic.pystr_to_symbolic(inner_var) - inner_start)
    buf_an = state.add_write(delta_buf)
    state.add_edge(final_edge.src, final_edge.src_conn, buf_an, None,
                   mm.Memlet(data=delta_buf,
                             subset=subsets.Range([(outer_idx, outer_idx, 1), (inner_idx, inner_idx, 1)])))
    if state.degree(write_an) == 0:
        state.remove_node(write_an)


def _emit_scan_nested(state: SDFGState, sdfg: SDFG, info: _Scan, delta_buf: str, scan_buf: str,
                      trip: Any, inner_var: str, inner_start: Any, inner_end: Any):
    """Emit a single ``Scan`` libnode over the full ``[trip, inner_size]`` buffer with
    ``stride = inner_size * info.scan_stride``. With layout ``[outer, inner]`` (outer-axis
    first, contiguous along inner), elements ``buf[i, j]`` and ``buf[i + scan_stride, j]``
    are exactly ``inner_size * scan_stride`` positions apart in memory -- the libnode's
    residue-class semantics then run ``inner_size * scan_stride`` independent interleaved
    streams, one per ``(j, i mod scan_stride)`` pair. No Map wrap.
    """
    inner_size = symbolic.simplify(inner_end - inner_start + 1)
    delta_read = state.add_read(delta_buf)
    scan_write = state.add_write(scan_buf)
    node = Scan(name=f'{state.label}_op', op=info.op, exclusive=False)
    node.stride = symbolic.simplify(info.scan_stride * inner_size)
    state.add_node(node)
    state.add_edge(delta_read, None, node, _SCAN_IN,
                   mm.Memlet(data=delta_buf, subset=subsets.Range([(0, trip - 1, 1),
                                                                   (0, inner_size - 1, 1)])))
    state.add_edge(node, _SCAN_OUT, scan_write, None,
                   mm.Memlet(data=scan_buf, subset=subsets.Range([(0, trip - 1, 1),
                                                                  (0, inner_size - 1, 1)])))


def _emit_seed_add_nested(state: SDFGState, sdfg: SDFG, info: _Scan, scan_buf: str, trip: Any,
                          inner_var: str, inner_start: Any, inner_end: Any):
    """Emit a 2-D ``Map[(i, j)]`` over (outer iteration, inner iterator) that combines the
    pre-loop seed with the scanned delta: ``out[start + k_w + i, j] = seed[start + k_r, j]
    OP scan_buf[i, j]``.
    """
    map_i = f'_seed_i_{info.out_name}'
    map_j = f'_seed_j_{info.out_name}'
    while map_i in sdfg.symbols:
        map_i += '_'
    while map_j in sdfg.symbols or map_j == map_i:
        map_j += '_'

    out_desc = sdfg.arrays[info.out_name]
    seed_axis_expr = symbolic.simplify(info.iter_start + info.k_r)
    write_axis_expr = symbolic.simplify(info.iter_start + info.k_w + symbolic.pystr_to_symbolic(map_i))

    out_read = state.add_read(info.out_name)
    scan_read = state.add_read(scan_buf)
    out_write = state.add_write(info.out_name)
    code = _scan_op_expression(info.op)
    tasklet = state.add_tasklet(f'{state.label}_apply',
                                inputs={'_seed', '_delta'},
                                outputs={'_o'},
                                code=code)
    me, mx = state.add_map(state.label + '_map',
                           {map_i: subsets.Range([(0, trip - 1, 1)]),
                            map_j: subsets.Range([(inner_start, inner_end, 1)])})
    me.add_in_connector('IN_seed')
    me.add_in_connector('IN_scan')
    me.add_out_connector('OUT_seed')
    me.add_out_connector('OUT_scan')
    mx.add_in_connector('IN_o')
    mx.add_out_connector('OUT_o')
    inner_size = symbolic.simplify(inner_end - inner_start + 1)
    buf_j_local = symbolic.simplify(symbolic.pystr_to_symbolic(map_j) - inner_start)
    map_j_sym = symbolic.pystr_to_symbolic(map_j)
    # ``out`` carrier may have additional loop-invariant axes (e.g. ``[species,
    # jk, jl]`` with ``species`` constant); ``_build_nested_carrier_*`` weaves
    # the scan-axis / inner-var axis / constant axes into a full N-dim subset
    # consistent with ``out_desc.shape``.
    state.add_edge(out_read, None, me, 'IN_seed',
                   mm.Memlet(data=info.out_name,
                             subset=_build_nested_carrier_range_subset(
                                 out_desc, info, seed_axis_expr, seed_axis_expr, inner_var,
                                 inner_start, inner_end)))
    state.add_edge(scan_read, None, me, 'IN_scan',
                   mm.Memlet(data=scan_buf,
                             subset=subsets.Range([(0, trip - 1, 1),
                                                   (0, inner_size - 1, 1)])))
    state.add_edge(me, 'OUT_seed', tasklet, '_seed',
                   mm.Memlet(data=info.out_name,
                             subset=_build_nested_carrier_subset(
                                 out_desc, info, seed_axis_expr, inner_var, map_j_sym)))
    state.add_edge(me, 'OUT_scan', tasklet, '_delta',
                   mm.Memlet(data=scan_buf,
                             subset=subsets.Range([(map_i, map_i, 1), (buf_j_local, buf_j_local, 1)])))
    state.add_edge(tasklet, '_o', mx, 'IN_o',
                   mm.Memlet(data=info.out_name,
                             subset=_build_nested_carrier_subset(
                                 out_desc, info, write_axis_expr, inner_var, map_j_sym)))
    state.add_edge(mx, 'OUT_o', out_write, None,
                   mm.Memlet(data=info.out_name,
                             subset=_build_nested_carrier_range_subset(
                                 out_desc, info,
                                 symbolic.simplify(info.iter_start + info.k_w),
                                 symbolic.simplify(info.iter_start + info.k_w + trip - 1),
                                 inner_var, inner_start, inner_end)))


def _scan_op_expression(op) -> str:
    """Render a ``ScanOp`` as a Python tasklet expression combining the seed and
    the per-iteration delta. Used by :func:`_emit_seed_add_nested`.
    """
    if op == ScanOp.SUM:
        return '_o = _seed + _delta'
    if op == ScanOp.PRODUCT:
        return '_o = _seed * _delta'
    if op == ScanOp.MAX:
        return '_o = max(_seed, _delta)'
    if op == ScanOp.MIN:
        return '_o = min(_seed, _delta)'
    raise ValueError(f'Unsupported ScanOp: {op}')


def _can_emit_direct_write(info: _Scan, out_desc) -> bool:
    """Whether the scan output can be written straight into ``out`` -- skipping
    the seed-add Map -- via :func:`_emit_scan_with_init_direct`. Requires a 1-D
    output array, no other-axis indices, and ``scan_stride == 1`` so the write
    subset ``out[start + k_w : start + k_w + trip]`` is a contiguous range on
    the single axis the scan covers. Stride > 1 residue-class scans fall
    through to the general 3-stage path.
    """
    return len(out_desc.shape) == 1 and not info.other_indices and info.scan_stride == 1


def _emit_scan_with_init_direct(state: SDFGState, sdfg: SDFG, info: _Scan, delta_buf: str, trip: Any):
    """1-D direct-write path: Scan reads ``delta_buf``, takes ``out[start + k_r]``
    as its ``_scan_init`` seed, and writes the inclusive result directly into
    ``out[start + k_w : start + k_w + trip]`` -- no seed-add Map.

    The seed is materialised through a per-instance scalar transient so the
    state has a clean ``out`` read -> seed scalar -> scan-init connector path;
    a direct ``out``-to-init edge would force DaCe to view the same array as
    both read and write source within one state which the read/write subset
    pair satisfies but is harder to reason about for downstream cleanup.
    """
    out_desc = sdfg.arrays[info.out_name]
    seed_name, _ = sdfg.add_scalar(f'{_SEED_SCALAR_PREFIX}{info.out_name}', out_desc.dtype, transient=True,
                                   find_new_name=True)
    seed_axis_expr = symbolic.simplify(info.iter_start + info.k_r)
    write_start = symbolic.simplify(info.iter_start + info.k_w)
    write_end = symbolic.simplify(write_start + trip - 1)

    out_seed_read = state.add_read(info.out_name)
    seed_an = state.add_access(seed_name)
    delta_read = state.add_read(delta_buf)
    out_write = state.add_write(info.out_name)
    node = Scan(name=f'{state.label}_op', op=info.op, exclusive=False)
    node.add_in_connector(_SCAN_INIT)
    state.add_node(node)

    state.add_edge(out_seed_read, None, seed_an, None,
                   mm.Memlet(data=info.out_name, subset=subsets.Range([(seed_axis_expr, seed_axis_expr, 1)]),
                             other_subset=subsets.Range([(0, 0, 1)])))
    state.add_edge(seed_an, None, node, _SCAN_INIT,
                   mm.Memlet(data=seed_name, subset=subsets.Range([(0, 0, 1)])))
    state.add_edge(delta_read, None, node, _SCAN_IN,
                   mm.Memlet(data=delta_buf, subset=subsets.Range([(0, trip - 1, 1)])))
    state.add_edge(node, _SCAN_OUT, out_write, None,
                   mm.Memlet(data=info.out_name, subset=subsets.Range([(write_start, write_end, 1)])))


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

    # 2. The scan-update tasklet becomes a passthrough emitting the delta.
    #    v1/v2 (data delta): propagate ``delta_in_conn`` verbatim. v3 (literal
    #    delta, e.g. TSVC s242 ``a[i-1] + 0.5``): emit the literal directly;
    #    the carry connector is already severed and the tasklet now has zero
    #    in-edges, so the previous ``__in1 + literal`` body must NOT reference
    #    ``__in1`` (it was just removed).
    if info.literal_delta is not None:
        tasklet.code.as_string = f'{info.out_conn} = {info.literal_delta}'
    else:
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
    """Scan(delta_buf) -> scan_buf via the libnode. The libnode's ``stride``
    property carries ``info.scan_stride`` so residue-class scans (TSVC s1221,
    where the recurrence skips ``S`` positions) lift to the same libnode call."""
    r = state.add_read(delta_buf)
    w = state.add_write(scan_buf)
    node = Scan(name=f'{state.label}_op', op=info.op, exclusive=False)
    node.stride = info.scan_stride
    state.add_node(node)
    state.add_edge(r, None, node, _SCAN_IN, mm.Memlet(data=delta_buf, subset=subsets.Range([(0, trip - 1, 1)])))
    state.add_edge(node, _SCAN_OUT, w, None, mm.Memlet(data=scan_buf, subset=subsets.Range([(0, trip - 1, 1)])))


def _emit_seed_add(state: SDFGState, sdfg: SDFG, info: _Scan, scan_buf: str, trip: Any):
    """Map over ``_i`` writing ``out[start + k_w + _i, ...] = seed OP scan_buf[_i]``.

    For ``scan_stride == 1`` the seed is a single broadcast value
    ``out[start + k_r, ...]``. For ``scan_stride == S > 1`` (residue-class
    scan, e.g. TSVC s1221 ``b[i] = b[i-4] + a[i]``) the seed depends on the
    class index ``k = _i mod S`` and lives at
    ``out[start + k_r + k, ...]`` (the pre-loop value at the corresponding
    head of that class). The libnode has already run the ``S`` independent
    class scans into ``scan_buf``; this Map fans the ``S`` pre-loop seeds
    out by ``_i mod S``.
    """
    out_desc = sdfg.arrays[info.out_name]
    _i = symbolic.pystr_to_symbolic('_i')
    if info.scan_stride == 1:
        seed_axis_expr = symbolic.simplify(info.iter_start + info.k_r)
    else:
        import sympy
        class_idx = sympy.Mod(_i, info.scan_stride)
        seed_axis_expr = symbolic.simplify(info.iter_start + info.k_r + class_idx)
    seed_subset = _build_subset(out_desc, info.scan_axis, seed_axis_expr, info.other_indices)
    write_axis_expr = symbolic.simplify(info.iter_start + info.k_w) + _i
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


def _rewrite_scalar_carry(parent: ControlFlowRegion, loop: LoopRegion, info: _ScalarCarryScan,
                          sdfg: SDFG):
    """Rewrite a scalar-carry prefix-scan loop into three sibling states.

    1. **delta-build** -- Map over the iter range copying
       ``delta[i + delta_offset, ...]`` into a fresh 1-D transient
       ``_scan_in_<out>[i - iter_start]``.
    2. **scan** -- :class:`Scan` libnode reading ``_scan_in_<out>`` with init
       connector wired to ``acc[0]`` (the accumulator's value at loop entry),
       writing inclusive prefix into ``_scan_out_<out>``.
    3. **out-write** -- Map over the iter range copying
       ``_scan_out_<out>[i - iter_start]`` into ``out[i + out_offset, ...]``.

    If ``info.acc_used_post_loop`` is ``True``, a fourth state writes the
    last scan output ``_scan_out_<out>[trip - 1]`` back into ``acc[0]`` so
    downstream readers of ``acc`` see the final running value (matches the
    pre-rewrite sequential post-loop scalar value).
    """
    import dace
    out_desc = sdfg.arrays[info.out_name]
    delta_desc = sdfg.arrays[info.delta_name]
    acc_desc = sdfg.arrays[info.acc_name]
    trip = symbolic.simplify(info.iter_end - info.iter_start + 1)

    delta_buf, _ = sdfg.add_array(f'{_DELTA_BUF_PREFIX}{info.out_name}', [trip], out_desc.dtype,
                                  transient=True, find_new_name=True)
    scan_buf, _ = sdfg.add_array(f'{_SCAN_BUF_PREFIX}{info.out_name}', [trip], out_desc.dtype,
                                 transient=True, find_new_name=True)

    in_edges = list(parent.in_edges(loop))
    out_edges = list(parent.out_edges(loop))
    is_start = (parent.start_block is loop)
    s_build = parent.add_state(loop.label + '_scan_build')
    s_scan = parent.add_state(loop.label + '_scan_op')
    s_write = parent.add_state(loop.label + '_scan_write')
    parent.add_edge(s_build, s_scan, dace.InterstateEdge())
    parent.add_edge(s_scan, s_write, dace.InterstateEdge())
    tail_state = s_write
    if info.acc_used_post_loop:
        s_acc_post = parent.add_state(loop.label + '_scan_acc_post')
        parent.add_edge(s_write, s_acc_post, dace.InterstateEdge())
        tail_state = s_acc_post
    # Reroute predecessors of the loop to ``s_build`` and successors of the
    # loop from ``tail_state`` BEFORE removing the loop node; otherwise the
    # new states would be unreachable / dangling.
    #
    # ``e.data`` is the original :class:`InterstateEdge` object -- passing it
    # through ``add_edge`` preserves any iedge assignments (the symbol
    # bindings the pipeline cascades onto loop boundary edges) verbatim.
    # Conditions are intentionally not duplicated -- canonicalize pipeline
    # callers run after structural cleanup, which strips any non-trivial
    # condition off loop-boundary iedges.
    for e in in_edges:
        parent.remove_edge(e)
        parent.add_edge(e.src, s_build, e.data)
    for e in out_edges:
        parent.remove_edge(e)
        parent.add_edge(tail_state, e.dst, e.data)

    _emit_scalar_carry_delta_build(s_build, sdfg, info, delta_buf)
    _emit_scalar_carry_scan(s_scan, sdfg, info, delta_buf, scan_buf, trip)
    _emit_scalar_carry_out_write(s_write, sdfg, info, scan_buf, trip)
    if info.acc_used_post_loop:
        _emit_scalar_carry_acc_post(tail_state, sdfg, info, scan_buf, trip)

    # Remove the original loop now that its semantics are captured by the new states.
    parent.remove_node(loop)
    if is_start:
        parent.start_block = parent.node_id(s_build)
    sdfg.reset_cfg_list()


def _emit_scalar_carry_delta_build(state: SDFGState, sdfg: SDFG, info: _ScalarCarryScan,
                                    delta_buf: str):
    """Map ``_i`` over ``[0, trip)`` copying
    ``delta[iter_start + _i + delta_offset, ...]`` into ``delta_buf[_i]``."""
    trip = symbolic.simplify(info.iter_end - info.iter_start + 1)
    delta_desc = sdfg.arrays[info.delta_name]
    _i = symbolic.pystr_to_symbolic('_i')
    delta_axis_expr = symbolic.simplify(info.iter_start + _i + info.delta_offset)
    delta_subset = _build_subset(delta_desc, info.delta_scan_axis, delta_axis_expr,
                                 info.delta_other_indices)
    state.add_mapped_tasklet(
        f'{state.label}_tasklet',
        {'_i': f'0:{trip}'},
        {'__d': mm.Memlet(data=info.delta_name, subset=delta_subset)},
        '__o = __d',
        {'__o': mm.Memlet(data=delta_buf, subset=subsets.Range([(_i, _i, 1)]))},
        external_edges=True,
    )


def _emit_scalar_carry_scan(state: SDFGState, sdfg: SDFG, info: _ScalarCarryScan,
                             delta_buf: str, scan_buf: str, trip: Any):
    """Run the ``Scan`` libnode with the accumulator's pre-loop value wired into
    the optional ``_scan_init`` connector. Inclusive semantics +
    ``acc[0]`` seed make ``scan_buf[i] = acc_initial OP delta_buf[0]
    OP ... OP delta_buf[i]``.

    The seed is read into a per-instance scalar transient to keep the libnode's
    init-connector edge a clean Scalar memlet (the same pattern
    :func:`_emit_scan_with_init_direct` uses).
    """
    acc_desc = sdfg.arrays[info.acc_name]
    seed_name, _ = sdfg.add_scalar(f'{_SEED_SCALAR_PREFIX}{info.out_name}', acc_desc.dtype,
                                   transient=True, find_new_name=True)
    acc_read = state.add_read(info.acc_name)
    seed_an = state.add_access(seed_name)
    delta_read = state.add_read(delta_buf)
    scan_write = state.add_write(scan_buf)

    node = Scan(name=f'{state.label}_op', op=info.op, exclusive=False)
    node.add_in_connector(_SCAN_INIT)
    state.add_node(node)

    state.add_edge(acc_read, None, seed_an, None,
                   mm.Memlet(data=info.acc_name, subset=subsets.Range([(0, 0, 1)]),
                             other_subset=subsets.Range([(0, 0, 1)])))
    state.add_edge(seed_an, None, node, _SCAN_INIT,
                   mm.Memlet(data=seed_name, subset=subsets.Range([(0, 0, 1)])))
    state.add_edge(delta_read, None, node, _SCAN_IN,
                   mm.Memlet(data=delta_buf, subset=subsets.Range([(0, trip - 1, 1)])))
    state.add_edge(node, _SCAN_OUT, scan_write, None,
                   mm.Memlet(data=scan_buf, subset=subsets.Range([(0, trip - 1, 1)])))


def _emit_scalar_carry_out_write(state: SDFGState, sdfg: SDFG, info: _ScalarCarryScan,
                                  scan_buf: str, trip: Any):
    """Map ``_i`` over ``[0, trip)`` copying ``scan_buf[_i]`` into
    ``out[iter_start + _i + out_offset, ...]``."""
    out_desc = sdfg.arrays[info.out_name]
    _i = symbolic.pystr_to_symbolic('_i')
    out_axis_expr = symbolic.simplify(info.iter_start + _i + info.out_offset)
    out_subset = _build_subset(out_desc, info.out_scan_axis, out_axis_expr, info.out_other_indices)
    state.add_mapped_tasklet(
        f'{state.label}_tasklet',
        {'_i': f'0:{trip}'},
        {'__v': mm.Memlet(data=scan_buf, subset=subsets.Range([(_i, _i, 1)]))},
        '__o = __v',
        {'__o': mm.Memlet(data=info.out_name, subset=out_subset)},
        external_edges=True,
    )


def _emit_scalar_carry_acc_post(state: SDFGState, sdfg: SDFG, info: _ScalarCarryScan,
                                 scan_buf: str, trip: Any):
    """Copy ``scan_buf[trip - 1]`` into ``acc[0]`` so downstream readers of
    ``acc`` see the post-loop running value. Single-tasklet state, no Map (it's
    a scalar move)."""
    scan_read = state.add_read(scan_buf)
    acc_write = state.add_write(info.acc_name)
    t = state.add_tasklet(name=f'{state.label}_writeback', inputs={'__v'}, outputs={'__o'},
                          code='__o = __v', language=dtypes.Language.Python)
    state.add_edge(scan_read, None, t, '__v',
                   mm.Memlet(data=scan_buf, subset=subsets.Range([(trip - 1, trip - 1, 1)])))
    state.add_edge(t, '__o', acc_write, None,
                   mm.Memlet(data=info.acc_name, subset=subsets.Range([(0, 0, 1)])))
