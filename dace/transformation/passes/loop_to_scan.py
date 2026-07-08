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
from typing import Any, Dict, List, NamedTuple, Optional

import sympy

from dace import SDFG, data, dtypes, properties, subsets, symbolic
from dace import memlet as mm
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis

# Re-export the supported associative ops via :class:`ScanOp`; the matcher recognises
# the same four ops the libnode expansions cover.
from dace.libraries.standard.nodes.scan import (Scan, ScanOp, INPUT_CONNECTOR_NAME, OUTPUT_CONNECTOR_NAME,
                                                INIT_CONNECTOR_NAME)

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
    # Direction: ``+1`` for the canonical forward iteration (write at
    # ``loop_var + k_w``); ``-1`` for the reverse iteration (write at
    # ``k_w - loop_var``) -- the cloudsc ``for_1079`` shape after
    # :class:`NormalizeNegativeStride` normalises the loop. With ``coef == -1``,
    # ``k_w`` / ``k_r`` hold the array constants (i.e. the array index visited
    # at the first iteration), and the rewrite emits seed-add Map subscripts
    # ``k_w - i`` instead of ``iter_start + k_w + i``.
    coef: int = 1


class _CompositeBodyScan(NamedTuple):
    """A composite-body scan match (cloudsc ``for_1133`` shape).

    The outer body has a carry-copy state writing ``carrier[*, loop_var] =
    carrier[*, loop_var - 1]`` via an identity-assign tasklet, plus one or more
    sibling accumulate states (possibly inside nested LoopRegions) writing
    ``carrier[*, loop_var] += per-iter-term``. Net recurrence:
    ``carrier[*, loop_var] = carrier[*, loop_var - 1] + sum(terms)`` -- a
    standard prefix scan, but expressed as multiple body writes rather than a
    single binop tasklet. The accumulate inner loops stay as loops in the
    rewrite -- only the outer scan recurrence gets lifted.

    :param out_name: The carrier array name.
    :param scan_axis: Carrier axis along which the prefix scan runs.
    :param other_indices: Non-scan-axis carrier indices ``(axis, expr)`` --
        loop-invariant constants or enclosing-scope symbols (NOT the inner
        loop's iterator -- composite bodies typically write the carrier at a
        per-(inner-iter, scan-axis) slot, e.g. ``pfsqif[jl, jk]``).
    :param iter_start: Loop start (inclusive).
    :param iter_end: Loop end (inclusive).
    :param carry_copy_state: The state whose identity-assign tasklet writes
        ``carrier[*, loop_var] = carrier[*, loop_var - 1]``. The rewrite mutates
        this tasklet to emit ``0`` and severs the carrier-read input.
    :param carry_copy_tasklet: The identity-assign tasklet to mutate.
    :param carry_copy_carry_anchor: The AccessNode the carry input enters; used
        for orphan cleanup after the rewrite severs the carrier-read chain.
    :param carry_copy_in_conn: The tasklet's carrier-read input connector.
    :param carry_copy_out_conn: The tasklet's output connector.
    """
    out_name: str
    scan_axis: int
    other_indices: List[Any]
    iter_start: Any
    iter_end: Any
    carry_copy_state: SDFGState
    carry_copy_tasklet: nodes.Tasklet
    carry_copy_carry_anchor: nodes.AccessNode
    carry_copy_in_conn: str
    carry_copy_out_conn: str
    accumulate_states: List[SDFGState] = []


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

    interchange_carry_with_map = properties.Property(
        dtype=bool,
        default=False,
        desc=("If True, also detect the Map-wrapped carry shape (outer carry-axis "
              "``LoopRegion`` whose body is a single state containing exactly one "
              "Map -- the post-``LoopToMap`` form of the cloudsc ``for_1133`` "
              "kernel) and lift it via loop interchange: the outer ``LoopRegion`` "
              "becomes the new outer parallel ``Map``, and a per-column 1-D "
              "``Scan`` libnode replaces the carry loop inside. Off by default; "
              "opt in for A/B perf comparison against the unchanged "
              "post-``LoopToMap`` shape (outer carry-``LoopRegion`` + inner "
              "parallel Map)."),
    )

    lift_nested_scan = properties.Property(
        dtype=bool,
        default=False,
        desc=("Controls the NESTED (vector) scan shape ``for j: for i: a[j,i] = "
              "a[j-1,i] OP b[j,i]`` -- a carry loop ``j`` wrapping a data-parallel "
              "inner loop ``i``. With the default ``False`` the lift is REFUSED when "
              "the inner loop is parallelizable: the carry loop is left as a plain "
              "sequential ``LoopRegion`` and the inner loop is mapped by ``LoopToMap`` "
              "(``for j(seq): map[i]``) -- a contiguous unit-stride map with no ``Scan`` "
              "libnode, the preferred shape after ``LoopStridePermutation`` interchanges "
              "the unit-stride axis innermost. Set ``True`` to still lift the vector "
              "scan (a ``Scan`` with a Map over the inner axis). Non-parallelizable "
              "inner loops are lifted regardless (there is no map to keep)."),
    )

    def __init__(self, interchange_carry_with_map: bool = False, lift_nested_scan: bool = False):
        super().__init__()
        self.interchange_carry_with_map = interchange_carry_with_map
        self.lift_nested_scan = lift_nested_scan

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Descriptors | ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    @staticmethod
    def _inner_loop_parallelizable(inner_loop: LoopRegion, sdfg: SDFG) -> bool:
        """``True`` iff the nested scan's inner loop is a DOALL loop ``LoopToMap``
        would parallelize -- i.e. there is a map to keep instead of lifting."""
        from dace.transformation.interstate.loop_to_map import LoopToMap
        try:
            return LoopToMap.can_be_applied_to(sdfg, loop=inner_loop)
        except Exception:  # noqa: BLE001 -- oracle refuses exotic shapes -> not a keepable map
            return False

    def _specialize_scan_under_stride_guard(self, parent: ControlFlowRegion, loop: LoopRegion, guard: str, sdfg: SDFG):
        """Replace a symbolic-stride scan ``loop`` with ``if (guard) { scan } else
        { original sequential loop }`` via :func:`specialize_loop_under_condition`.

        The true-branch clone is re-matched and lifted to the ``Scan`` pipeline;
        the else-branch clone is pinned sequential (``LoopToMap`` / a re-run of
        this pass leave it alone). ``guard`` is the ``stride >= 1`` predicate from
        :func:`_symbolic_stride_guard` under which the residue-class scan is valid.
        """
        from dace.transformation.passes.loop_specialization import specialize_loop_under_condition

        def _lift(par_loop: LoopRegion, par_region: ControlFlowRegion, _owner: SDFG):
            par_infos = _match_all(par_loop, sdfg)
            if par_infos and not self.lift_nested_scan:
                par_infos = [
                    info for info in par_infos
                    if not (info.inner_loop is not None and self._inner_loop_parallelizable(info.inner_loop, sdfg))
                ]
            for info in par_infos:
                _rewrite(par_region, par_loop, info, sdfg)

        specialize_loop_under_condition(loop, guard, _lift, sdfg)

    def apply_pass(self, sdfg: SDFG, _pipeline_results) -> Optional[int]:
        # Whole-SDFG preprocess: strip frontend ``__out = __inp`` copy tasklets so the
        # matcher sees the bare ``out[i+1] = out[i] + delta[i]`` shape. Without this the
        # carry hides behind an ``assign_NN`` copy node on the write side.
        from dace.transformation.dataflow.trivial_tasklet_elimination import TrivialTaskletElimination
        from dace.transformation.dataflow.wcr_conversion import WCRToAugAssign
        from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
        # Normalise reductions written as WCR edges back to in-body augmented
        # assignment so the matcher sees a uniform tasklet shape. No-op on SDFGs
        # whose pre-existing reductions are already in augassign form (the
        # common case for canonicalised Fortran frontends).
        PatternMatchAndApplyRepeated([WCRToAugAssign()]).apply_pass(sdfg, {})
        PatternMatchAndApplyRepeated([TrivialTaskletElimination()]).apply_pass(sdfg, {})

        # NOTE: D4 (CleanAccessNode + CleanTasklet) is deliberately NOT applied
        # here. LoopToScan's matcher already handles the frontend's scalar-
        # slice intermediates via ``_chase_forward_to_accum`` and friends;
        # the existing WCR/TrivialTasklet preprocess above is sufficient.
        # Running the clean folds in addition (in either order vs WCR) is
        # redundant work and previously regressed the
        # ``for_1133_shape_reverse_engineered`` case by stripping
        # intermediates the matcher relies on.

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
        # Optional first pass: interchange the Map-wrapped carry shape.
        # Done up-front so the carry loop runs sequentially per-thread INSIDE
        # the parallel Map (no buffers, no Scan libnode, just a per-thread
        # sequential ``for jk`` reading/writing global memory directly).
        # We track the relocated loops by id so the regular matcher pass below
        # leaves them alone (they have already been rewritten to their final
        # sequential-per-thread form).
        interchanged_loop_ids = set()
        if self.interchange_carry_with_map:
            for loop, parent in list(_collect_loops(sdfg)):
                shape = _detect_carry_loop_with_inner_map(loop, sdfg)
                if shape is None:
                    continue
                relocated = _rewrite_interchange_carry_with_map(shape, sdfg)
                if relocated is None:
                    continue
                if isinstance(relocated, LoopRegion):
                    interchanged_loop_ids.add(id(relocated))
                count += 1

        for loop, parent in _collect_loops(sdfg):
            if id(loop) in interchanged_loop_ids:
                continue
            if loop.pinned_sequential:
                # A deliberate sequential fallback spliced in by a prior stride
                # specialization (below); re-matching it would recurse into
                # another if/else. Leave it as the original sequential loop.
                continue
            infos = _match_all(loop, sdfg)
            # NESTED (vector) scan default: when the matched scan wraps a
            # data-parallel inner loop, prefer to KEEP THE MAP INSIDE -- leave
            # this carry loop sequential and let ``LoopToMap`` map the inner
            # loop (``for j(seq): map[i]``). That avoids a ``Scan`` over a
            # strided apply and is the shape ``LoopStridePermutation`` sets up
            # by moving the unit-stride parallel axis innermost. Lifting the
            # vector scan is opt-in via ``lift_nested_scan``. A non-parallel
            # inner loop has no map to keep, so it is lifted regardless.
            if infos and not self.lift_nested_scan:
                infos = [
                    info for info in infos
                    if not (info.inner_loop is not None and self._inner_loop_parallelizable(info.inner_loop, sdfg))
                ]
            if infos:
                guard = _symbolic_stride_guard(infos)
                if guard is not None:
                    # At least one matched scan has a symbolic stride whose sign
                    # is unproven. The residue-class decomposition is valid only
                    # for stride >= 1, so specialize the loop into
                    # ``if stride >= 1: <scan pipeline> else: <sequential loop>``
                    # rather than lift unconditionally: a violating runtime value
                    # (stride 0 -> a degenerate in-place update) degrades to the
                    # sequential fallback and still computes correctly.
                    self._specialize_scan_under_stride_guard(parent, loop, guard, sdfg)
                    count += 1
                    continue
                for info in infos:
                    _rewrite(parent, loop, info, sdfg)
                    count += 1
                continue
            # The COMPOSITE-BODY shape (cloudsc ``for_1133``): outer body has
            # a carry-copy state + sibling accumulate states writing the same
            # carrier. The rewrite redirects body writes to a fresh
            # ``delta_buf[trip, inner_size]`` (inserting fresh delta_buf
            # AccessNodes at the chain endpoints, leaving intermediate
            # transients untouched) and emits the standard
            # ``Scan`` + seed-add via the nested-scan helpers.
            comp = _match_composite_body(loop, sdfg)
            if comp is not None:
                if _rewrite_composite_body(parent, loop, comp, sdfg):
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
    cond_blocks = [b for b in blocks if isinstance(b, ConditionalBlock)]
    states = [b for b in blocks if isinstance(b, SDFGState)]
    others = [b for b in blocks if not isinstance(b, (LoopRegion, SDFGState, ConditionalBlock))]
    if others:
        return []
    content_states = [s for s in states if len(s.nodes()) > 0]
    candidates = []
    # Flat candidates at the outer body level: every content state is itself
    # a possible scan-update host (v5 case).
    if not inner_loop_regions and not cond_blocks:
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
        # ``ConditionalBlock`` branches: each branch's content state is also a
        # candidate (the cloudsc-like conditional-carry shape -- if/else
        # whose branches each implement a scan-update on the same carrier).
        # The matcher then picks the branch that has the scan-update tasklet;
        # the rewrite step also mutates any sibling branch's writes (see
        # ``_mutate_sibling_branches_to_zero_delta`` -- emitted post-match).
        for cb in cond_blocks:
            for cond, branch in cb.branches:
                branch_state, deeper = _descend_to_content_state(branch)
                if branch_state is None or deeper is not None:
                    continue
                candidates.append((branch_state, None))
    return candidates


def _fuse_body_states(loop: LoopRegion) -> int:
    """Body-local state fusion: merge adjacent SDFGStates inside ``loop`` when
    the existing ``StateFusionExtended`` transformation would accept the fuse.

    Whole-SDFG ``StateFusion`` doesn't reach into LoopRegion bodies via
    ``match_patterns`` (it's a top-level ``MultiStateTransformation``); this
    helper does the same merge directly on the body. Required for the v5 case
    -- the cloudsc ``pfsqrf`` inner loop's body has two content states joined
    by a no-op iedge that the single-content-state matcher otherwise refuses.

    The safety check delegates to ``StateFusionExtended.can_be_applied``
    rather than re-implementing a hand-rolled subset of its analysis. The
    extended variant already understands RAW / WAW hazards between the two
    states' read/write sets and refuses when the merge would race a sibling
    carried scalar (TSVC s252 ``t = s`` carry after ``a[i] = s + t`` -- the
    s1-reads-``t`` / s2-writes-``t`` pattern collapses into one state with
    no ordering edge between the two ``t`` AccessNodes, and codegen
    schedules the write before the read, breaking the carry).

    :param loop: The LoopRegion to mutate in place.
    :returns: The number of fusions performed.
    """
    from dace.transformation.interstate.state_fusion_with_happens_before import StateFusionExtended
    owner_sdfg = loop.sdfg
    n_fused = 0
    while True:
        blocks = loop.nodes()
        if not all(isinstance(b, SDFGState) for b in blocks):
            return n_fused
        # Find a fusion candidate: a pair (s1, s2) with one s1->s2 edge, both
        # states non-empty, and the ``StateFusionExtended`` matcher's checks
        # all green. The matcher takes the parent CFG region as ``graph`` and
        # the owning SDFG separately.
        cand = None
        for s1 in blocks:
            outs = loop.out_edges(s1)
            if len(outs) != 1:
                continue
            e = outs[0]
            if e.data is None:
                continue
            s2 = e.dst
            if not isinstance(s2, SDFGState):
                continue
            if len(loop.in_edges(s2)) != 1:
                continue
            if s1.is_empty() or s2.is_empty():
                continue
            xform = StateFusionExtended()
            xform.first_state = s1
            xform.second_state = s2
            try:
                accepts = xform.can_be_applied(loop, expr_index=0, sdfg=owner_sdfg)
            except Exception:
                accepts = False
            if not accepts:
                continue
            cand = (s1, s2, e)
            break
        if cand is None:
            return n_fused
        s1, s2, iedge = cand
        _merge_state_into(s1, s2)
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
        if (isinstance(n, nodes.AccessNode) and n.data in s1_writes and s2.out_degree(n) > 0 and s2.in_degree(n) == 0):
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
            # Try the multi-slot path first (same carrier, distinct constant-only
            # slots). If it returns a non-empty list, treat each slot as its own
            # ``_Scan`` -- one ``Scan`` libnode per slot in the rewrite.
            multi = _match_multi_slot(loop, sdfg, state, out_name, start, end)
            if multi:
                for info in multi:
                    if inner_loop is not None:
                        info = info._replace(inner_loop=inner_loop)
                        if not _other_indices_match_inner(info.other_indices, inner_loop.loop_variable):
                            cand_failed = True
                            break
                    cand_matched.append(info)
                if cand_failed:
                    break
                continue
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

    # A matched carrier written by a sibling block of the loop (outside the loop
    # body but in its containing region) is the scan's SEED -- the pre-loop value
    # ``out[iter_start + k_r, ...]`` the seed-add reads back.
    #
    # For a FORWARD FLAT 1-D scan (``coef == 1``, no inner/vector axis, no
    # non-scan ``other_indices``) that seed is consumed correctly with no extra
    # work: :func:`_emit_seed_add` reads the seed slot straight from the LIVE
    # array after the sibling state runs, and the scan writes only
    # ``out[iter_start + k_w + _i]`` for ``_i >= 0`` -- strictly ahead of the read
    # slot ``iter_start + k_r`` (``k_w > k_r``), so the seed slot is never
    # overwritten. So an in-kernel seed like ``a[0] = x[0]`` before
    # ``for i: a[i] = a[i-1] + x[i]`` (TSVC ``fission_dep_then_indep`` /
    # ``fission_dep_const_offset``) lifts safely.
    #
    # The NESTED / per-row 2-D seed shape (``flux[i, 0] = fall[i, 0]`` before the
    # inner ``for k`` prefix scan; Thomas's ``x[i, K-1] = dp[K-1]`` backward
    # sweep) is NOT yet seed-captured into the carry buffer, so it stays refused.
    seed_captured = all(s.coef == 1 and s.inner_loop is None and not s.other_indices for s in matched)
    parent = loop.parent_graph
    if parent is not None and not seed_captured:
        sibling_blocks = [b for b in parent.nodes() if b is not loop]
        for sb in sibling_blocks:
            states = list(sb.all_states()) if isinstance(sb, (ControlFlowRegion, LoopRegion)) else [sb]
            for st in states:
                if not isinstance(st, SDFGState):
                    continue
                for node in st.data_nodes():
                    if st.in_degree(node) == 0:
                        continue
                    if node.data in carrier_set:
                        return []

    # Refuse when the body reads the carrier at more than one distinct subset
    # (multi-step recurrence ``b[i] = b[i+1] + b[i+2] + a[i]``, or a stencil
    # whose in-place carrier is read at several offsets -- seidel_2d's
    # ``A[i,j] = (A[i-1,j-1] + ... + A[i,j-1] + A[i,j+1] + ... ) / 9``). The
    # scan rewrite emits a single carry buffer for the one-step recurrence the
    # matcher accepted; any OTHER carrier read at a different offset is NOT
    # routed through that buffer and remains a direct array load. For an
    # in-place carrier (read AND written) those direct loads see the wrong
    # values once the scan reorders iterations.
    #
    # The distinct read subsets must be AGGREGATED ACROSS ALL AccessNodes of
    # the carrier, not counted per node: after ``SplitTasklets`` a 9-point
    # stencil reads the carrier through nine SEPARATE single-edge AccessNodes,
    # so a per-node tally sees ``1`` at each and never trips. Aggregating
    # restores the documented "the body reads the carrier at more than one
    # distinct subset" intent. A legitimate scan reads its carrier at exactly
    # the one carry offset, so its aggregate count is 1 and it still matches.
    carrier_reads: Dict[str, set] = {name: set() for name in carrier_set}
    for st in loop.all_states():
        for n in st.data_nodes():
            if n.data not in carrier_set:
                continue
            for e in st.out_edges(n):
                if e.data is None or e.data.subset is None:
                    continue
                carrier_reads[n.data].add(str(e.data.subset))
    if any(len(subs) > 1 for subs in carrier_reads.values()):
        return []

    # Refuse the multi-slot shape: several matched scan recurrences on the SAME
    # carrier array at distinct constant slots (e.g. ``acc[0, i]``, ``acc[1, i]``,
    # ... in one body -- both the flat shape and the nested ``zvqx[r, jk, jl]``
    # cloudsc for_430 shape). The multi-slot rewrite chains a per-slot ``_rewrite``
    # over the shared loop and mis-captures each slot's external seed
    # (``acc[r, start]``), reading an uninitialised buffer -> numerically wrong
    # (verified maxdiff ~0.85 vs the sequential oracle for the nested case). Leave
    # the loop sequential -- for the flat shape ``LoopFission`` then splits the
    # slots into independent single-slot loops, each lifted correctly by the
    # single-carrier path. Keyed on the count of matched infos per array (not raw
    # write subsets) so a single scan with an extra side-effect write to the
    # carrier (the v5 fused-body shape -- one matched info) is NOT refused. The
    # multi-*array* case (one info per distinct array, e.g. cloudsc pfsqrf's five
    # different carriers) is likewise untouched.
    infos_per_array: Dict[str, int] = {}
    for s in matched:
        infos_per_array[s.out_name] = infos_per_array.get(s.out_name, 0) + 1
    if any(c > 1 for c in infos_per_array.values()):
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


def _match_one_carrier(loop: LoopRegion, sdfg: SDFG, state: SDFGState, out_name: str, iter_start: Any,
                       iter_end: Any) -> Optional[_Scan]:
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
        write_axis, k_w, write_others, write_coef = _classify_subset(write_edge.data.subset, loop.loop_variable)
        if write_axis is None:
            continue
        cand = _find_scan_update_tasklet(state, sdfg, out_name, loop.loop_variable, write_axis, write_others, k_w,
                                         write_coef)
        if cand is None:
            continue
        tasklet, carry_edge, delta_edge, op, carry_anchor, scan_stride, literal_delta = cand
        out_edges_t = [e for e in state.out_edges(tasklet) if e.data is not None and not e.data.is_empty()]
        if len(out_edges_t) != 1:
            continue
        k_r = symbolic.simplify(k_w - scan_stride if write_coef == 1 else k_w + scan_stride)
        candidates.append(
            _Scan(
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
                coef=write_coef,
            ))
    # Single match: the v1-v5 case -- one scan-update tasklet writing the carrier.
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        return None
    # Multi-slot case: each candidate writes the SAME array at a DISTINCT
    # constant slot (different ``other_indices``) -- e.g. cloudsc ``for_430``
    # zvqx[0..N-1] where each slot is its own prefix-sum on the level axis.
    # Accept the set if every candidate has matching scan-shape (op, axis,
    # offsets, stride) and each writes a unique constant-slot configuration
    # that's pairwise NON-overlapping. Return the FIRST candidate; the others
    # are reported separately via :func:`_match_multi_slot` so the caller can
    # emit one ``Scan`` libnode per slot.
    if _multi_slot_compatible(candidates):
        return candidates[0]
    # Otherwise multiple distinct recurrences on the same carrier (e.g.
    # different scan offsets); refuse since we cannot pick between them.
    return None


def _multi_slot_compatible(candidates) -> bool:
    """Return ``True`` if all candidates have the same scan shape but distinct
    constant-only ``other_indices`` (each candidate is its own independent slot
    on the same carrier). The shape contract: same ``op``, ``scan_axis``,
    ``k_w``, ``k_r``, ``scan_stride`` -- and ``other_indices`` differs in the
    constant value(s) (axis indices being identical).
    """
    if len(candidates) < 2:
        return False
    ref = candidates[0]
    seen_keys = set()
    for c in candidates:
        if c.op != ref.op or c.scan_axis != ref.scan_axis:
            return False
        if c.k_w != ref.k_w or c.k_r != ref.k_r or c.scan_stride != ref.scan_stride:
            return False
        # ``other_indices`` axes must be the same set; the constants may differ.
        if {a for a, _ in c.other_indices} != {a for a, _ in ref.other_indices}:
            return False
        key = tuple(sorted((a, str(e)) for a, e in c.other_indices))
        if key in seen_keys:
            return False
        seen_keys.add(key)
    return True


def _match_multi_slot(loop: LoopRegion, sdfg: SDFG, state: SDFGState, out_name: str, iter_start: Any,
                      iter_end: Any) -> List[_Scan]:
    """Same matching as :func:`_match_one_carrier` but returns ALL slot candidates
    when the carrier is written at multiple distinct constant slots. Returns an
    empty list when the multi-slot shape doesn't apply (caller falls back to the
    single-match path).
    """
    candidates = []
    for write_edge in _iter_write_edges(state, out_name):
        if write_edge.data is None or write_edge.data.subset is None:
            continue
        write_axis, k_w, write_others, write_coef = _classify_subset(write_edge.data.subset, loop.loop_variable)
        if write_axis is None:
            continue
        cand = _find_scan_update_tasklet(state, sdfg, out_name, loop.loop_variable, write_axis, write_others, k_w,
                                         write_coef)
        if cand is None:
            continue
        tasklet, carry_edge, delta_edge, op, carry_anchor, scan_stride, literal_delta = cand
        out_edges_t = [e for e in state.out_edges(tasklet) if e.data is not None and not e.data.is_empty()]
        if len(out_edges_t) != 1:
            continue
        k_r = symbolic.simplify(k_w - scan_stride if write_coef == 1 else k_w + scan_stride)
        candidates.append(
            _Scan(
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
                coef=write_coef,
            ))
    if _multi_slot_compatible(candidates):
        return candidates
    return []


def _match(loop: LoopRegion, sdfg: SDFG) -> Optional[_Scan]:
    """Single-carrier match (kept for callers that handled one carry at a time)."""
    infos = _match_all(loop, sdfg)
    if len(infos) != 1:
        return None
    return infos[0]


# ---------------------------------------------------------------------------
# Loop-interchange path for the Map-wrapped carry shape (cloudsc for_1133).
# Gated by the ``LoopToScan.interchange_carry_with_map`` Property.
# ---------------------------------------------------------------------------


class _CarryMapShape(NamedTuple):
    """Description of an outer carry-``LoopRegion`` whose body is a single
    state containing exactly one parallel ``Map``, the post-``LoopToMap``
    shape of the cloudsc ``for_1133`` kernel."""

    loop: LoopRegion  # outer carry-axis LoopRegion (e.g. jk in 1:KLEV)
    parent: ControlFlowRegion  # parent of ``loop`` (insertion site for the rewrite)
    body_state: SDFGState  # the single state inside ``loop``
    map_entry: nodes.MapEntry  # the inner Map's entry (e.g. jl in 0:KLON)
    map_exit: nodes.MapExit
    nsdfg: nodes.NestedSDFG  # the NestedSDFG node inside the Map's scope
    inner_state: SDFGState  # the single state inside ``nsdfg.sdfg``
    inner_scan: _Scan  # the scan-update info derived from inner memlets


def _detect_carry_loop_with_inner_map(loop: LoopRegion, sdfg: SDFG) -> Optional[_CarryMapShape]:
    """Recognise the Map-wrapped carry shape.

    Required structure:
      - ``loop`` is a ``LoopRegion`` with a unit-stride loop variable (the
        carry axis, e.g. ``jk``).
      - ``loop`` body contains exactly one ``SDFGState`` with computation.
      - That state contains exactly one ``MapEntry``/``MapExit`` pair (the
        parallel axis, e.g. ``jl``) and exactly one ``NestedSDFG`` whose
        outer connections route through that Map.
      - The NestedSDFG body is a single ``SDFGState`` whose memlets refer
        to the carry axis via ``symbol_mapping``.
      - The inner state matches the single-carrier scan-update pattern
        (re-uses :func:`_match_one_carrier`).

    Returns the shape description, or ``None`` if any condition fails.
    """
    # 1. Unit-stride loop variable.
    if not loop.loop_variable:
        return None
    stride = loop_analysis.get_loop_stride(loop)
    if stride is None or stride != 1:
        return None
    iter_start = loop_analysis.get_init_assignment(loop)
    iter_end = loop_analysis.get_loop_end(loop)
    if iter_start is None or iter_end is None:
        return None
    parent = loop.parent_graph
    if parent is None:
        return None
    # 2. Single content state in the body.
    body_blocks = [b for b in loop.nodes() if isinstance(b, SDFGState)]
    if len(body_blocks) != 1 or any(not isinstance(b, SDFGState) for b in loop.nodes()):
        return None
    body_state: SDFGState = body_blocks[0]
    # 3. Exactly one MapEntry, one matching MapExit, and one NestedSDFG.
    map_entries = [n for n in body_state.nodes() if isinstance(n, nodes.MapEntry)]
    map_exits = [n for n in body_state.nodes() if isinstance(n, nodes.MapExit)]
    nsdfgs = [n for n in body_state.nodes() if isinstance(n, nodes.NestedSDFG)]
    if len(map_entries) != 1 or len(map_exits) != 1 or len(nsdfgs) != 1:
        return None
    map_entry, map_exit, nsdfg = map_entries[0], map_exits[0], nsdfgs[0]
    if map_exit.map is not map_entry.map:
        return None
    # The Map must be data-parallel (single iteration variable; symbolic
    # bounds are fine), and the NSDFG must live inside its scope (every
    # path NSDFG -> MapExit and MapEntry -> NSDFG).
    if not any(e.src is map_entry for e in body_state.in_edges(nsdfg)):
        return None
    if not any(e.dst is map_exit for e in body_state.out_edges(nsdfg)):
        return None
    # 4. NSDFG body has a single state.
    inner_sdfg = nsdfg.sdfg
    inner_states = list(inner_sdfg.states())
    if len(inner_states) != 1:
        return None
    inner_state = inner_states[0]
    # 5. Find the single-carrier scan-update pattern inside the inner state.
    # The inner state references the outer carry variable through the NSDFG
    # ``symbol_mapping``. Build a copy of the symbol_mapping inverse so the
    # matcher can look for the outer name's inner alias.
    inner_carry_name = None
    for inner_sym, outer_expr in nsdfg.symbol_mapping.items():
        try:
            outer_syms = {str(s) for s in symbolic.pystr_to_symbolic(str(outer_expr)).free_symbols}
        except Exception:
            outer_syms = {str(outer_expr)}
        if loop.loop_variable in outer_syms:
            inner_carry_name = inner_sym
            break
    if inner_carry_name is None:
        return None
    # 6. Find the single carrier inside the inner state via the inner alias.
    inner_carriers = _find_carried_arrays(inner_state, inner_sdfg, inner_carry_name)
    # ``_find_carried_arrays`` returns the inner descriptor names; map back to
    # outer-side names via the NSDFG's connectors. The matcher restricts to
    # connector-named arrays anyway.
    if len(inner_carriers) != 1:
        return None
    inner_carrier_name = inner_carriers[0]
    # 7. Match the scan-update pattern on the inner state using the inner alias.
    inner_match = _match_one_carrier(_InterchangeFakeLoop(inner_carry_name, iter_start, iter_end), inner_sdfg,
                                     inner_state, inner_carrier_name, iter_start, iter_end)
    if inner_match is None:
        return None
    return _CarryMapShape(loop=loop,
                          parent=parent,
                          body_state=body_state,
                          map_entry=map_entry,
                          map_exit=map_exit,
                          nsdfg=nsdfg,
                          inner_state=inner_state,
                          inner_scan=inner_match)


class _InterchangeFakeLoop:
    """Thin shim so :func:`_match_one_carrier` can be called on the inner
    state without instantiating a real LoopRegion -- only the ``loop_variable``
    attribute is read from the loop parameter inside the matcher path used."""

    def __init__(self, loop_variable: str, iter_start, iter_end):
        self.loop_variable = loop_variable
        self._iter_start = iter_start
        self._iter_end = iter_end


def _rewrite_interchange_carry_with_map(shape: _CarryMapShape, sdfg: SDFG) -> Optional[LoopRegion]:
    """Loop-interchange rewrite: relocate the outer carry ``LoopRegion[jk]``
    from outside the inner Map into the NestedSDFG, where it becomes a
    sequential per-thread carry loop.

    Before::

        LoopRegion[jk]
          state
            MapEntry[jl]
              NestedSDFG
                state  -- carry tasklet:  out[jk, jl] = out[jk-1, jl] OP delta[jk, jl]
              MapExit[jl]

    After::

        state                              <-- new parent-graph state
          MapEntry[jl]                     <-- now parallel outer
            NestedSDFG
              LoopRegion[jk]               <-- relocated, runs sequentially per thread
                state  -- same tasklet
            MapExit[jl]

    Zero new buffers, zero new tasklets, zero copies. The carry runs as
    a plain sequential loop INSIDE the parallel Map on the host AND inside
    the GPU kernel per thread; the accumulator lives in a register and the
    array reads/writes go straight to ``pfsqrf`` / ``delta`` global memory.
    """
    parent = shape.parent
    loop = shape.loop
    nsdfg_node = shape.nsdfg
    inner_state = shape.inner_state

    # 1. Create a fresh state in the parent graph that will replace ``loop``.
    new_state = parent.add_state(label=f'{loop.label}_interchanged', is_start_block=(parent.start_block is loop))
    # Rewire interstate edges of the parent around the old LoopRegion -> new state.
    for ie in list(parent.in_edges(loop)):
        parent.add_edge(ie.src, new_state, ie.data)
        parent.remove_edge(ie)
    for oe in list(parent.out_edges(loop)):
        parent.add_edge(new_state, oe.dst, oe.data)
        parent.remove_edge(oe)
    parent.remove_node(loop)
    if parent.start_block is loop:
        parent.start_block = parent.node_id(new_state)

    # 2. Move ALL nodes and edges of the original body state (which holds
    #    the Map + AccessNodes + NSDFG) into ``new_state``.
    body_state = shape.body_state
    for n in list(body_state.nodes()):
        new_state.add_node(n)
    for e in list(body_state.edges()):
        new_state.add_edge(e.src, e.src_conn, e.dst, e.dst_conn, e.data)

    # 3. Inside the NestedSDFG, replace its single state with a new
    #    LoopRegion[carry] whose body is that same single state. This is
    #    the actual interchange: the outer carry-loop becomes the NSDFG's
    #    new top-level CFR, executed per Map thread.
    inner_sdfg = nsdfg_node.sdfg
    carry_var = loop.loop_variable
    # Materialise the loop's init / cond / update statements so they don't
    # share Python objects with the soon-removed outer ``LoopRegion``.
    new_inner_loop = LoopRegion(label=f'{carry_var}_inner_carry',
                                condition_expr=copy.deepcopy(loop.loop_condition),
                                loop_var=carry_var,
                                initialize_expr=copy.deepcopy(loop.init_statement),
                                update_expr=copy.deepcopy(loop.update_statement),
                                inverted=loop.inverted)
    # Move ``inner_state`` (and any other blocks the inner SDFG had) into
    # the new ``LoopRegion``. The inner SDFG already had ``inner_state`` as
    # its only state; we relocate it.
    old_inner_blocks = list(inner_sdfg.nodes())
    old_start = inner_sdfg.start_block
    # Remove inner SDFG -> add the new LoopRegion as its single block ->
    # move the old blocks into the LoopRegion.
    for blk in old_inner_blocks:
        inner_sdfg.remove_node(blk)
    inner_sdfg.add_node(new_inner_loop, is_start_block=True)
    for blk in old_inner_blocks:
        new_inner_loop.add_node(blk, is_start_block=(blk is old_start))

    # 4. The inner SDFG previously had ``carry_var`` (= the outer loop var)
    #    coming in via ``symbol_mapping``. Now ``carry_var`` is OWNED by
    #    the new inner LoopRegion (it's the loop variable), so drop the
    #    symbol_mapping entry to avoid a redundant binding.
    if carry_var in nsdfg_node.symbol_mapping:
        del nsdfg_node.symbol_mapping[carry_var]
    if carry_var in inner_sdfg.symbols:
        # The carry variable is now the loop's own iterator; remove it
        # from the inner SDFG's external symbol set so codegen doesn't
        # expect it as a kernel argument.
        del inner_sdfg.symbols[carry_var]

    # 5. Clean up the (now empty) ``body_state``. We can't remove it from
    #    its parent because it's the body of the original LoopRegion which
    #    has just been removed; the orphaned reference is harmless.
    return new_inner_loop


def _walk_back_to_computation(state: SDFGState, sdfg: SDFG, src_node, carrier: str):
    """Walk backward from a non-transient carrier write through transient
    intermediates and identity-assign (``__out = __inp``) tasklets, looking for
    the first non-identity computational tasklet (a BinOp) OR a pure data
    copy from the carrier itself. Returns ``(primary, in_edges, carry_in_edge)``
    where ``primary`` is either a Tasklet (BinOp / identity) or an AccessNode
    (pure data copy from carrier -- the post-TrivialTaskletElimination
    carry-copy shape ``pfsqif -> pfsqif_index -> pfsqif``).

    Stops on any branch / multi-input chain other than the identity-assign
    pattern. The walk's primary purpose is to surface the SAME computational
    binop that the matcher would have seen on a flat body, even when the
    frontend has wrapped the carrier write in an extra ``assign_NN`` copy
    tasklet (the cloudsc ``slice_pfsqXf`` shape) -- and to also accept the
    post-TTE pure-AN-chain form where the assign tasklet has been folded away.
    """
    cur = src_node
    seen = set()
    incoming_edge = None  # edge whose dst is ``cur`` (or last-traversed transient)
    while True:
        if id(cur) in seen:
            return None, [], None
        seen.add(id(cur))
        if isinstance(cur, nodes.AccessNode):
            desc = sdfg.arrays.get(cur.data)
            if desc is None:
                return None, [], None
            if not getattr(desc, 'transient', False):
                # Non-transient source AccessNode. If it's the carrier itself,
                # we've reached a pure data copy from the carrier -- treat the
                # AN as the primary and the chain's last edge as the carrier
                # read whose subset determines carry-copy vs accumulate
                # classification.
                if cur.data == carrier and incoming_edge is not None:
                    return cur, [incoming_edge], incoming_edge
                return None, [], None
            ins = list(state.in_edges(cur))
            if len(ins) != 1:
                return None, [], None
            incoming_edge = ins[0]
            cur = ins[0].src
            continue
        if isinstance(cur, nodes.Tasklet):
            if cur.code.language != dtypes.Language.Python:
                return None, [], None
            try:
                tree = ast.parse((cur.code.as_string or '').strip())
            except SyntaxError:
                return None, [], None
            if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
                return None, [], None
            rhs = tree.body[0].value
            ie = [e for e in state.in_edges(cur) if e.data is not None and not e.data.is_empty()]
            # Identity-assign: walk through it (its sole input source is the
            # next node back in the chain).
            if isinstance(rhs, ast.Name) and len(ie) == 1:
                # Stop if this identity-tasklet reads carrier directly (the
                # carry-copy case): return it as the primary so the matcher's
                # carry-copy classification handles it.
                src_name, _src_sub = _resolve_input(state, ie[0])
                if src_name == carrier:
                    return cur, ie, ie[0]
                incoming_edge = ie[0]
                cur = ie[0].src
                continue
            # Non-identity: this is the primary.
            return cur, ie, None
        return None, [], None


def _match_composite_body(loop: LoopRegion, sdfg: SDFG) -> Optional[_CompositeBodyScan]:
    """Detect the composite-body scan shape (cloudsc ``for_1133``).

    The outer body has:
    * a carry-copy state with a single identity-assign tasklet (``__out =
      __inp``) reading the carrier at ``[*, loop_var - 1]`` and writing
      ``[*, loop_var]``; AND
    * at least one accumulate state (possibly inside a nested LoopRegion)
      with a binop tasklet whose carry input is the carrier at ``[*, loop_var]``
      and whose write is also to ``[*, loop_var]`` (read-then-modify the same
      slot, accumulating a delta into the level just initialised by the
      carry-copy).

    Returns the match info if both shapes are present on the SAME carrier,
    otherwise ``None``.
    """
    start = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if start is None or end is None or stride is None or stride != 1:
        return None
    loop_var = loop.loop_variable
    if not loop_var:
        return None
    # Per-carrier collection of write sites with their classification.
    # write_sites[carrier] = [(state, primary_tasklet, w_axis, w_others, kind, in_conn, carry_anchor)]
    # kind: 'carry_copy' if the chain into the carrier resolves to a pure read of
    #       carrier at offset -1 (no binop -- identity assign)
    #       'accumulate'  if the chain has a binop tasklet whose carry input
    #       resolves to carrier at offset 0
    write_sites: Dict[str, list] = {}
    for state in loop.all_states():
        # Walk backward from each non-transient AccessNode of any array that
        # has in-edges in this state. The first non-identity tasklet upstream
        # is the "primary" tasklet whose shape determines the kind.
        for an in state.data_nodes():
            if state.in_degree(an) == 0:
                continue
            desc = sdfg.arrays.get(an.data)
            if desc is None or getattr(desc, 'transient', False):
                continue
            carrier = an.data
            in_edges = list(state.in_edges(an))
            if len(in_edges) != 1:
                continue
            write_subset = in_edges[0].data.subset if in_edges[0].data is not None else None
            if write_subset is None:
                continue
            w_axis, k_w, w_others, w_coef = _classify_subset(write_subset, loop_var)
            if w_axis is None or w_coef != 1 or k_w != 0:
                continue
            # Walk back: find the first BinOp tasklet upstream (the
            # computational tasklet) by traversing identity-assign tasklets and
            # transient AccessNodes. The walk stops at the first non-identity
            # tasklet OR returns None if it reaches a non-transient source.
            primary, primary_in_edges, primary_carry_in_edge = _walk_back_to_computation(
                state, sdfg, in_edges[0].src, carrier)
            if primary is None:
                continue
            kind = None
            in_conn = None
            carry_anchor = None
            # Special case: primary is an AccessNode (post-TTE pure data copy
            # from the carrier itself). Classify the chain's source-edge
            # subset directly.
            if isinstance(primary, nodes.AccessNode):
                if primary.data != carrier or primary_carry_in_edge is None:
                    continue
                src_subset = primary_carry_in_edge.data.subset
                if src_subset is None:
                    continue
                r_axis, k_r, r_others, r_coef = _classify_subset(src_subset, loop_var)
                if (r_axis == w_axis and r_coef == 1 and _same_other_indices(r_others, w_others) and k_r is not None):
                    try:
                        diff = int(symbolic.simplify(k_w - k_r))
                    except Exception:
                        diff = None
                    if diff == 1:
                        kind = 'carry_copy_pure'
                        # No tasklet to mutate -- the rewrite must replace the
                        # AN-source with a tasklet emitting 0.
                        carry_anchor = primary_carry_in_edge.src
                if kind is not None:
                    write_sites.setdefault(carrier, []).append(
                        (state, primary, w_axis, w_others, kind, in_conn, carry_anchor))
                continue
            # Parse the primary tasklet's body.
            try:
                tree = ast.parse((primary.code.as_string or '').strip())
            except SyntaxError:
                continue
            if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
                continue
            rhs = tree.body[0].value
            # Identity-assign primary: the chain into the carrier is a pure
            # read of carrier at SOME subset. Walk back from primary's input to
            # find the source AccessNode subset.
            if isinstance(rhs, ast.Name) and len(primary_in_edges) == 1:
                ie = primary_in_edges[0]
                src_name, src_subset = _resolve_input(state, ie)
                if src_name == carrier and src_subset is not None:
                    r_axis, k_r, r_others, r_coef = _classify_subset(src_subset, loop_var)
                    if (r_axis == w_axis and r_coef == 1 and _same_other_indices(r_others, w_others)
                            and k_r is not None):
                        try:
                            diff = int(symbolic.simplify(k_w - k_r))
                        except Exception:
                            diff = None
                        if diff == 1:
                            kind = 'carry_copy'
                            in_conn = ie.dst_conn
                            carry_anchor = ie.src
            # BinOp primary tasklet (the accumulate): two inputs, one is the
            # carrier at offset 0.
            if kind is None and isinstance(rhs, ast.BinOp) and len(primary_in_edges) == 2:
                op = _BINOP_TO_SCAN_OP.get(type(rhs.op))
                if op == ScanOp.SUM:
                    for ie in primary_in_edges:
                        src_name, src_subset = _resolve_input(state, ie)
                        if src_name != carrier or src_subset is None:
                            continue
                        r_axis, k_r, r_others, r_coef = _classify_subset(src_subset, loop_var)
                        if (r_axis == w_axis and r_coef == 1 and _same_other_indices(r_others, w_others)
                                and k_r == k_w):
                            kind = 'accumulate'
                            break
            if kind is None:
                continue
            write_sites.setdefault(carrier, []).append((state, primary, w_axis, w_others, kind, in_conn, carry_anchor))

    # Look for a carrier with EXACTLY one carry-copy + at least one accumulate.
    for carrier, sites in write_sites.items():
        carry_copies = [s for s in sites if s[4] in ('carry_copy', 'carry_copy_pure')]
        accumulates = [s for s in sites if s[4] == 'accumulate']
        if len(carry_copies) != 1 or not accumulates:
            continue
        cc_state, cc_tasklet, w_axis, w_others, _, cc_in_conn, cc_anchor = carry_copies[0]
        cc_out_edges = [e for e in cc_state.out_edges(cc_tasklet) if e.data is not None and not e.data.is_empty()]
        if len(cc_out_edges) != 1:
            continue
        cc_out_conn = cc_out_edges[0].src_conn
        # Ensure every accumulate carries the SAME other_indices as the carry-copy
        # (same axis assignments). Composite bodies that write different slots
        # would need per-slot handling; out of scope.
        if not all(_same_other_indices(s[3], w_others) and s[2] == w_axis for s in accumulates):
            continue
        return _CompositeBodyScan(
            out_name=carrier,
            scan_axis=w_axis,
            other_indices=w_others,
            iter_start=start,
            iter_end=end,
            carry_copy_state=cc_state,
            carry_copy_tasklet=cc_tasklet,
            carry_copy_carry_anchor=cc_anchor,
            carry_copy_in_conn=cc_in_conn,
            carry_copy_out_conn=cc_out_conn,
            accumulate_states=[s[0] for s in accumulates],
        )
    return None


def _mutate_carry_copy_to_zero(info: _CompositeBodyScan, sdfg: SDFG):
    """Rewrite the carry-copy state so it writes ``0`` to the carrier instead of
    the carrier's prior-iteration value. Handles both the with-tasklet shape
    (primary is a ``Tasklet``) and the post-TTE pure-data-copy shape (primary
    is an ``AccessNode`` -- the carrier read AN).
    """
    state = info.carry_copy_state
    primary = info.carry_copy_tasklet
    # Find the write AccessNode for the carrier in this state.
    write_an = _find_carried_write_an(state, info.out_name)
    if write_an is None:
        return
    write_in_edges = list(state.in_edges(write_an))
    if len(write_in_edges) != 1:
        return
    write_in = write_in_edges[0]
    if isinstance(primary, nodes.Tasklet):
        # Tasklet primary: sever carry-input chain + rewrite body.
        _disconnect_carry_chain(state, primary, info.carry_copy_in_conn, info.carry_copy_carry_anchor)
        primary.code.as_string = f'{info.carry_copy_out_conn} = 0'
        return
    # AccessNode primary (post-TTE pure copy): the chain is
    # ``carrier_read_AN -> transient_AN -> carrier_write_AN`` with no tasklet.
    # Remove the in-edge into the write AN and prune any now-orphaned transient
    # source ANs. Insert a fresh constant-emit tasklet feeding the write AN.
    write_subset = write_in.data.subset
    state.remove_edge(write_in)
    src = write_in.src
    # Prune the chain backward. Remove any transient AN with no remaining
    # outgoing edges; also remove the non-transient source AN (the carrier-read
    # end of the chain) if it ends up isolated.
    while isinstance(src, nodes.AccessNode):
        # If this node still has other in/out connections, leave it in place.
        if state.out_degree(src) > 0:
            break
        ins = list(state.in_edges(src))
        next_src = ins[0].src if len(ins) == 1 else None
        for ie in ins:
            state.remove_edge(ie)
        # Remove the AN unconditionally now that its outgoing chain is gone
        # and any incoming edges have been cleared. The carrier read AN is
        # safe to drop here: the post-loop seed-add reads ``carrier`` from
        # the parent state, not from this body state.
        state.remove_node(src)
        src = next_src
    # Insert a constant-emit tasklet feeding the write AN.
    zero_t = state.add_tasklet(state.label + '_zero', inputs=set(), outputs={'_o'}, code='_o = 0')
    state.add_edge(zero_t, '_o', write_an, None, mm.Memlet(data=info.out_name, subset=write_subset))


def _rewrite_composite_body(parent: ControlFlowRegion, loop: LoopRegion, info: _CompositeBodyScan, sdfg: SDFG) -> bool:
    """Rewrite a composite-body scan into a vector-scan-layout chain.

    The crucial design point: the post-loop ``Scan`` libnode walks its input
    LINEARLY in memory. Reading the carrier's sliced region directly (e.g.
    ``pfsqif[0:KLON, 1:KLEV + 1]`` for the cloudsc shape) gives a NON-contiguous
    walk -- the scan reads past the slice on every row boundary. To keep the
    Scan correct we allocate a FRESH contiguous ``delta_buf`` shaped
    ``[trip, inner_size]`` and mutate the body's carrier writes to land in
    that buffer instead of the carrier. Then the standard ``Scan(stride =
    inner_size)`` + ``Map`` seed-add (the same as the nested-scan rewrite)
    handles the post-loop work over a linearly contiguous buffer.

    Returns ``True`` on successful rewrite, ``False`` if the rewrite cannot
    proceed (e.g. inner-loop range cannot be determined). The caller treats a
    ``False`` return as "skip this loop; try other matchers".
    """
    import dace
    # Determine the inner-loop variable and its range. The inner var is the
    # symbol in ``other_indices`` whose value comes from an enclosing
    # ``LoopRegion`` (the "column" loop). For cloudsc ``for_1133``,
    # ``other_indices = [(0, jl)]`` and ``jl`` is the iterator of the
    # ``for_jl`` LoopRegion that contains the carry-copy state.
    inner_info = _find_composite_inner_loop_range(info)
    if inner_info is None:
        return False
    inner_var, inner_start, inner_end = inner_info
    inner_size = symbolic.simplify(inner_end - inner_start + 1)

    out_desc = sdfg.arrays[info.out_name]
    trip = symbolic.simplify(info.iter_end - info.iter_start + 1)
    delta_buf, _ = sdfg.add_array(f'{_DELTA_BUF_PREFIX}{info.out_name}', [trip, inner_size],
                                  out_desc.dtype,
                                  transient=True,
                                  find_new_name=True)
    scan_buf, _ = sdfg.add_array(f'{_SCAN_BUF_PREFIX}{info.out_name}', [trip, inner_size],
                                 out_desc.dtype,
                                 transient=True,
                                 find_new_name=True)

    # The OUTER scan loop variable is ``loop.loop_variable`` (the matched
    # outer ``LoopRegion`` that owns the prefix-scan recurrence). The carry-
    # copy state's immediate parent_graph is typically an INNER LoopRegion
    # (e.g. ``for_jl`` in the cloudsc shape) -- walking up to the first
    # ``LoopRegion`` would land on that inner iterator instead of the outer
    # scan axis.
    outer_var = loop.loop_variable
    if not outer_var:
        return False

    # Mutate carry-copy state: write ``0`` to ``delta_buf[outer-iter_start,
    # inner-inner_start]`` instead of the carrier-copy chain.
    if not _composite_replace_carry_copy(info, delta_buf, outer_var, inner_var, inner_start, sdfg):
        return False

    # Mutate each accumulate state: redirect every carrier AccessNode + memlet
    # to ``delta_buf`` with the [outer, inner] subset. The read+write at the
    # SAME carrier slot becomes read+write at the SAME delta_buf slot
    # (the accumulator semantic is preserved).
    for acc_state in info.accumulate_states:
        _composite_redirect_carrier_to_delta_buf(acc_state, info, delta_buf, outer_var, inner_var, inner_start, sdfg)

    # Synthesize a ``_Scan`` info to reuse the nested-scan emit helpers. These
    # produce the [trip, inner_size] Scan + 2-D seed-add Map.
    synth = _Scan(
        op=ScanOp.SUM,
        out_name=info.out_name,
        scan_axis=info.scan_axis,
        k_w=0,
        k_r=-1,
        scan_stride=1,
        other_indices=info.other_indices,
        iter_start=info.iter_start,
        iter_end=info.iter_end,
        body_state=info.carry_copy_state,
        scan_update_tasklet=info.carry_copy_tasklet if isinstance(info.carry_copy_tasklet, nodes.Tasklet) else None,
        carry_in_conn='',
        delta_in_conn=None,
        out_conn='',
        carry_anchor=info.carry_copy_carry_anchor,
        literal_delta=None,
        coef=1,
    )

    out_edges = list(parent.out_edges(loop))
    s_scan = parent.add_state(loop.label + '_scan')
    s_apply = parent.add_state(loop.label + '_scan_apply')
    parent.add_edge(loop, s_scan, dace.InterstateEdge())
    parent.add_edge(s_scan, s_apply, dace.InterstateEdge())
    for e in out_edges:
        parent.remove_edge(e)
        parent.add_edge(s_apply, e.dst, e.data)

    _emit_scan_nested(s_scan, sdfg, synth, delta_buf, scan_buf, trip, inner_var, inner_start, inner_end)
    _emit_seed_add_nested(s_apply, sdfg, synth, scan_buf, trip, inner_var, inner_start, inner_end)
    sdfg.reset_cfg_list()
    return True


def _find_composite_inner_loop_range(info: _CompositeBodyScan):
    """Find the inner-loop range used by the composite-body carrier writes.

    Walks the parent_graph chain of the carry-copy state for a ``LoopRegion``
    whose ``loop_variable`` appears in ``info.other_indices`` and whose
    iteration is unit-stride. Returns ``(inner_var, inner_start, inner_end)``
    or ``None`` if no such range can be determined.
    """
    other_syms = set()
    for _axis, expr in info.other_indices:
        try:
            for s in symbolic.pystr_to_symbolic(str(expr)).free_symbols:
                other_syms.add(str(s))
        except Exception:
            pass
    cur = info.carry_copy_state.parent_graph
    while cur is not None:
        if isinstance(cur, LoopRegion) and cur.loop_variable in other_syms:
            start = loop_analysis.get_init_assignment(cur)
            end = loop_analysis.get_loop_end(cur)
            stride = loop_analysis.get_loop_stride(cur)
            if start is not None and end is not None and stride == 1:
                return cur.loop_variable, start, end
        cur = getattr(cur, 'parent_graph', None)
    return None


def _composite_replace_carry_copy(info: _CompositeBodyScan, delta_buf: str, outer_var: str, inner_var: str,
                                  inner_start: Any, sdfg: SDFG) -> bool:
    """Mutate the carry-copy state so its single carrier-write becomes a
    ``0``-emit tasklet writing ``delta_buf[outer_var - iter_start, inner_var
    - inner_start]``. Severs the carrier-read chain entirely.

    Returns ``True`` on success.
    """
    state = info.carry_copy_state
    write_an = _find_carried_write_an(state, info.out_name)
    if write_an is None:
        return False
    # Shared-carrier-chain refusal: the cloudsc ``for_1133`` two-carrier
    # shape chains ``pfsqlf[jk-1] -> pfsqlf_index -> pfsqlf[jk] (written) ->
    # pfsqrf_slice -> pfsqrf[jk]``. The just-written carrier feeds the
    # sibling carrier's slice via an outgoing edge. Removing ``write_an``
    # via the prune walk below would sever the edge, leaving the sibling's
    # slice transient with ``in_degree == 0`` while still read by the
    # sibling write -- silent garbage on the second carrier. The
    # carry-copy state should terminate at ``write_an`` with NO downstream
    # consumers in the same state; downstream consumption happens in a
    # subsequent state (e.g. the inner accumulate LoopRegion). Any
    # outgoing edge from ``write_an`` here signals a sibling-chain hazard
    # and we refuse the lift.
    if state.out_degree(write_an) > 0:
        return False
    write_in_edges = list(state.in_edges(write_an))
    if len(write_in_edges) != 1:
        return False
    write_in = write_in_edges[0]
    state.remove_edge(write_in)
    # Sever the carry-read chain. Walk back through transients pruning
    # nodes that become orphaned.
    src = write_in.src
    if isinstance(src, nodes.Tasklet):
        state.remove_node(src)
        # Continue pruning upstream of the tasklet.
        upstream_edges = []  # noop -- tasklet remove cascades via remove_node
    while isinstance(src, nodes.AccessNode):
        if state.out_degree(src) > 0:
            break
        ins = list(state.in_edges(src))
        next_src = ins[0].src if len(ins) == 1 else None
        for ie in ins:
            state.remove_edge(ie)
        state.remove_node(src)
        src = next_src
    # Remove the original write AN entirely; we'll add a fresh delta_buf AN.
    state.remove_node(write_an)
    # Insert ``0``-emit tasklet writing delta_buf at [outer_idx, inner_idx].
    outer_idx = symbolic.simplify(symbolic.pystr_to_symbolic(outer_var) - info.iter_start)
    inner_idx = symbolic.simplify(symbolic.pystr_to_symbolic(inner_var) - inner_start)
    zero_t = state.add_tasklet(state.label + '_zero', inputs=set(), outputs={'_o'}, code='_o = 0')
    db_an = state.add_write(delta_buf)
    state.add_edge(
        zero_t, '_o', db_an, None,
        mm.Memlet(data=delta_buf, subset=subsets.Range([(outer_idx, outer_idx, 1), (inner_idx, inner_idx, 1)])))
    return True


def _composite_redirect_carrier_to_delta_buf(state: SDFGState, info: _CompositeBodyScan, delta_buf: str, outer_var: str,
                                             inner_var: str, inner_start: Any, sdfg: SDFG):
    """Replace each carrier ``AccessNode`` in ``state`` with a FRESH ``delta_buf``
    ``AccessNode`` and update only the BOUNDARY edges' memlets (the edges that
    touched the carrier AN). Intermediate transients (``pfsqif_index_0`` /
    ``pfsqif_slice_plus_*``) keep their original 1-D scalar-slice shapes and
    their interior memlets stay untouched. Element counts match (1 -> 1) on
    every boundary edge, so the post-rewrite SDFG validates and codegens
    correctly.
    """
    outer_idx = symbolic.simplify(symbolic.pystr_to_symbolic(outer_var) - info.iter_start)
    inner_idx = symbolic.simplify(symbolic.pystr_to_symbolic(inner_var) - inner_start)
    new_subset = subsets.Range([(outer_idx, outer_idx, 1), (inner_idx, inner_idx, 1)])
    carrier_ans = [n for n in list(state.nodes()) if isinstance(n, nodes.AccessNode) and n.data == info.out_name]
    for an in carrier_ans:
        # Create the fresh delta_buf AN to replace this carrier endpoint.
        new_an = state.add_access(delta_buf)
        # Move every in-edge to the new AN with a re-subset memlet.
        for e in list(state.in_edges(an)):
            state.remove_edge(e)
            state.add_edge(e.src, e.src_conn, new_an, None, mm.Memlet(data=delta_buf, subset=copy.deepcopy(new_subset)))
        # Move every out-edge from the new AN with a re-subset memlet.
        for e in list(state.out_edges(an)):
            state.remove_edge(e)
            state.add_edge(new_an, None, e.dst, e.dst_conn, mm.Memlet(data=delta_buf, subset=copy.deepcopy(new_subset)))
        state.remove_node(an)


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
        is_scalar = isinstance(desc, data.Scalar) or (isinstance(desc, data.Array) and tuple(desc.shape) == (1, ))
        if not is_scalar:
            continue
        candidates.add(n.data)
        if state.in_degree(n) > 0:
            writes.add(n.data)
        if state.out_degree(n) > 0:
            reads.add(n.data)
    return sorted(candidates & reads & writes)


def _match_one_scalar_carry(loop: LoopRegion, sdfg: SDFG, state: SDFGState, acc_name: str, iter_start: Any,
                            iter_end: Any, loop_var: str) -> Optional[_ScalarCarryScan]:
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
    tasklet, acc_in_conn, delta_in_conn, out_conn, op = _trace_back_to_rmw_tasklet(state, write_an, acc_name)
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


def _trace_back_to_rmw_tasklet(state: SDFGState, write_an: nodes.AccessNode, acc_name: str):
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
    in_data_edges = [e for e in state.in_edges(tasklet) if e.data is not None and not e.data.is_empty()]
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
            scan_axis, offset, others, coef = _classify_subset(sub, loop_var)
            if scan_axis is None or coef != 1:
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
                scan_axis, offset, others, coef = _classify_subset(sub, loop_var)
                if scan_axis is None or coef != 1:
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

    Scans ALL edges in the state -- top-level edges plus any inside Map scopes --
    for memlets whose data is a non-transient and whose subset depends on
    ``loop_var``. Additionally descends into ``NestedSDFG`` scopes (the post-
    ``LoopToMap`` shape: outer carry-loop -> body state with a Map -> NestedSDFG
    holding the actual prefix-scan tasklet). State-level edges around the Map
    have widened subsets that no longer reference ``loop_var`` (Map-exit
    propagation), so we look inside the NestedSDFG, find the inner memlets that
    DO carry the carrier subset, and lift them back through the NSDFG's
    ``symbol_mapping`` to detect that the outer-loop-var appears in the
    underlying access. This is the for_1133 / cloudsc descend-into-Map path.
    Walking edges rather than only AccessNode-incident memlets catches
    carriers whose state-level memlets are intermediates of slice copies in
    the flat (v1-v5) case.
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

    # Descend into NestedSDFG scopes within this state (Map bodies after L2M).
    # An inner memlet ``pfsqrf[jk, jl]`` whose inner symbol ``jk`` is bound via
    # ``symbol_mapping`` to the outer ``loop_var`` flags ``pfsqrf`` as a carrier.
    arrays_with_carrier_subset.update(_find_carried_arrays_via_nested(state, sdfg, loop_var))

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


def _find_carried_arrays_via_nested(state: SDFGState, sdfg: SDFG, loop_var: str) -> set:
    """Walk ``state``'s NestedSDFG nodes; for each, look at inner memlets and
    lift them through ``symbol_mapping`` to see whether the outer ``loop_var``
    appears in the underlying access. Returns the set of OUTER array names
    (the names of the NSDFG's outer-side connectors) flagged as carriers.

    Caveat: the lookup only follows one level of nesting at a time, but
    recurses if the inner SDFG itself contains a NestedSDFG.
    """
    flagged: set = set()
    for n in state.nodes():
        if not isinstance(n, nodes.NestedSDFG):
            continue
        # Build the connector -> outer-array map by looking at the cross edges.
        conn_to_outer: Dict[str, str] = {}
        for e in state.in_edges(n):
            if e.dst_conn and e.data is not None and e.data.data is not None:
                conn_to_outer[e.dst_conn] = e.data.data
        for e in state.out_edges(n):
            if e.src_conn and e.data is not None and e.data.data is not None:
                conn_to_outer[e.src_conn] = e.data.data
        # Inner symbols that bind to the outer ``loop_var``. ``symbol_mapping``
        # is ``{inner_sym: outer_expr}``: collect inner_sym keys whose mapped
        # outer expression has ``loop_var`` as a free symbol.
        inner_syms_carrying: set = set()
        for inner_sym, outer_expr in n.symbol_mapping.items():
            try:
                outer_sym_strs = {str(s) for s in symbolic.pystr_to_symbolic(str(outer_expr)).free_symbols}
            except Exception:
                outer_sym_strs = {str(outer_expr)}
            if loop_var in outer_sym_strs:
                inner_syms_carrying.add(inner_sym)
        # Walk inner SDFG memlets; for each that uses one of those inner
        # symbols, flag the corresponding outer array.
        for inner_state in n.sdfg.states():
            for e in inner_state.edges():
                m = e.data
                if m is None or m.data is None or m.subset is None:
                    continue
                inner_desc = n.sdfg.arrays.get(m.data)
                if inner_desc is None:
                    continue
                # The inner-side data name == the connector name (DaCe binds
                # connector -> inner descriptor by name). Map back to outer.
                outer_name = conn_to_outer.get(m.data)
                if outer_name is None:
                    continue
                outer_desc = sdfg.arrays.get(outer_name)
                if outer_desc is None or getattr(outer_desc, 'transient', False):
                    continue
                # The subset uses ``loop_var`` (after mapping) if any of the
                # inner symbols carrying ``loop_var`` appears in the subset's
                # free symbols.
                inner_subset_syms = {str(s) for s in m.subset.free_symbols}
                if inner_subset_syms & inner_syms_carrying:
                    flagged.add(outer_name)
        # Recurse one more level for safety (NSDFG inside NSDFG).
        for inner_state in n.sdfg.states():
            flagged.update(_find_carried_arrays_via_nested(inner_state, n.sdfg, loop_var))
    return flagged


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


def _admissible_scan_stride(diff):
    """Return the scan stride if ``diff`` (``= k_w - k_r`` in iteration order) is an
    admissible positive stride, else ``None``.

    A *constant* integer stride must be ``>= 1`` -- ``1`` is the contiguous scan,
    ``S > 1`` the residue-class scan (TSVC ``s1221`` ``b[i] = b[i-4] + a[i]``).

    A *symbolic* integer stride whose sign cannot be proven ``<= 0`` is also
    admissible (TSVC-2.5 ``scan_strided_sym`` ``a[i] = a[i-K] + x[i]``): the
    residue-class decomposition into ``stride`` independent prefix scans is valid
    for any runtime value ``>= 1``, and :meth:`LoopToScan.apply_pass` guards the
    lift with an ``if stride >= 1: scan else: sequential`` specialization for the
    values it cannot prove. Returns an ``int`` for a constant stride and the
    sympy expression itself for a symbolic one (the ``Scan`` libnode's ``stride``
    property and its residue-class expansions accept both via ``sym2cpp``).
    """
    if diff is None or not isinstance(diff, sympy.Basic):
        return None
    if diff.is_Integer:
        return int(diff) if int(diff) >= 1 else None
    # Symbolic: admit an integer-typed stride whose sign is not provably non-positive.
    if diff.is_integer and diff.is_nonpositive is not True:
        return diff
    return None


def _symbolic_stride_guard(infos: List['_Scan']) -> Optional[str]:
    """The ``&&``-joined ``stride >= 1`` predicate for every matched scan whose
    stride is symbolic with a sign not provably positive, or ``None`` when all
    strides lift unconditionally (constant, or a provably-positive symbol).

    A residue-class scan is only valid for ``stride >= 1``; the returned
    predicate is the true-branch guard of the ``if stride >= 1: scan else: seq``
    specialization :meth:`LoopToScan._specialize_scan_under_stride_guard` builds.
    """
    conds: List[str] = []
    seen = set()
    for info in infos:
        s = info.scan_stride
        if isinstance(s, sympy.Basic) and not s.is_Integer and s.is_positive is not True:
            cond = f'({symbolic.symstr(s)}) >= 1'
            if cond not in seen:
                seen.add(cond)
                conds.append(cond)
    return ' && '.join(conds) if conds else None


def _find_scan_update_tasklet(state: SDFGState,
                              sdfg: SDFG,
                              out_name: str,
                              loop_var: str,
                              scan_axis: int,
                              write_others,
                              k_w,
                              write_coef=1):
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
        in_edges = [e for e in state.in_edges(node) if e.data is not None and not e.data.is_empty()]
        out_edges = [e for e in state.out_edges(node) if e.data is not None and not e.data.is_empty()]
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
                r_axis, k_r_cand, r_others, r_coef = _classify_subset(src_subset, loop_var)
                # Carry-read must have the SAME coefficient on the scan axis as
                # the write (both forward or both reverse). Otherwise we'd be
                # mixing iteration directions, which the scan rewrite can't
                # express.
                # Accept any *positive integer* scan stride in iter order:
                #   forward (coef +1): stride = k_w - k_r
                #   reverse (coef -1): stride = k_r - k_w   (write moves DOWN
                #     in array order; read at higher array index = prior iter's
                #     write target)
                # ``1`` is the contiguous scan; ``S > 1`` is the residue-class
                # scan that the ``Scan`` libnode's ``stride`` property handles
                # natively (TSVC s1221 ``b[i] = b[i-4] + a[i]``).
                if r_axis == scan_axis and _same_other_indices(r_others, write_others) and r_coef == write_coef:
                    try:
                        diff = symbolic.simplify((k_w - k_r_cand) if write_coef == 1 else (k_r_cand - k_w))
                    except Exception:
                        diff = None
                    stride_val = _admissible_scan_stride(diff)
                    if stride_val is not None:
                        if carry_edge is not None:
                            ambiguous = True
                            break
                        carry_edge = e
                        carry_anchor = e.src
                        scan_stride = stride_val
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
        if (isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub) and isinstance(n.operand, ast.Constant)
                and isinstance(n.operand.value, (int, float)) and not isinstance(n.operand.value, bool)):
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
    """Return ``(scan_axis, offset, non_scan_indices, coef)`` for ``subset``, or
    ``(None, None, None, 0)`` if the subset doesn't fit the v1 shape.

    The "v1 shape": every dimension is a single point (``lo == hi``, stride 1),
    *exactly one* axis depends on ``loop_var`` (linearly with constant offset),
    all other axes are loop-invariant.

    ``coef`` is the coefficient of ``loop_var`` on the scan axis -- ``+1`` for
    the canonical forward case (``loop_var + k``) or ``-1`` for the reverse
    case (``c - loop_var`` -- the cloudsc ``for_1079`` backward shape after
    :class:`NormalizeNegativeStride` rewrites the loop but leaves the body
    subsets in their original form). When ``coef == -1`` the returned
    ``offset`` is the constant ``c`` (i.e. the array position visited at the
    first iteration); when ``coef == +1`` it is the canonical ``k`` offset.
    """
    if not isinstance(subset, subsets.Range):
        return None, None, None, 0
    loop_var_sym = symbolic.pystr_to_symbolic(loop_var)
    scan_axis = None
    offset = None
    coef = 0
    others: List[Any] = []
    for axis_idx, (lo, hi, st) in enumerate(subset.ranges):
        if lo != hi or st != 1:
            return None, None, None, 0
        lo_sym = symbolic.pystr_to_symbolic(str(lo))
        if loop_var_sym in lo_sym.free_symbols:
            if scan_axis is not None:
                return None, None, None, 0
            # Try forward (coef +1): off = lo - loop_var.
            try:
                off_pos = symbolic.simplify(lo_sym - loop_var_sym)
            except Exception:
                off_pos = None
            if off_pos is not None and loop_var_sym not in off_pos.free_symbols:
                scan_axis = axis_idx
                offset = off_pos
                coef = 1
                continue
            # Reverse (coef -1): off = lo + loop_var (so lo = const - loop_var).
            try:
                off_neg = symbolic.simplify(lo_sym + loop_var_sym)
            except Exception:
                off_neg = None
            if off_neg is not None and loop_var_sym not in off_neg.free_symbols:
                scan_axis = axis_idx
                offset = off_neg
                coef = -1
                continue
            return None, None, None, 0
        else:
            others.append((axis_idx, lo_sym))
    return scan_axis, offset, others, coef


def _same_other_indices(a, b) -> bool:
    """Compare two ``[(axis, expr), ...]`` lists for exact symbolic equality."""
    if len(a) != len(b):
        return False
    for (ax_a, ex_a), (ax_b, ex_b) in zip(a, b):
        if ax_a != ax_b or symbolic.simplify(ex_a - ex_b) != 0:
            return False
    return True


def _in_conditional_branch(state: SDFGState) -> bool:
    """``True`` iff the parent chain of ``state`` includes a ``ConditionalBlock``."""
    cur = getattr(state, 'parent_graph', None)
    while cur is not None:
        if isinstance(cur, ConditionalBlock):
            return True
        cur = getattr(cur, 'parent_graph', None)
    return False


def _identity_for_op(op) -> float:
    """The identity element for an associative scan op -- writes to a per-iter
    ``delta_buf[i]`` of this value contribute nothing to the fold.
    """
    if op == ScanOp.SUM:
        return 0.0
    if op == ScanOp.PRODUCT:
        return 1.0
    # MIN / MAX: identity would be +inf / -inf; conditional-branch matches that
    # use these ops are out of scope for the bare init helper.
    return 0.0


def _emit_delta_buf_zero_init(parent: ControlFlowRegion, loop: LoopRegion, sdfg: SDFG, delta_buf: str, trip: Any, op):
    """Insert a state BEFORE ``loop`` that fills ``delta_buf`` with the op's
    identity element (0 for SUM, 1 for PRODUCT). The state's single ``Map`` over
    the trip range writes one element per iteration.
    """
    import dace
    init_val = _identity_for_op(op)
    s_init = parent.add_state(loop.label + '_delta_init')
    pre_edges = list(parent.in_edges(loop))
    for e in pre_edges:
        parent.remove_edge(e)
        parent.add_edge(e.src, s_init, e.data)
    parent.add_edge(s_init, loop, dace.InterstateEdge())
    s_init.add_mapped_tasklet(
        s_init.label + '_t',
        {'_di': f'0:{trip}'},
        {},
        f'_o = {init_val}',
        {'_o': mm.Memlet(data=delta_buf, subset=subsets.Range([('_di', '_di', 1)]))},
        external_edges=True,
    )


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
    delta_buf, _ = sdfg.add_array(f'{_DELTA_BUF_PREFIX}{info.out_name}', [trip],
                                  out_desc.dtype,
                                  transient=True,
                                  find_new_name=True)

    # If the matched scan-update lives inside a ``ConditionalBlock`` branch,
    # sibling branches that skip the delta computation leave their iteration's
    # ``delta_buf[i]`` uninitialised. Pre-zero ``delta_buf`` so those iterations
    # contribute the identity element. Cheap (one Map over the iter range) and
    # only emitted when actually needed.
    if _in_conditional_branch(info.body_state):
        _emit_delta_buf_zero_init(parent, loop, sdfg, delta_buf, trip, info.op)

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
        scan_buf, _ = sdfg.add_array(f'{_SCAN_BUF_PREFIX}{info.out_name}', [trip],
                                     out_desc.dtype,
                                     transient=True,
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
    delta_buf, _ = sdfg.add_array(f'{_DELTA_BUF_PREFIX}{info.out_name}', [trip, inner_size],
                                  out_desc.dtype,
                                  transient=True,
                                  find_new_name=True)
    scan_buf, _ = sdfg.add_array(f'{_SCAN_BUF_PREFIX}{info.out_name}', [trip, inner_size],
                                 out_desc.dtype,
                                 transient=True,
                                 find_new_name=True)

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


def _build_nested_carrier_subset(desc: data.Array, info: _Scan, scan_expr, inner_var: str,
                                 inner_axis_expr) -> subsets.Range:
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


def _build_nested_carrier_range_subset(desc: data.Array, info: _Scan, scan_lo, scan_hi, inner_var: str, inner_start,
                                       inner_end) -> subsets.Range:
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
    state.add_edge(
        final_edge.src, final_edge.src_conn, buf_an, None,
        mm.Memlet(data=delta_buf, subset=subsets.Range([(outer_idx, outer_idx, 1), (inner_idx, inner_idx, 1)])))
    if state.degree(write_an) == 0:
        state.remove_node(write_an)


def _emit_scan_nested(state: SDFGState, sdfg: SDFG, info: _Scan, delta_buf: str, scan_buf: str, trip: Any,
                      inner_var: str, inner_start: Any, inner_end: Any):
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
    state.add_edge(delta_read, None, node, INPUT_CONNECTOR_NAME,
                   mm.Memlet(data=delta_buf, subset=subsets.Range([(0, trip - 1, 1), (0, inner_size - 1, 1)])))
    state.add_edge(node, OUTPUT_CONNECTOR_NAME, scan_write, None,
                   mm.Memlet(data=scan_buf, subset=subsets.Range([(0, trip - 1, 1), (0, inner_size - 1, 1)])))


def _emit_seed_add_nested(state: SDFGState, sdfg: SDFG, info: _Scan, scan_buf: str, trip: Any, inner_var: str,
                          inner_start: Any, inner_end: Any):
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
    tasklet = state.add_tasklet(f'{state.label}_apply', inputs={'_seed', '_delta'}, outputs={'_o'}, code=code)
    me, mx = state.add_map(state.label + '_map', {
        map_i: subsets.Range([(0, trip - 1, 1)]),
        map_j: subsets.Range([(inner_start, inner_end, 1)])
    })
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
    state.add_edge(
        out_read, None, me, 'IN_seed',
        mm.Memlet(data=info.out_name,
                  subset=_build_nested_carrier_range_subset(out_desc, info, seed_axis_expr, seed_axis_expr, inner_var,
                                                            inner_start, inner_end)))
    state.add_edge(scan_read, None, me, 'IN_scan',
                   mm.Memlet(data=scan_buf, subset=subsets.Range([(0, trip - 1, 1), (0, inner_size - 1, 1)])))
    state.add_edge(
        me, 'OUT_seed', tasklet, '_seed',
        mm.Memlet(data=info.out_name,
                  subset=_build_nested_carrier_subset(out_desc, info, seed_axis_expr, inner_var, map_j_sym)))
    state.add_edge(me, 'OUT_scan', tasklet, '_delta',
                   mm.Memlet(data=scan_buf, subset=subsets.Range([(map_i, map_i, 1), (buf_j_local, buf_j_local, 1)])))
    state.add_edge(
        tasklet, '_o', mx, 'IN_o',
        mm.Memlet(data=info.out_name,
                  subset=_build_nested_carrier_subset(out_desc, info, write_axis_expr, inner_var, map_j_sym)))
    state.add_edge(
        mx, 'OUT_o', out_write, None,
        mm.Memlet(data=info.out_name,
                  subset=_build_nested_carrier_range_subset(out_desc, info,
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
    output array, no other-axis indices, ``scan_stride == 1``, AND forward
    iteration (``coef == 1``) so the write subset
    ``out[start + k_w : start + k_w + trip]`` is a contiguous range walked in
    iter order. Stride > 1 residue-class scans + reverse-iteration scans fall
    through to the general 3-stage path (the reverse case needs an explicit Map
    to write ``out[k_w - i]`` in array-reversed order from the iter-order
    ``scan_buf``).
    """
    return (len(out_desc.shape) == 1 and not info.other_indices and info.scan_stride == 1 and info.coef == 1)


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
    seed_name, _ = sdfg.add_scalar(f'{_SEED_SCALAR_PREFIX}{info.out_name}',
                                   out_desc.dtype,
                                   transient=True,
                                   find_new_name=True)
    seed_axis_expr = symbolic.simplify(info.iter_start + info.k_r)
    write_start = symbolic.simplify(info.iter_start + info.k_w)
    write_end = symbolic.simplify(write_start + trip - 1)

    out_seed_read = state.add_read(info.out_name)
    seed_an = state.add_access(seed_name)
    delta_read = state.add_read(delta_buf)
    out_write = state.add_write(info.out_name)
    node = Scan(name=f'{state.label}_op', op=info.op, exclusive=False)
    node.add_in_connector(INIT_CONNECTOR_NAME)
    state.add_node(node)

    state.add_edge(
        out_seed_read, None, seed_an, None,
        mm.Memlet(data=info.out_name,
                  subset=subsets.Range([(seed_axis_expr, seed_axis_expr, 1)]),
                  other_subset=subsets.Range([(0, 0, 1)])))
    state.add_edge(seed_an, None, node, INIT_CONNECTOR_NAME, mm.Memlet(data=seed_name,
                                                                       subset=subsets.Range([(0, 0, 1)])))
    state.add_edge(delta_read, None, node, INPUT_CONNECTOR_NAME,
                   mm.Memlet(data=delta_buf, subset=subsets.Range([(0, trip - 1, 1)])))
    state.add_edge(node, OUTPUT_CONNECTOR_NAME, out_write, None,
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
    # Walk up the parent_graph chain to find the enclosing scan ``LoopRegion``.
    # For the flat case ``state.parent_graph`` IS that LoopRegion; for a
    # state inside a ``ConditionalBlock`` branch the chain is
    # ``state -> branch (ControlFlowRegion) -> ConditionalBlock -> LoopRegion``.
    cur = info.body_state.parent_graph
    while cur is not None and not (isinstance(cur, LoopRegion) and cur.loop_variable):
        cur = getattr(cur, 'parent_graph', None)
    if cur is None:
        return
    loop_var = cur.loop_variable
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


def _disconnect_carry_chain(state: SDFGState, tasklet: nodes.Tasklet, conn: str, anchor: nodes.AccessNode):
    """Remove the tasklet's carry-input edge and the now-dead chain that fed it.

    The feeding chain is either a pure slice-copy (``out_read -> tmp ->`` carry)
    or a frontend copy tasklet that materialises the carry element into a scalar
    (``a_index = a[i - 1]`` -- the TSVC s242 family). Both become dead once the
    carry edge is severed: the post-loop ``Scan`` supplies the recurrence carry
    and the seed-add reads ``out`` from its own state, so nothing in the body
    still needs the per-iteration carry read. The remaining ``a_index = a[i - 1]``
    is an array->scalar read on a loop-varying subset, which the vectorizer
    cannot lower (``SCALAR_ARRAY_ASSIGNMENT``) -- leaving it dead blocks the
    otherwise-parallel delta-build Map from vectorizing.

    Walk backward from ``anchor`` removing every node that no longer has
    consumers, recursing into its producers. A node is safe to drop when its
    out-degree is 0 and it is a ``Tasklet``, a transient AccessNode (a dead
    transient write), or an isolated non-transient read AccessNode -- never a
    live write to a non-transient array, nor a node still read elsewhere.
    """
    if conn in tasklet.in_connectors:
        for e in list(state.in_edges(tasklet)):
            if e.dst_conn == conn:
                state.remove_edge(e)
        tasklet.remove_in_connector(conn)
    worklist = [anchor]
    while worklist:
        cur = worklist.pop()
        if cur not in state.nodes() or state.out_degree(cur) != 0:
            # Gone already, or still consumed elsewhere (e.g. another reader of
            # ``out``, or the carry source also feeding the seed read).
            continue
        if isinstance(cur, nodes.AccessNode):
            desc = state.sdfg.arrays.get(cur.data)
            if (desc is None or not desc.transient) and state.in_degree(cur) != 0:
                # A live write to a non-transient array -- never drop.
                continue
        elif not isinstance(cur, nodes.Tasklet):
            continue
        producers = [e.src for e in state.in_edges(cur)]
        for ie in list(state.in_edges(cur)):
            state.remove_edge(ie)
        state.remove_node(cur)
        worklist.extend(producers)


def _collect_output_chain(state: SDFGState, tasklet: nodes.Tasklet, out_conn: str) -> List[nodes.AccessNode]:
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
    state.add_edge(r, None, node, INPUT_CONNECTOR_NAME,
                   mm.Memlet(data=delta_buf, subset=subsets.Range([(0, trip - 1, 1)])))
    state.add_edge(node, OUTPUT_CONNECTOR_NAME, w, None,
                   mm.Memlet(data=scan_buf, subset=subsets.Range([(0, trip - 1, 1)])))


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
    if info.coef == -1:
        # Reverse iteration: ``k_w`` / ``k_r`` are array constants (positions at
        # iter 0). Seed reads from ``out[k_r]`` (pre-loop value, broadcast),
        # write goes to ``out[k_w - _i]`` (i.e. iter 0 writes the highest array
        # index, iter trip-1 writes the lowest).
        seed_axis_expr = symbolic.simplify(info.k_r)
        write_axis_expr = symbolic.simplify(info.k_w - _i)
    elif info.scan_stride == 1:
        seed_axis_expr = symbolic.simplify(info.iter_start + info.k_r)
        write_axis_expr = symbolic.simplify(info.iter_start + info.k_w) + _i
    else:
        import sympy
        class_idx = sympy.Mod(_i, info.scan_stride)
        seed_axis_expr = symbolic.simplify(info.iter_start + info.k_r + class_idx)
        write_axis_expr = symbolic.simplify(info.iter_start + info.k_w) + _i
    seed_subset = _build_subset(out_desc, info.scan_axis, seed_axis_expr, info.other_indices)
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


def _rewrite_scalar_carry(parent: ControlFlowRegion, loop: LoopRegion, info: _ScalarCarryScan, sdfg: SDFG):
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

    delta_buf, _ = sdfg.add_array(f'{_DELTA_BUF_PREFIX}{info.out_name}', [trip],
                                  out_desc.dtype,
                                  transient=True,
                                  find_new_name=True)
    scan_buf, _ = sdfg.add_array(f'{_SCAN_BUF_PREFIX}{info.out_name}', [trip],
                                 out_desc.dtype,
                                 transient=True,
                                 find_new_name=True)

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


def _emit_scalar_carry_delta_build(state: SDFGState, sdfg: SDFG, info: _ScalarCarryScan, delta_buf: str):
    """Map ``_i`` over ``[0, trip)`` copying
    ``delta[iter_start + _i + delta_offset, ...]`` into ``delta_buf[_i]``."""
    trip = symbolic.simplify(info.iter_end - info.iter_start + 1)
    delta_desc = sdfg.arrays[info.delta_name]
    _i = symbolic.pystr_to_symbolic('_i')
    delta_axis_expr = symbolic.simplify(info.iter_start + _i + info.delta_offset)
    delta_subset = _build_subset(delta_desc, info.delta_scan_axis, delta_axis_expr, info.delta_other_indices)
    state.add_mapped_tasklet(
        f'{state.label}_tasklet',
        {'_i': f'0:{trip}'},
        {'__d': mm.Memlet(data=info.delta_name, subset=delta_subset)},
        '__o = __d',
        {'__o': mm.Memlet(data=delta_buf, subset=subsets.Range([(_i, _i, 1)]))},
        external_edges=True,
    )


def _emit_scalar_carry_scan(state: SDFGState, sdfg: SDFG, info: _ScalarCarryScan, delta_buf: str, scan_buf: str,
                            trip: Any):
    """Run the ``Scan`` libnode with the accumulator's pre-loop value wired into
    the optional ``_scan_init`` connector. Inclusive semantics +
    ``acc[0]`` seed make ``scan_buf[i] = acc_initial OP delta_buf[0]
    OP ... OP delta_buf[i]``.

    The seed is read into a per-instance scalar transient to keep the libnode's
    init-connector edge a clean Scalar memlet (the same pattern
    :func:`_emit_scan_with_init_direct` uses).
    """
    acc_desc = sdfg.arrays[info.acc_name]
    seed_name, _ = sdfg.add_scalar(f'{_SEED_SCALAR_PREFIX}{info.out_name}',
                                   acc_desc.dtype,
                                   transient=True,
                                   find_new_name=True)
    acc_read = state.add_read(info.acc_name)
    seed_an = state.add_access(seed_name)
    delta_read = state.add_read(delta_buf)
    scan_write = state.add_write(scan_buf)

    node = Scan(name=f'{state.label}_op', op=info.op, exclusive=False)
    node.add_in_connector(INIT_CONNECTOR_NAME)
    state.add_node(node)

    state.add_edge(
        acc_read, None, seed_an, None,
        mm.Memlet(data=info.acc_name, subset=subsets.Range([(0, 0, 1)]), other_subset=subsets.Range([(0, 0, 1)])))
    state.add_edge(seed_an, None, node, INIT_CONNECTOR_NAME, mm.Memlet(data=seed_name,
                                                                       subset=subsets.Range([(0, 0, 1)])))
    state.add_edge(delta_read, None, node, INPUT_CONNECTOR_NAME,
                   mm.Memlet(data=delta_buf, subset=subsets.Range([(0, trip - 1, 1)])))
    state.add_edge(node, OUTPUT_CONNECTOR_NAME, scan_write, None,
                   mm.Memlet(data=scan_buf, subset=subsets.Range([(0, trip - 1, 1)])))


def _emit_scalar_carry_out_write(state: SDFGState, sdfg: SDFG, info: _ScalarCarryScan, scan_buf: str, trip: Any):
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


def _emit_scalar_carry_acc_post(state: SDFGState, sdfg: SDFG, info: _ScalarCarryScan, scan_buf: str, trip: Any):
    """Copy ``scan_buf[trip - 1]`` into ``acc[0]`` so downstream readers of
    ``acc`` see the post-loop running value. Single-tasklet state, no Map (it's
    a scalar move)."""
    scan_read = state.add_read(scan_buf)
    acc_write = state.add_write(info.acc_name)
    t = state.add_tasklet(name=f'{state.label}_writeback',
                          inputs={'__v'},
                          outputs={'__o'},
                          code='__o = __v',
                          language=dtypes.Language.Python)
    state.add_edge(scan_read, None, t, '__v', mm.Memlet(data=scan_buf, subset=subsets.Range([(trip - 1, trip - 1, 1)])))
    state.add_edge(t, '__o', acc_write, None, mm.Memlet(data=info.acc_name, subset=subsets.Range([(0, 0, 1)])))
