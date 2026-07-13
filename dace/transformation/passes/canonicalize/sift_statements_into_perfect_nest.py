# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Sift outer-level statements into the inner loop to make an imperfect nest perfect (GPU-only).

The loop analogue of :class:`~dace.transformation.passes.move_if_into_loop.MoveIfIntoLoop`.
An imperfect nest runs the outer-level statements once per outer iteration, on either
side of the inner loop::

    for i in range(N):
        pre_body(i)              # blocks BEFORE the inner loop
        for j in range(M):
            body(i, j)
        post_body(i)             # blocks AFTER the inner loop

This pass sinks each of those statement groups into the inner loop body, guarded by the
boundary iteration, so the nest becomes perfect and a later interchange / collapse can
expose the outer axis for GPU parallelism::

    for i in range(N):
        for j in range(M):
            if j == <first>: pre_body(i)
            body(i, j)
            if j == <last>:  post_body(i)

It is a **GPU-only** canonicalization. On CPU it is a no-op: guarding the pre/post work by
``j == <boundary>`` and burying it in the inner body destroys the sequential-fusion
locality the outer level otherwise has, so the pass returns ``None`` unless ``target='gpu'``.

Soundness. The pre/post blocks currently run once per ``i`` regardless of the inner trip
count; after the sift they run only from inside the inner body, so the transform is only
value-preserving when:

* **S1 -- provably non-empty inner loop.** If the inner loop can execute zero times, the
  guarded pre/post work would be dropped. The inner trip count must be provably ``>= 1``
  (``sympy`` proof over the parsed init/condition/stride); otherwise the pass refuses.
* **S7 -- outer-axis independence.** The sift only pays off if the outer ``i`` axis is
  parallelizable after perfect nesting. If a container is written at ``i`` and read/written
  at ``i +/- c`` (``c != 0``) anywhere across pre / body / post, a later interchange would
  break that loop-carried dependence, so the pass refuses. An inner-``j`` carry (a scalar
  reduction accumulator) is fine -- it never indexes the outer iterator.
* A pre/post block whose feeding interstate-edge assignment LHS is reassigned by the inner
  body is refused (its value at the boundary iteration would be ambiguous).

The construction mirrors ``MoveIfIntoLoop``: each statement group is deep-copied into a
single-branch (no-else) :class:`ConditionalBlock` placed as the first (pre) / last (post)
block of the inner body, and the interstate-edge data (conditions **and** assignments) that
fed the moved blocks is re-emitted inside the guarded region so nothing is silently dropped.
"""
import copy
from typing import Any, Dict, List, Optional, Set, Tuple

import sympy

from dace import SDFG, properties, symbolic
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.sdfg.utils import set_nested_sdfg_parent_references
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.analysis.loop_analysis import (get_init_assignment, get_loop_end, get_loop_stride)


def _linear_order(region: ControlFlowRegion) -> Optional[List]:
    """Blocks of ``region`` in order iff it is a plain linear chain (unconditional edges;
    interstate assignments are allowed -- they carry statement prep); else ``None``.

    Mirrors ``move_if_into_loop._linear_order`` so the two passes agree on what a
    siftable chain is.
    """
    blocks = list(region.nodes())
    edges = list(region.edges())
    if not blocks or len(edges) != len(blocks) - 1:
        return None
    for e in edges:
        if e.data.condition.as_string not in ('1', 'True', '(1)'):
            return None
    succ = {e.src: e.dst for e in edges}
    order = [region.start_block]
    while order[-1] in succ:
        order.append(succ[order[-1]])
    return order if len(order) == len(blocks) else None


def _provably_nonempty(loop: LoopRegion) -> bool:
    """S1: prove the inner ``loop`` executes at least once.

    For a positive stride the loop is non-empty iff ``init <= last-reached``; for a negative
    stride iff ``init >= last-reached``. Uses the parsed init/condition/stride and a
    ``sympy`` non-negativity proof; an unknown result (e.g. a free symbolic bound with no
    positivity assumption) is treated as *not* provable and refuses.
    """
    init = get_init_assignment(loop)
    end = get_loop_end(loop)
    stride = get_loop_stride(loop)
    if init is None or end is None or stride is None:
        return False
    s = symbolic.pystr_to_symbolic(stride)
    if s.is_positive:
        diff = symbolic.simplify(symbolic.pystr_to_symbolic(end) - symbolic.pystr_to_symbolic(init))
    elif s.is_negative:
        diff = symbolic.simplify(symbolic.pystr_to_symbolic(init) - symbolic.pystr_to_symbolic(end))
    else:
        return False
    return diff.is_nonnegative is True


def _last_reached_iterate(loop: LoopRegion):
    """The value of the loop variable on the final executed iteration.

    ``get_loop_end`` normalizes the raw bound (``i < a -> a-1``, ``i <= a -> a``); with a
    non-unit stride the last *reached* iterate is ``init + ((end - init) // stride) * stride``
    -- not the raw bound -- so the post-guard fires on the iteration that actually runs.
    """
    init = get_init_assignment(loop)
    end = get_loop_end(loop)
    stride = get_loop_stride(loop)
    if init is None or end is None or stride is None:
        return None
    init_s = symbolic.pystr_to_symbolic(init)
    end_s = symbolic.pystr_to_symbolic(end)
    stride_s = symbolic.pystr_to_symbolic(stride)
    if stride_s == 1:
        return end_s
    return symbolic.simplify(init_s + sympy.floor((end_s - init_s) / stride_s) * stride_s)


def _has_outer_carry(subset, iv: str) -> bool:
    """``True`` iff ``subset`` indexes the outer iterator ``iv`` at a non-zero offset.

    A single-point / range begin or end of the form ``iv`` (offset 0) is per-iteration and
    safe; anything else that references ``iv`` -- ``iv-1``, ``iv+j``, ``2*iv`` -- is a
    potential cross-outer-iteration access, so it is reported (conservatively) as a carry.
    """
    if subset is None:
        return False
    iv_sym = symbolic.pystr_to_symbolic(iv)
    for rb, re_, _ in subset.ndrange():
        for expr in (rb, re_):
            try:
                e = symbolic.pystr_to_symbolic(str(expr))
            except Exception:
                return True  # unparseable -> assume the worst
            if iv_sym in e.free_symbols:
                if symbolic.simplify(e - iv_sym) != 0:
                    return True
    return False


def _outer_axis_independent(outer: LoopRegion, inner: LoopRegion, pre: List[SDFGState], post: List[SDFGState]) -> bool:
    """S7: refuse if the outer axis carries a dependence a later interchange would break.

    Restricted to the outer iterator: any *written* container that is accessed at ``iv +/- c``
    (``c != 0``) anywhere across pre / inner-body / post is a loop-carried dependence on the
    axis this pass is trying to make parallel. Scalar accumulators (no ``iv`` index) are inner
    carries and stay clear of this check.
    """
    iv = str(outer.loop_variable)
    states: List[SDFGState] = list(pre) + list(post) + list(inner.all_states())

    written: Set[str] = set()
    for st in states:
        for n in st.nodes():
            if isinstance(n, nodes.AccessNode) and st.in_degree(n) > 0:
                written.add(n.data)

    for st in states:
        for e in st.edges():
            if e.data is None or e.data.data is None or e.data.data not in written:
                continue
            if _has_outer_carry(e.data.subset, iv):
                return False
    return True


def _body_written_names(inner: LoopRegion) -> Set[str]:
    """Symbols (interstate-edge LHS) and data containers written inside ``inner``'s body."""
    names: Set[str] = set()
    for e in inner.all_interstate_edges(recursive=True):
        names |= set(e.data.assignments.keys())
    for st in inner.all_states():
        for n in st.nodes():
            if isinstance(n, nodes.AccessNode) and st.in_degree(n) > 0:
                names.add(n.data)
    return names


def _match(sdfg: SDFG) -> Optional[Tuple[LoopRegion, LoopRegion, List[SDFGState], List[SDFGState]]]:
    """Find an outer ``LoopRegion`` whose linear body is ``[pre..., one inner loop, post...]``
    and that satisfies every soundness gate.

    :returns: ``(outer, inner, pre, post)`` or ``None``.
    """
    for outer in sdfg.all_control_flow_regions(recursive=True):
        if not isinstance(outer, LoopRegion) or not outer.loop_variable:
            continue
        order = _linear_order(outer)
        if order is None:
            continue
        inner_loops = [b for b in order if isinstance(b, LoopRegion)]
        if len(inner_loops) != 1:
            continue  # need exactly one inner loop
        inner = inner_loops[0]
        idx = order.index(inner)
        pre = order[:idx]
        post = order[idx + 1:]
        if not pre and not post:
            continue  # already perfect
        if any(not isinstance(b, SDFGState) for b in pre + post):
            continue  # pre/post must be plain statements, not loops/conditionals
        if not inner.loop_variable:
            continue
        if not _provably_nonempty(inner):  # S1
            continue
        if get_init_assignment(inner) is None or _last_reached_iterate(inner) is None:
            continue
        if post and len(inner.sink_nodes()) != 1:
            continue  # need a unique tail to append the post-guard after
        if not _outer_axis_independent(outer, inner, pre, post):  # S7
            continue
        # Refuse a sifted interstate assignment whose LHS the body reassigns.
        sifted_lhs: Set[str] = set()
        edge_of = {(e.src, e.dst): e for e in outer.edges()}
        chain = pre + [inner] + post
        for a, b in zip(chain, chain[1:]):
            e = edge_of.get((a, b))
            if e is not None:
                sifted_lhs |= set(e.data.assignments.keys())
        if sifted_lhs & _body_written_names(inner):
            continue
        return outer, inner, pre, post
    return None


def _assemble_region(pairs: List[Tuple[SDFGState, Optional[InterstateEdge]]], label: str) -> ControlFlowRegion:
    """Build a fresh linear ``ControlFlowRegion`` from ``(block, incoming-edge-data)`` pairs.

    The first pair's edge data is ``None`` (it is the region start); every other block is
    joined to its predecessor by the given interstate edge (carrying the preserved condition
    and assignments) or a plain unconditional edge when ``None``.
    """
    region = ControlFlowRegion(label=label)
    for k, (blk, _) in enumerate(pairs):
        region.add_node(blk, is_start_block=(k == 0), ensure_unique_name=True)
    for k in range(1, len(pairs)):
        prev = pairs[k - 1][0]
        blk, edata = pairs[k]
        region.add_edge(prev, blk, edata if edata is not None else InterstateEdge())
    region.start_block = region.node_id(pairs[0][0])
    return region


def _sift(outer: LoopRegion, inner: LoopRegion, pre: List[SDFGState], post: List[SDFGState]) -> None:
    """Sink ``pre`` / ``post`` into ``inner``'s body under boundary guards; make ``outer`` perfect."""
    lv = str(inner.loop_variable)
    first_val = get_init_assignment(inner)
    last_val = _last_reached_iterate(inner)
    pre_pred = f"{lv} == {symbolic.symstr(first_val)}"
    post_pred = f"{lv} == {symbolic.symstr(last_val)}"

    edge_of = {(e.src, e.dst): e for e in outer.edges()}

    pre_if = None
    if pre:
        pre_copies = [copy.deepcopy(b) for b in pre]
        pairs: List[Tuple[SDFGState, Optional[InterstateEdge]]] = [(pre_copies[0], None)]
        for k in range(1, len(pre)):
            e = edge_of[(pre[k - 1], pre[k])]
            pairs.append((pre_copies[k], copy.deepcopy(e.data)))
        # Assignments on the edge that fed the inner loop run once, before the body; re-emit
        # them at the boundary iteration via a trailing empty state inside the guard.
        e_into = edge_of[(pre[-1], inner)]
        if e_into.data.assignments:
            pairs.append((SDFGState('pre_guard_tail'), copy.deepcopy(e_into.data)))
        pre_region = _assemble_region(pairs, 'pre_guard_body')
        pre_if = ConditionalBlock('pre_guard')
        pre_if.add_branch(CodeBlock(pre_pred), pre_region)

    post_if = None
    if post:
        post_copies = [copy.deepcopy(b) for b in post]
        # Assignments on the edge out of the inner loop run once, after the body; re-emit them
        # via a leading empty state inside the guard.
        e_outof = edge_of[(inner, post[0])]
        if e_outof.data.assignments:
            pairs = [(SDFGState('post_guard_lead'), None), (post_copies[0], copy.deepcopy(e_outof.data))]
        else:
            pairs = [(post_copies[0], None)]
        for k in range(1, len(post)):
            e = edge_of[(post[k - 1], post[k])]
            pairs.append((post_copies[k], copy.deepcopy(e.data)))
        post_region = _assemble_region(pairs, 'post_guard_body')
        post_if = ConditionalBlock('post_guard')
        post_if.add_branch(CodeBlock(post_pred), post_region)

    old_start = inner.start_block
    old_sink = inner.sink_nodes()[0] if post else None

    if pre_if is not None:
        inner.add_node(pre_if, is_start_block=True, ensure_unique_name=True)
        inner.add_edge(pre_if, old_start, InterstateEdge())
    if post_if is not None:
        inner.add_node(post_if, ensure_unique_name=True)
        inner.add_edge(old_sink, post_if, InterstateEdge())
    if pre_if is not None:
        # Make the prepended pre-guard the explicit inner-body start (the getter otherwise
        # prefers a unique source node, and a stale start corrupts dominator analysis).
        inner.start_block = inner.node_id(pre_if)

    # Strip the now-sifted blocks from the outer body; it must be left with exactly one child
    # (the inner loop). Reset the outer start block after the removals shift node ids.
    for b in list(pre) + list(post):
        outer.remove_node(b)
    outer.start_block = outer.node_id(inner)


@properties.make_properties
@transformation.explicit_cf_compatible
class SiftStatementsIntoPerfectNest(ppl.Pass):
    """Sift imperfect-nest outer statements into the inner loop under boundary guards (GPU-only).

    See the module docstring for the transform, the soundness gates (S1 non-empty inner loop,
    S7 outer-axis independence, boundary-assignment refusal) and the GPU-only rationale.
    """

    CATEGORY: str = 'Optimization Preparation'

    target = properties.Property(dtype=str,
                                 default='cpu',
                                 choices=['cpu', 'gpu'],
                                 desc="Target policy: 'gpu' sifts to expose the outer axis; 'cpu' is a no-op.")

    def __init__(self, target: str = 'cpu'):
        super().__init__()
        self.target = target

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _: Dict[str, Any]) -> Optional[int]:
        """Sift every qualifying imperfect nest in ``sdfg``.

        :param sdfg: The SDFG to transform in place.
        :returns: The number of nests sifted, or ``None`` if none / not a GPU target.
        """
        if self.target != 'gpu':
            return None
        count = 0
        # Each sift rewrites the CFG (moves/removes blocks), so re-scan from scratch after
        # every application until nothing more matches.
        while True:
            m = _match(sdfg)
            if m is None:
                break
            _sift(*m)
            count += 1
        if count:
            # _sift deep-copies blocks; any nested SDFG carried along keeps a stale parent
            # reference until repaired (mirrors MoveIfIntoLoop.apply_pass).
            set_nested_sdfg_parent_references(sdfg)
        return count or None


__all__ = ['SiftStatementsIntoPerfectNest']
