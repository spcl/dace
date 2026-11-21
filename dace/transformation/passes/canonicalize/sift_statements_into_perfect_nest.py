# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Sift outer-level statements into the inner loop to make an imperfect nest perfect (GPU-only)."""

import copy

from dace import SDFG, symbolic
from dace.optionals import required
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.narrowing import as_expr
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SDFGState, is_read, is_written
from dace.subsets import Subset
from dace.transformation.passes.analysis.loop_analysis import (
    get_init_assignment,
    get_loop_end,
    get_loop_stride,
    loop_header_names,
    loop_jumps,
)
from dace.transformation.passes.loop_fission import _linear_blocks
from dace.transformation.passes.move_if_into_loop import provably_nonempty


def _last_reached_iterate(loop: LoopRegion) -> symbolic.SymbolicType | None:
    """The value of the loop variable on the final executed iteration."""
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
    # ``int_floor``, never ``sympy.floor`` over a division: ``simplify`` distributes the floor into a rational and
    # ``sym2cpp`` prints it without the floor, so the guard below compares against a truncated value and the post-body
    # fires mid-loop.
    return symbolic.simplify(init_s + symbolic.int_floor(as_expr(end_s) - as_expr(init_s), stride_s) * stride_s)


def _has_outer_carry(subset: Subset | None, iv: str) -> bool:
    """``True`` iff ``subset`` indexes the outer iterator ``iv`` at a non-zero offset."""
    if subset is None:
        return False
    iv_sym = symbolic.pystr_to_symbolic(iv)
    for rb, re_, _ in subset.ndrange():
        for expr in (rb, re_):
            try:
                e = symbolic.pystr_to_symbolic(str(expr))
            except Exception:
                return True  # unparseable -> assume the worst
            # Equalized: e is reparsed here while iv_sym is minted from the name, so membership by
            # identity misses the dependence and reports the index as independent of the loop.
            e, iv = symbolic.equalize_symbols_across(e, iv_sym)
            if iv in e.free_symbols:
                if symbolic.simplify(as_expr(e) - iv) != 0:
                    return True
    return False


def _outer_axis_independent(outer: LoopRegion, inner: LoopRegion, pre: list[SDFGState], post: list[SDFGState]) -> bool:
    """S7: refuse if the outer axis carries a dependence a later interchange would break."""
    iv = str(outer.loop_variable)
    states: list[SDFGState] = list(pre) + list(post) + list(inner.states())

    written: dict[str, None] = {}
    for st in states:
        for n in st.nodes():
            if isinstance(n, nodes.AccessNode) and st.in_degree(n) > 0:
                written[n.data] = None

    for st in states:
        for e in st.edges():
            if e.data is None or e.data.data is None or e.data.data not in written:
                continue
            if _has_outer_carry(e.data.subset, iv):
                return False
    return True


def _accumulates_across_outer(
    outer: LoopRegion, inner: LoopRegion, pre: list[SDFGState], post: list[SDFGState]
) -> bool:
    """Whether a ``pre`` / ``post`` statement updates a container at elements free of ``outer``'s iterator, reading the
    value an earlier outer iteration left: every outer iteration then updates the same elements, a carry no interchange
    can parallelize."""
    iv = str(outer.loop_variable)
    written: set[str] = set()
    exposed: set[str] = set()
    updated: set[str] = set()
    for group in (pre, None, post):
        if group is None:
            written |= set(_body_written_names(inner))
            continue
        for st in group:
            for n in st.data_nodes():
                if st.in_degree(n) == 0 and st.out_degree(n) > 0 and n.data not in written:
                    exposed.add(n.data)
            for n in st.data_nodes():
                for e in st.in_edges(n):
                    if e.data.is_empty():
                        continue
                    subset = e.data.get_dst_subset(e, st)
                    if subset is not None and iv not in {str(f) for f in subset.free_symbols}:
                        updated.add(n.data)
                if is_written(st, n):
                    written.add(n.data)
    return bool(exposed & updated)


def _carries_statement_data(inner: LoopRegion, pre: list[SDFGState], post: list[SDFGState]) -> bool:
    """Whether ``inner``'s body both reads and writes a container a ``pre`` / ``post`` statement writes."""
    stmt_written = _written_data(pre + post)
    return bool(stmt_written & _read_data(inner.states()) & _written_data(inner.states()))


def _written_data(states) -> set[str]:
    """The containers ``states`` write."""
    return {n.data for st in states for n in st.data_nodes() if is_written(st, n)}


def _read_data(states) -> set[str]:
    """The containers ``states`` read."""
    return {n.data for st in states for n in st.data_nodes() if is_read(st, n)}


def _body_written_names(inner: LoopRegion) -> dict[str, None]:
    """Symbols (interstate-edge LHS) and data containers written inside ``inner``'s body."""
    names: dict[str, None] = {}
    for e in inner.all_interstate_edges(recursive=True):
        names.update(dict.fromkeys(e.data.assignments.keys()))
    for st in inner.states():
        for n in st.nodes():
            if isinstance(n, nodes.AccessNode) and st.in_degree(n) > 0:
                names[n.data] = None
    return names


def _match(sdfg: SDFG) -> tuple[LoopRegion, LoopRegion, list[SDFGState], list[SDFGState]] | None:
    """Find an outer ``LoopRegion`` whose linear body is ``[pre..., one inner loop, post...]`` and that satisfies every
    soundness gate."""
    for outer in sdfg.all_control_flow_regions(recursive=True):
        if not isinstance(outer, LoopRegion) or not outer.loop_variable:
            continue
        order = _linear_blocks(outer, allow_assignments=True)
        if order is None:
            continue
        inner_loops = [b for b in order if isinstance(b, LoopRegion)]
        if len(inner_loops) != 1:
            continue  # need exactly one inner loop
        inner = inner_loops[0]
        idx = order.index(inner)
        pre = order[:idx]
        post = order[idx + 1 :]
        if not pre and not post:
            continue  # already perfect
        if any(not isinstance(b, SDFGState) for b in pre + post):
            continue  # pre/post must be plain statements, not loops/conditionals
        if not inner.loop_variable:
            continue
        if not provably_nonempty(inner) or loop_jumps(inner):  # S1: the guarded post must run
            continue
        if get_init_assignment(inner) is None or _last_reached_iterate(inner) is None:
            continue
        if post and len(inner.sink_nodes()) != 1:
            continue  # need a unique tail to append the post-guard after
        if not _outer_axis_independent(outer, inner, pre, post):  # S7
            continue
        if _accumulates_across_outer(outer, inner, pre, post):
            continue  # the outer axis stays sequential; see the module docstring
        if _carries_statement_data(inner, pre, post):
            continue  # a reduction: lifted as it stands, see the module docstring
        # Refuse a sifted interstate assignment whose LHS the body reassigns.
        sifted_lhs: dict[str, None] = {}
        pre_lhs: set[str] = set()
        edge_of = {(e.src, e.dst): e for e in outer.edges()}
        chain = pre + [inner] + post
        for k, (a, b) in enumerate(zip(chain, chain[1:])):
            e = edge_of.get((a, b))
            if e is not None:
                sifted_lhs.update(dict.fromkeys(e.data.assignments.keys()))
                if k < len(pre):
                    pre_lhs.update(e.data.assignments.keys())
        body_written = _body_written_names(inner)
        if any(s in sifted_lhs for s in body_written):
            continue
        # The inner header runs before any sifted statement, so it must not read what ``pre`` produces
        pre_written = _written_data(pre)
        if loop_header_names(inner) & (pre_written | sifted_lhs.keys() | body_written.keys()):
            continue
        # Guarded, ``pre`` defines its values at the first inner iteration only, and every later one reads them:
        # a carry that leaves the inner loop sequential, where the outer loop maps around the unsifted nest.
        if pre_written & _read_data(inner.states()) or (pre_written | pre_lhs) & inner.free_symbols:
            continue
        return outer, inner, pre, post
    return None


def _assemble_region(pairs: list[tuple[SDFGState, InterstateEdge | None]], label: str) -> ControlFlowRegion:
    """Build a fresh linear ``ControlFlowRegion`` from ``(block, incoming-edge-data)`` pairs."""
    region = ControlFlowRegion(label=label)
    for k, (blk, _) in enumerate(pairs):
        region.add_node(blk, is_start_block=(k == 0), ensure_unique_name=True)
    for k in range(1, len(pairs)):
        prev = pairs[k - 1][0]
        blk, edata = pairs[k]
        region.add_edge(prev, blk, edata if edata is not None else InterstateEdge())
    region.start_block = region.node_id(pairs[0][0])
    return region


def _sift(outer: LoopRegion, inner: LoopRegion, pre: list[SDFGState], post: list[SDFGState]) -> None:
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
        pairs: list[tuple[SDFGState, InterstateEdge | None]] = [(pre_copies[0], None)]
        for k in range(1, len(pre)):
            e = edge_of[(pre[k - 1], pre[k])]
            pairs.append((pre_copies[k], copy.deepcopy(e.data)))
        # Assignments on the edge that fed the inner loop run once, before the body
        e_into = edge_of[(pre[-1], inner)]
        if e_into.data.assignments:
            pairs.append((SDFGState("pre_guard_tail"), copy.deepcopy(e_into.data)))
        pre_region = _assemble_region(pairs, "pre_guard_body")
        pre_if = ConditionalBlock("pre_guard")
        pre_if.add_branch(CodeBlock(pre_pred), pre_region)

    post_if = None
    if post:
        post_copies = [copy.deepcopy(b) for b in post]
        # Assignments on the edge out of the inner loop run once, after the body
        e_outof = edge_of[(inner, post[0])]
        if e_outof.data.assignments:
            pairs = [(SDFGState("post_guard_lead"), None), (post_copies[0], copy.deepcopy(e_outof.data))]
        else:
            pairs = [(post_copies[0], None)]
        for k in range(1, len(post)):
            e = edge_of[(post[k - 1], post[k])]
            pairs.append((post_copies[k], copy.deepcopy(e.data)))
        post_region = _assemble_region(pairs, "post_guard_body")
        post_if = ConditionalBlock("post_guard")
        post_if.add_branch(CodeBlock(post_pred), post_region)

    old_start = inner.start_block
    old_sink = inner.sink_nodes()[0] if post else None

    if pre_if is not None:
        inner.add_node(pre_if, is_start_block=True, ensure_unique_name=True)
        inner.add_edge(pre_if, old_start, InterstateEdge())
    if post_if is not None:
        inner.add_node(post_if, ensure_unique_name=True)
        inner.add_edge(required(old_sink), post_if, InterstateEdge())
    if pre_if is not None:
        # Make the prepended pre-guard the explicit inner-body start (the getter otherwise
        # prefers a unique source node, and a stale start corrupts dominator analysis).
        inner.start_block = inner.node_id(pre_if)

    # Strip the now-sifted blocks from the outer body; it must be left with exactly one child (the inner loop).
    for b in list(pre) + list(post):
        outer.remove_node(b)
    outer.start_block = outer.node_id(inner)


def sift_imperfect_nests(sdfg: SDFG) -> int:
    """Sift every qualifying imperfect nest in ``sdfg``; return how many were sifted.

    :param sdfg: The SDFG to transform in place.
    :returns: The number of nests sifted (0 if none matched).
    """
    count = 0
    # Each sift rewrites the CFG (moves/removes blocks), so re-scan from scratch after
    # every application until nothing more matches.
    while True:
        m = _match(sdfg)
        if m is None:
            break
        # _sift deep-copies blocks; ``add_node`` re-homes every nested SDFG a copy carries.
        _sift(*m)
        count += 1
    return count


__all__ = ["sift_imperfect_nests"]
