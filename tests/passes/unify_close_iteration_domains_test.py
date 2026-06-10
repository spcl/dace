# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`UnifyCloseIterationDomains` and its pass form.

The transformation makes two maps in a vertical ``MapExit -> AccessNode -> MapEntry``
chain share an identical iteration range by extending each to their per-dim union and
guarding the bodies with the original (smaller) range. After applying, the existing
vertical :class:`MapFusion` can fuse the two maps as if they were already same-range
siblings.
"""
import dace
import pytest
import sympy

from dace.transformation.passes.unify_close_iteration_domains import (UnifyCloseIterationDomains,
                                                                      UnifyCloseIterationDomainsPass, _ranges_are_close,
                                                                      _union_range)


def _build_vertical_chain_close_ranges(end_a_extra: int = 1, end_b_extra: int = 0) -> dace.SDFG:
    """Two CPU-scheduled maps in a vertical producer -> consumer chain.

    ``map_a`` writes ``T`` over ``(0:N + end_a_extra, 0:M)``; ``map_b`` reads ``T``
    over ``(0:N + end_b_extra, 0:M)``. The two ranges differ by a constant in the
    leading dimension, so the unify transformation matches.
    """
    N = dace.symbol('N', dtype=dace.int32)
    M = dace.symbol('M', dtype=dace.int32)

    sdfg = dace.SDFG(f'unify_close_a{end_a_extra}_b{end_b_extra}')
    sdfg.add_array('A', [N + max(end_a_extra, end_b_extra), M], dace.float64)
    sdfg.add_array('B', [N + max(end_a_extra, end_b_extra), M], dace.float64)
    sdfg.add_array('T', [N + max(end_a_extra, end_b_extra), M], dace.float64, transient=True)
    state = sdfg.add_state('s')

    me_a, mx_a = state.add_map('map_a', dict(i=f'0:N + {end_a_extra}', j='0:M'))
    me_b, mx_b = state.add_map('map_b', dict(i=f'0:N + {end_b_extra}', j='0:M'))

    a_in = state.add_read('A')
    t_intermediate = state.add_access('T')
    b_out = state.add_write('B')

    t_a = state.add_tasklet('write_t', {'_a': dace.float64}, {'_t': dace.float64}, '_t = _a + 1.0')
    state.add_memlet_path(a_in, me_a, t_a, dst_conn='_a', memlet=dace.Memlet('A[i, j]'))
    state.add_memlet_path(t_a, mx_a, t_intermediate, src_conn='_t', memlet=dace.Memlet('T[i, j]'))

    t_b = state.add_tasklet('read_t', {'_t': dace.float64}, {'_b': dace.float64}, '_b = _t * 2.0')
    state.add_memlet_path(t_intermediate, me_b, t_b, dst_conn='_t', memlet=dace.Memlet('T[i, j]'))
    state.add_memlet_path(t_b, mx_b, b_out, src_conn='_b', memlet=dace.Memlet('B[i, j]'))
    return sdfg


def _map_entries(state):
    return [n for n in state.nodes() if isinstance(n, dace.nodes.MapEntry)]


def _conditional_blocks_inside(sdfg: dace.SDFG) -> int:
    """Count ``ConditionalBlock`` nodes across the SDFG hierarchy without double-counting.

    Iterate every nested SDFG once, then iterate the *top-level nodes* of every CFG region
    inside that SDFG; ``all_control_flow_blocks(recursive=True)`` would re-enter nested
    SDFGs that ``all_sdfgs_recursive`` already visits.
    """
    count = 0
    for s in sdfg.all_sdfgs_recursive():
        for cfg in s.all_control_flow_regions():
            for b in cfg.nodes():
                if isinstance(b, dace.sdfg.state.ConditionalBlock):
                    count += 1
    return count


# ----- Range helpers -----


def test_ranges_close_accepts_constant_diff():
    N = dace.symbol('N')
    M = dace.symbol('M')
    range_a = dace.subsets.Range.from_string(f'0:{N} + 1, 0:{M}')
    range_b = dace.subsets.Range.from_string(f'0:{N}, 0:{M}')
    assert _ranges_are_close(range_a, range_b, max_constant_diff=0)
    assert _ranges_are_close(range_a, range_b, max_constant_diff=1)


def test_ranges_close_rejects_when_diff_exceeds_cap():
    N = dace.symbol('N')
    range_a = dace.subsets.Range.from_string(f'0:{N} + 5, 0:{N}')
    range_b = dace.subsets.Range.from_string(f'0:{N}, 0:{N}')
    assert _ranges_are_close(range_a, range_b, max_constant_diff=0)  # constant diff accepted
    assert _ranges_are_close(range_a, range_b, max_constant_diff=5)
    assert not _ranges_are_close(range_a, range_b, max_constant_diff=4)


def test_ranges_close_rejects_non_constant_diff():
    N = dace.symbol('N')
    M = dace.symbol('M')
    range_a = dace.subsets.Range.from_string(f'0:{N}, 0:{M}')
    range_b = dace.subsets.Range.from_string(f'0:{M}, 0:{M}')
    # N - M is symbolic, not a Number -> not close
    assert not _ranges_are_close(range_a, range_b, max_constant_diff=0)


def test_ranges_close_rejects_different_dimcount():
    N = dace.symbol('N')
    range_a = dace.subsets.Range.from_string(f'0:{N}, 0:{N}')
    range_b = dace.subsets.Range.from_string(f'0:{N}')
    assert not _ranges_are_close(range_a, range_b, max_constant_diff=0)


def test_union_range_takes_max_of_ends():
    """DaCe's ``Range.from_string`` parses ``0:K`` as ``(0, K-1, 1)`` (inclusive end), so
    ``0:N+1`` becomes ``(0, N, 1)``. The per-dim union takes ``Max`` of the inclusive
    ends. SymPy reduces ``Max(N, N-1) -> N`` and ``Max(M-1, M+2) -> M+2``."""
    N = dace.symbol('N')
    M = dace.symbol('M')
    range_a = dace.subsets.Range.from_string(f'0:{N} + 1, 0:{M}')
    range_b = dace.subsets.Range.from_string(f'0:{N}, 0:{M} + 3')
    u = _union_range(range_a, range_b)
    assert sympy.simplify(u[0][1] - N) == 0, u[0][1]
    assert sympy.simplify(u[1][1] - (M + 2)) == 0, u[1][1]


# ----- Transformation form -----


def test_transformation_unifies_ranges_and_guards_smaller_body():
    """After applying the transformation, both maps' ranges equal the union; the smaller
    map's body is wrapped in an if-bound-check NSDFG that re-imposes the original range.
    The larger map (whose range is already the union) is left alone."""
    sdfg = _build_vertical_chain_close_ranges(end_a_extra=1, end_b_extra=0)
    state = next(iter(sdfg.states()))
    me_a, me_b = _map_entries(state)
    mx_a = state.exit_node(me_a)
    array = next(n for n in state.nodes() if isinstance(n, dace.nodes.AccessNode) and n.data == 'T')

    xform = UnifyCloseIterationDomains()
    xform.setup_match(
        sdfg, sdfg.cfg_id, state.block_id, {
            UnifyCloseIterationDomains.first_map_exit: state.node_id(mx_a),
            UnifyCloseIterationDomains.array: state.node_id(array),
            UnifyCloseIterationDomains.second_map_entry: state.node_id(me_b),
        }, 0)
    assert xform.can_be_applied(state, 0, sdfg)
    xform.apply(state, sdfg)

    # After applying, both maps have the same range.
    me_a2, me_b2 = _map_entries(state)
    assert str(me_a2.map.range) == str(me_b2.map.range), (str(me_a2.map.range), str(me_b2.map.range))

    # Exactly one bound-check ConditionalBlock was inserted (for the smaller map ``map_b``).
    assert _conditional_blocks_inside(sdfg) == 1, _conditional_blocks_inside(sdfg)

    sdfg.validate()


def test_transformation_refuses_when_ranges_already_identical():
    """Identical ranges -> nothing to unify -> ``can_be_applied`` returns ``False``."""
    sdfg = _build_vertical_chain_close_ranges(end_a_extra=0, end_b_extra=0)
    state = next(iter(sdfg.states()))
    me_a, me_b = _map_entries(state)
    mx_a = state.exit_node(me_a)
    array = next(n for n in state.nodes() if isinstance(n, dace.nodes.AccessNode) and n.data == 'T')

    xform = UnifyCloseIterationDomains()
    xform.setup_match(
        sdfg, sdfg.cfg_id, state.block_id, {
            UnifyCloseIterationDomains.first_map_exit: state.node_id(mx_a),
            UnifyCloseIterationDomains.array: state.node_id(array),
            UnifyCloseIterationDomains.second_map_entry: state.node_id(me_b),
        }, 0)
    assert not xform.can_be_applied(state, 0, sdfg)


def test_transformation_respects_max_constant_diff_cap():
    """With ``max_constant_diff=1`` and ranges differing by 5 the transformation refuses."""
    sdfg = _build_vertical_chain_close_ranges(end_a_extra=5, end_b_extra=0)
    state = next(iter(sdfg.states()))
    me_a, me_b = _map_entries(state)
    mx_a = state.exit_node(me_a)
    array = next(n for n in state.nodes() if isinstance(n, dace.nodes.AccessNode) and n.data == 'T')

    xform = UnifyCloseIterationDomains()
    xform.max_constant_diff = 1
    xform.setup_match(
        sdfg, sdfg.cfg_id, state.block_id, {
            UnifyCloseIterationDomains.first_map_exit: state.node_id(mx_a),
            UnifyCloseIterationDomains.array: state.node_id(array),
            UnifyCloseIterationDomains.second_map_entry: state.node_id(me_b),
        }, 0)
    assert not xform.can_be_applied(state, 0, sdfg)


# ----- Pass form -----


def test_pass_applies_to_one_vertical_pair():
    """The pass walks the SDFG, finds the one matching vertical chain, and unifies it."""
    sdfg = _build_vertical_chain_close_ranges(end_a_extra=1, end_b_extra=0)
    result = UnifyCloseIterationDomainsPass().apply_pass(sdfg, {})
    assert result == 1, result
    state = next(iter(sdfg.states()))
    me_a, me_b = _map_entries(state)
    assert str(me_a.map.range) == str(me_b.map.range)
    sdfg.validate()


def test_pass_noop_when_no_close_pairs():
    """Identical ranges -> the pass is a no-op (returns ``None``)."""
    sdfg = _build_vertical_chain_close_ranges(end_a_extra=0, end_b_extra=0)
    assert UnifyCloseIterationDomainsPass().apply_pass(sdfg, {}) is None


# ----- Combined: unify + MapFusion vertical -----


def test_combined_unify_establishes_map_fusion_precondition():
    """After ``UnifyCloseIterationDomainsPass``, the two maps share an identical
    iteration range -- the precondition that allows vertical :class:`MapFusion` to fuse
    them. Whether MapFusion actually fires depends on the specific intermediate-buffer
    handling (the smaller map's body is now an NSDFG with an if-guard), which is
    MapFusion's own concern; this test verifies only that the precondition holds and
    the post-unify SDFG is structurally sound.
    """
    sdfg = _build_vertical_chain_close_ranges(end_a_extra=1, end_b_extra=0)
    UnifyCloseIterationDomainsPass().apply_pass(sdfg, {})

    state = next(iter(sdfg.states()))
    me_a, me_b = _map_entries(state)
    assert str(me_a.map.range) == str(me_b.map.range), (str(me_a.map.range), str(me_b.map.range))

    # The chain ``map_a -> T -> map_b`` is preserved -- the AccessNode is still the
    # bridge between the two same-range maps.
    t_node = next(n for n in state.nodes() if isinstance(n, dace.nodes.AccessNode) and n.data == 'T')
    pre = {e.src for e in state.in_edges(t_node)}
    post = {e.dst for e in state.out_edges(t_node)}
    assert any(isinstance(p, dace.nodes.MapExit) and state.entry_node(p) is me_a for p in pre)
    assert me_b in post

    sdfg.validate()


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
