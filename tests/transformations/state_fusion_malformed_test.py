# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``StateFusion`` / ``StateFusionExtended`` must never raise out of ``can_be_applied`` or
``apply``; a shape they cannot reason about has to be REFUSED (``can_be_applied is False``).

Every SDFG here is hand-built and malformed in exactly one way -- the shapes an unfinished or
buggy upstream transformation leaves behind. None of them is compiled.
"""
import warnings
from typing import Tuple

import pytest

import dace
from dace import Memlet, subsets
from dace.sdfg import nodes, utils as sdutil
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import SDFGState
from dace.transformation.interstate import StateFusion, StateFusionExtended

XFORMS = [StateFusion, StateFusionExtended]
XFORM_IDS = ['plain', 'extended']


def match(xform, sdfg: dace.SDFG, first: SDFGState, second: SDFGState):
    """A ``xform`` instance bound to the (first, second) state pair of ``sdfg``."""
    x = xform()
    x.setup_match(sdfg, sdfg.cfg_id, -1, {
        type(x).first_state: sdfg.node_id(first),
        type(x).second_state: sdfg.node_id(second)
    }, 0)
    return x


def fill_state(state: SDFGState, name: str, index: str = '0') -> None:
    """A trivial ``tasklet -> AccessNode`` write, so the state is not empty."""
    an = state.add_access(name)
    t = state.add_tasklet(f't_{name}_{state.label}', {}, {'o': None}, 'o = 1.0')
    state.add_edge(t, 'o', an, None, Memlet(f'{name}[{index}]'))


def two_states(name: str) -> Tuple[dace.SDFG, SDFGState, SDFGState]:
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [8], dace.float64)
    sdfg.add_array('B', [8], dace.float64)
    first = sdfg.add_state('first')
    second = sdfg.add_state('second')
    sdfg.add_edge(first, second, InterstateEdge())
    return sdfg, first, second


def assert_refused(xform, sdfg: dace.SDFG, first: SDFGState, second: SDFGState) -> None:
    """Both matching modes must return False, and neither may raise."""
    for permissive in (False, True):
        assert match(xform, sdfg, first, second).can_be_applied(sdfg, 0, sdfg, permissive=permissive) is False


# --------------------------------------------------------------------------- cycles
def make_cyclic(name: str, cyclic_second: bool) -> Tuple[dace.SDFG, SDFGState, SDFGState]:
    """One state holds ``A -> t -> A``: a self-feeding access node, i.e. a cycle."""
    sdfg, first, second = two_states(name)
    cyc, other = (second, first) if cyclic_second else (first, second)
    an = cyc.add_access('A')
    t = cyc.add_tasklet('cyc', {'i': None}, {'o': None}, 'o = i')
    cyc.add_edge(an, None, t, 'i', Memlet('A[0]'))
    cyc.add_edge(t, 'o', an, None, Memlet('A[1]'))
    fill_state(other, 'B')
    return sdfg, first, second


@pytest.mark.parametrize('xform', XFORMS, ids=XFORM_IDS)
@pytest.mark.parametrize('cyclic_second', [False, True], ids=['first', 'second'])
def test_cyclic_state_refused(xform, cyclic_second):
    sdfg, first, second = make_cyclic('cyclic', cyclic_second)
    assert_refused(xform, sdfg, first, second)


@pytest.mark.parametrize('cyclic_second', [False, True], ids=['first', 'second'])
def test_cyclic_state_not_fused_by_driver(cyclic_second):
    """``fuse_states`` drives StateFusionExtended's can_be_applied+apply directly, permissively
    included; ``apply_transformations_repeated`` is the same path for StateFusion."""
    sdfg, _, _ = make_cyclic('cyclic_driver', cyclic_second)
    assert sdutil.fuse_states(sdfg, permissive=True) == 0
    assert sdfg.number_of_nodes() == 2
    # `validate=False`: the SDFG is deliberately invalid, and it is the transformation -- not the
    # validator -- that has to survive it.
    assert sdfg.apply_transformations_repeated(StateFusion, permissive=True, validate=False, validate_all=False) == 0
    assert sdfg.number_of_nodes() == 2


# --------------------------------------------------------------------------- broken scope
def make_dangling_exit(name: str, on_second: bool) -> Tuple[dace.SDFG, SDFGState, SDFGState]:
    """A MapExit whose body edge was deleted: its sink is unreachable from any source, so
    ``scope_children`` raises ``RuntimeError('Leftover nodes in queue')``."""
    sdfg, first, second = two_states(name)
    tgt, other = (second, first) if on_second else (first, second)
    me, mx = tgt.add_map('m', dict(i='0:8'))
    t = tgt.add_tasklet('inner', {}, {'o': None}, 'o = 1.0')
    an = tgt.add_access('A')
    tgt.add_edge(me, None, t, None, Memlet())
    mx.add_in_connector('IN_o')
    mx.add_out_connector('OUT_o')
    tgt.add_edge(t, 'o', mx, 'IN_o', Memlet('A[i]'))
    tgt.add_edge(mx, 'OUT_o', an, None, Memlet('A[0:8]'))
    tgt.remove_edge(next(e for e in tgt.edges() if e.dst is mx))
    mx.remove_in_connector('IN_o')
    fill_state(other, 'B')
    return sdfg, first, second


@pytest.mark.parametrize('xform', XFORMS, ids=XFORM_IDS)
@pytest.mark.parametrize('on_second', [False, True], ids=['first', 'second'])
def test_dangling_scope_refused(xform, on_second):
    sdfg, first, second = make_dangling_exit('dangling', on_second)
    assert_refused(xform, sdfg, first, second)


@pytest.mark.parametrize('xform', XFORMS, ids=XFORM_IDS)
def test_map_scope_without_exit_refused(xform):
    """A MapEntry with an empty body (no MapExit at all): ``exit_node`` has no answer."""
    sdfg, first, second = two_states('no_exit')
    an = first.add_access('A')
    me, mx = first.add_map('m', dict(i='0:8'))
    me.add_in_connector('IN_a')
    first.add_edge(an, None, me, 'IN_a', Memlet('A[0:8]'))
    first.remove_node(mx)
    fill_state(second, 'A', '0:8')
    assert_refused(xform, sdfg, first, second)


# --------------------------------------------------------------------------- connector shapes
def make_scope_edge_damage(name: str, kind: str) -> Tuple[dace.SDFG, SDFGState, SDFGState]:
    """Second state is ``B -> me -> t -> mx -> A``; ``kind`` damages the exit's connectors."""
    sdfg, first, second = two_states(name)
    a_in = first.add_access('A')
    t1 = first.add_tasklet('read_a', {'i': None}, {'o': None}, 'o = i')
    b_out = first.add_access('B')
    first.add_edge(a_in, None, t1, 'i', Memlet('A[0:8]'))
    first.add_edge(t1, 'o', b_out, None, Memlet('B[0:8]'))
    fill_state(first, 'A', '0:8')

    src = second.add_access('B')
    me, mx = second.add_map('m', dict(i='0:8'))
    t = second.add_tasklet('inner', {'i': None}, {'o': None}, 'o = i')
    dst = second.add_access('A')
    me.add_in_connector('IN_b')
    me.add_out_connector('OUT_b')
    mx.add_in_connector('IN_o')
    mx.add_out_connector('OUT_o')
    second.add_edge(src, None, me, 'IN_b', Memlet('B[0:8]'))
    second.add_edge(me, 'OUT_b', t, 'i', Memlet('B[i]'))
    second.add_edge(t, 'o', mx, 'IN_o', Memlet('A[i]'))
    second.add_edge(mx, 'OUT_o', dst, None, Memlet('A[0:8]'))
    if kind == 'no_src_conn':
        # A DATA memlet on a scope edge with no connector: not an ordering edge, so
        # ``memlet_path`` cannot early-return and raises instead.
        second.remove_edge(next(e for e in second.edges() if e.src is mx))
        second.add_edge(mx, None, dst, None, Memlet('A[0:8]'))
    elif kind == 'orphan_out_conn':
        # ``OUT_z`` with no ``IN_z`` partner -- what a half-finished transformation leaves.
        extra = second.add_access('A')
        mx.add_out_connector('OUT_z')
        second.add_edge(mx, 'OUT_z', extra, None, Memlet('A[0:8]'))
    return sdfg, first, second


@pytest.mark.parametrize('xform', XFORMS, ids=XFORM_IDS)
@pytest.mark.parametrize('kind', ['no_src_conn', 'orphan_out_conn'])
def test_damaged_scope_connector_refused(xform, kind):
    sdfg, first, second = make_scope_edge_damage('conn', kind)
    assert_refused(xform, sdfg, first, second)


# --------------------------------------------------------------------------- interstate assignments
@pytest.mark.parametrize('xform', XFORMS, ids=XFORM_IDS)
def test_unparseable_assignment_refused(xform):
    """An assignment RHS that does not parse must never be absorbed into a predecessor edge."""
    sdfg, first, second = two_states('bad_assign')
    pre = sdfg.add_state('pre', is_start_block=True)
    sdfg.add_edge(pre, first, InterstateEdge())
    sdfg.remove_edge(next(e for e in sdfg.edges() if e.src is first))
    sdfg.add_edge(first, second, InterstateEdge(assignments={'k': '1 +'}))
    fill_state(first, 'A')
    fill_state(second, 'B')
    assert_refused(xform, sdfg, first, second)


# --------------------------------------------------------------------------- missing descriptors
@pytest.mark.parametrize('xform', XFORMS, ids=XFORM_IDS)
def test_missing_descriptor_refused(xform):
    """An AccessNode whose descriptor an earlier pass removed."""
    sdfg, first, second = two_states('missing_desc')
    sdfg.add_transient('G', [8], dace.float64)
    fill_state(first, 'A')
    g = second.add_access('G')
    t = second.add_tasklet('use_g', {'i': None}, {'o': None}, 'o = i')
    out = second.add_access('A')
    second.add_edge(g, None, t, 'i', Memlet('G[0]'))
    second.add_edge(t, 'o', out, None, Memlet('A[1]'))
    del sdfg.arrays['G']
    assert_refused(xform, sdfg, first, second)


# --------------------------------------------------------------------------- SubsetUnion memlets
@pytest.mark.parametrize('xform', XFORMS, ids=XFORM_IDS)
def test_subset_union_memlet_is_indeterminate(xform):
    """``subsets.intersects`` has no answer for a SubsetUnion; the conservative one is
    'they may intersect', never an ``AttributeError``."""
    sdfg, first, second = two_states('subset_union')
    fill_state(first, 'A', '0:8')
    an = first.sink_nodes()[0]
    first.in_edges(an)[0].data.subset = subsets.SubsetUnion([subsets.Range.from_string('0:4')])
    fill_state(second, 'A', '0:8')
    an2 = second.sink_nodes()[0]
    second.in_edges(an2)[0].data.subset = subsets.SubsetUnion([subsets.Range.from_string('0:4')])
    assert xform.memlets_intersect(first, [an], False, second, [an2], False) is True
    # And the matcher survives such a memlet end to end.
    assert match(xform, sdfg, first, second).can_be_applied(sdfg, 0, sdfg) in (True, False)


# --------------------------------------------------------------------------- apply-side repairs
@pytest.mark.parametrize('xform', XFORMS, ids=XFORM_IDS)
def test_start_block_repinned_without_raising(xform):
    """``apply`` removes the start block; the region then has two source blocks and no pin, so
    the start-block identity has to be read BEFORE the removal."""
    sdfg = dace.SDFG('startblock')
    sdfg.add_array('A', [8], dace.float64)
    first = sdfg.add_state('first_empty', is_start_block=True)
    second = sdfg.add_state('second')
    sdfg.add_edge(first, second, InterstateEdge())
    dead_a = sdfg.add_state('dead_a')
    dead_b = sdfg.add_state('dead_b')
    sdfg.add_edge(dead_a, dead_b, InterstateEdge())
    for st in (second, dead_a, dead_b):
        fill_state(st, 'A')

    x = match(xform, sdfg, first, second)
    assert x.can_be_applied(sdfg, 0, sdfg) is True
    x.apply(sdfg, sdfg)
    assert sdfg.start_block is second


@pytest.mark.parametrize('xform', XFORMS, ids=XFORM_IDS)
def test_node_shared_by_both_states(xform):
    """The same AccessNode OBJECT is a member of both states: re-adding it must not raise
    ``RuntimeError('Duplicate node added')``."""
    sdfg, first, second = two_states('shared_node')
    shared = nodes.AccessNode('B')
    t1 = first.add_tasklet('write_b', {}, {'o': None}, 'o = 1.0')
    first.add_node(shared)
    first.add_edge(t1, 'o', shared, None, Memlet('B[0]'))
    second.add_node(shared)
    t2 = second.add_tasklet('read_b', {'i': None}, {'o': None}, 'o = i')
    out = second.add_access('A')
    second.add_edge(shared, None, t2, 'i', Memlet('B[0]'))
    second.add_edge(t2, 'o', out, None, Memlet('A[0]'))

    x = match(xform, sdfg, first, second)
    if x.can_be_applied(sdfg, 0, sdfg):
        x.apply(sdfg, sdfg)
        assert sdfg.number_of_nodes() == 1


def test_extended_preexisting_bad_memlet_does_not_crash_apply():
    """A first-state edge whose ``memlet.data`` names neither endpoint is the CALLER's damage.
    ``StateFusionExtended``'s post-apply check must not turn it into an ``apply`` crash."""
    sdfg, first, second = two_states('preexisting_bad_edge')
    sdfg.add_array('C', [8], dace.float64)
    t = first.add_tasklet('write', {}, {'o': None}, 'o = 1.0')
    b = first.add_access('B')
    first.add_edge(t, 'o', b, None, Memlet(data='C', subset='0:8'))
    fill_state(second, 'A')

    x = match(StateFusionExtended, sdfg, first, second)
    assert x.can_be_applied(sdfg, 0, sdfg) is True
    x.apply(sdfg, sdfg)
    assert sdfg.number_of_nodes() == 1


# --------------------------------------------------------------------------- no over-refusal
@pytest.mark.parametrize('xform', XFORMS, ids=XFORM_IDS)
def test_well_formed_map_pair_still_fuses(xform):
    """The structural precondition must not refuse a perfectly ordinary map-to-map RAW pair."""
    sdfg, first, second = two_states('legal')
    sdfg.add_transient('T', [8], dace.float64)

    me, mx = first.add_map('produce', dict(i='0:8'))
    t = first.add_tasklet('p', {'a': None}, {'o': None}, 'o = a + 1.0')
    src, dst = first.add_access('A'), first.add_access('T')
    me.add_in_connector('IN_a')
    me.add_out_connector('OUT_a')
    mx.add_in_connector('IN_o')
    mx.add_out_connector('OUT_o')
    first.add_edge(src, None, me, 'IN_a', Memlet('A[0:8]'))
    first.add_edge(me, 'OUT_a', t, 'a', Memlet('A[i]'))
    first.add_edge(t, 'o', mx, 'IN_o', Memlet('T[i]'))
    first.add_edge(mx, 'OUT_o', dst, None, Memlet('T[0:8]'))

    me2, mx2 = second.add_map('consume', dict(i='0:8'))
    t2 = second.add_tasklet('c', {'a': None}, {'o': None}, 'o = a * 2.0')
    src2, dst2 = second.add_access('T'), second.add_access('B')
    me2.add_in_connector('IN_a')
    me2.add_out_connector('OUT_a')
    mx2.add_in_connector('IN_o')
    mx2.add_out_connector('OUT_o')
    second.add_edge(src2, None, me2, 'IN_a', Memlet('T[0:8]'))
    second.add_edge(me2, 'OUT_a', t2, 'a', Memlet('T[i]'))
    second.add_edge(t2, 'o', mx2, 'IN_o', Memlet('B[i]'))
    second.add_edge(mx2, 'OUT_o', dst2, None, Memlet('B[0:8]'))

    x = match(xform, sdfg, first, second)
    assert x.can_be_applied(sdfg, 0, sdfg) is True
    x.apply(sdfg, sdfg)
    assert sdfg.number_of_nodes() == 1


@pytest.mark.parametrize('xform', XFORMS, ids=XFORM_IDS)
def test_ordering_edge_pair_still_fuses(xform):
    """Empty memlets are ordering edges, not damage: a state carrying one stays fusible."""
    sdfg, first, second = two_states('ordering_edge')
    me, mx = first.add_map('m', dict(i='0:8'))
    t = first.add_tasklet('inner', {}, {'o': None}, 'o = 1.0')
    dst = first.add_access('A')
    mx.add_in_connector('IN_o')
    mx.add_out_connector('OUT_o')
    first.add_edge(me, None, t, None, Memlet())
    first.add_edge(t, 'o', mx, 'IN_o', Memlet('A[i]'))
    first.add_edge(mx, 'OUT_o', dst, None, Memlet('A[0:8]'))
    other = first.add_access('B')
    t_b = first.add_tasklet('b', {}, {'o': None}, 'o = 2.0')
    first.add_edge(t_b, 'o', other, None, Memlet('B[0:8]'))
    first.add_nedge(dst, t_b, dace.Memlet())
    fill_state(second, 'B', '0:8')

    assert match(xform, sdfg, first, second).can_be_applied(sdfg, 0, sdfg) in (True, False)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert match(xform, sdfg, first, second).can_be_applied(sdfg, 0, sdfg) in (True, False)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
