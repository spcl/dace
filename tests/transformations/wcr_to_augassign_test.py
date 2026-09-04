# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests WCRToAugAssign. """

import dace
import numpy as np
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.dataflow import AugAssignToWCR, WCRToAugAssign
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated


def test_tasklet():

    @dace.program
    def test():
        a = np.zeros((10, ))
        for i in dace.map[1:9]:
            a[i - 1] += 1
        return a

    sdfg = test.to_sdfg(simplify=False)
    sdfg.apply_transformations(WCRToAugAssign)

    val = sdfg()
    ref = test.f()
    assert (np.allclose(val, ref))


def test_mapped_tasklet():

    @dace.program
    def test():
        a = np.zeros((10, ))
        for i in dace.map[1:9]:
            a[i - 1] += 1
        return a

    sdfg = test.to_sdfg(simplify=True)
    sdfg.apply_transformations(WCRToAugAssign)

    val = sdfg()
    ref = test.f()
    assert (np.allclose(val, ref))


def test_noncommutative_operand_order():
    """A subtraction WCR ``a[i] = a[i] - v[i]`` must lower to ``old - new``, not
    ``new - old``. Binding the WCR operands by argument name (not body position)
    keeps this correct; the prior position-based wiring silently produced
    ``new - old`` for non-commutative ops. The write is injective (``a[i]``), so
    the conversion's soundness gate allows it inside the parallel Map.
    """
    N = 16
    sdfg = dace.SDFG('wcr_sub_order')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('v', [N], dace.float64)
    state = sdfg.add_state()
    me, mx = state.add_map('m', dict(i=f'0:{N}'))
    tasklet = state.add_tasklet('t', {'inp'}, {'out'}, 'out = inp')
    v_node = state.add_read('v')
    a_node = state.add_write('a')
    state.add_memlet_path(v_node, me, tasklet, dst_conn='inp', memlet=dace.Memlet('v[i]'))
    state.add_memlet_path(tasklet,
                          mx,
                          a_node,
                          src_conn='out',
                          memlet=dace.Memlet(data='a', subset='i', wcr='lambda a, b: a - b'))

    rng = np.random.default_rng(0)
    a0 = rng.random(N)
    v0 = rng.random(N)
    ref = a0 - v0  # a[i] = a[i] - v[i]

    applied = sdfg.apply_transformations(WCRToAugAssign)
    assert applied == 1
    assert all(e.data.wcr is None for s in sdfg.all_states() for e in s.edges())

    a = a0.copy()
    sdfg(a=a, v=v0.copy())
    assert np.allclose(a, ref), f"operand order wrong: got {a[:3]} expected {ref[:3]}"


def test_scalar_source_multidim_target_subset():
    """expr_index-2 (``AccessNode -[wcr]-> AccessNode``): a SCALAR WCR source (the
    per-iteration transient ``NormalizeWCRSource`` inserts) writing a
    MULTI-DIMENSIONAL target element (``aa[2, 3]``). Reverting must read the source
    at its OWN scalar subset, not the target's 2-D slice -- regression for the
    s2101 / s2275 corpus failure ``Memlet subset does not match node dimension
    (expected 1, got 2)``. The soundness check is that the reverted SDFG validates
    and preserves the value (``aa[2, 3] += src``).
    """
    sdfg = dace.SDFG('wcr_scalar_src_md')
    sdfg.add_array('aa', [4, 4], dace.float64)
    sdfg.add_scalar('src', dace.float64, transient=True)
    state = sdfg.add_state()
    producer = state.add_tasklet('produce', {}, {'o'}, 'o = 1.0')
    src = state.add_access('src')
    aa = state.add_write('aa')
    state.add_edge(producer, 'o', src, None, dace.Memlet('src[0]'))
    # WCR source is the scalar ``src``; target is the 2-D element aa[2, 3]
    # (``other_subset`` unset -- the shape NormalizeWCRSource + LoopToMap produce).
    state.add_edge(src, None, aa, None, dace.Memlet(data='aa', subset='2, 3', wcr='lambda a, b: a + b'))
    sdfg.validate()

    applied = sdfg.apply_transformations(WCRToAugAssign)
    assert applied == 1
    sdfg.validate()  # regression: previously raised the dimension mismatch here
    assert all(e.data.wcr is None for s in sdfg.all_states() for e in s.edges()), "WCR must be gone after revert"

    rng = np.random.default_rng(0)
    aa0 = rng.random((4, 4))
    ref = aa0.copy()
    ref[2, 3] += 1.0
    got = aa0.copy()
    sdfg(aa=got)
    assert np.allclose(got, ref), "reverting a scalar-source multidim-target WCR must preserve the value"


def test_slice_source_offset_wcr():
    """expr_index-2 slice WCR ``A[0:n] (wcr+)= B[k:k+n]`` with a SHIFTED source (``k != 0``).
    The reverted elementwise map must read the source at its OWN ``k + i``, not at the
    destination's ``i`` -- regression for the offset bug where the map param indexed both the
    write and the read by the destination's range (reading ``B[0:n]``)."""
    n, k = 6, 2
    sdfg = dace.SDFG('wcr_slice_offset')
    sdfg.add_array('A', [n], dace.float64)
    sdfg.add_array('B', [n + k], dace.float64)
    state = sdfg.add_state()
    rb = state.add_read('B')
    wa = state.add_write('A')
    state.add_edge(rb, None, wa, None,
                   dace.Memlet(data='A', subset=f'0:{n}', other_subset=f'{k}:{k + n}', wcr='lambda a, b: a + b'))
    sdfg.validate()

    applied = sdfg.apply_transformations(WCRToAugAssign)
    assert applied == 1, 'the slice WCR (matching extent, shifted source) must revert'
    sdfg.validate()
    assert all(e.data.wcr is None for s in sdfg.all_states() for e in s.edges()), 'WCR must be gone after revert'

    rng = np.random.default_rng(1)
    A0 = rng.random(n)
    B = rng.random(n + k)
    ref = A0 + B[k:k + n]
    got = A0.copy()
    sdfg(A=got, B=B)
    assert np.allclose(got, ref), f'A[i] += B[k+i]; got {got}, ref {ref}'


def test_symbolic_overapproximated_wcr_refused_no_typeerror():
    """A WCR write with a SYMBOLIC over-approximated subset (a data-dependent
    scatter: the subset spans ``npt`` elements but the volume is 1) must be
    refused cleanly. Pre-fix the guard ``subset.num_elements() > volume`` raised
    ``TypeError: cannot determine truth value of Relational: npt > 1`` -- the two
    symbolic sizes cannot be bool-coerced by a raw ``>`` -- which the
    pattern-match framework only swallowed to a printed warning. ``can_be_applied``
    now decides the size comparison symbolically and returns ``False`` (keeps the
    WCR) instead of raising (the azimint_hist histogram-accumulator shape).

    Calls ``can_be_applied`` directly so the pre-fix ``TypeError`` would propagate
    (the framework's ``apply_transformations`` wrapper otherwise hides it).
    """
    npt = dace.symbol('npt')
    sdfg = dace.SDFG('wcr_symbolic_overapprox')
    sdfg.add_array('hist', [npt], dace.float64)
    sdfg.add_scalar('v', dace.float64, transient=True)
    state = sdfg.add_state()
    prod = state.add_tasklet('p', {}, {'o'}, 'o = 1.0')
    vnode = state.add_access('v')
    hist = state.add_write('hist')
    state.add_edge(prod, 'o', vnode, None, dace.Memlet('v[0]'))
    # Over-approximated dynamic scatter: subset 0:npt (npt elements), volume 1.
    m = dace.Memlet(data='hist', subset=f'0:{npt}', wcr='lambda a, b: a + b')
    m.volume = 1
    m.dynamic = True
    state.add_edge(vnode, None, hist, None, m)

    # expr_index 2 == ``inp -[wcr]-> output`` (AccessNode -> AccessNode).
    xform = WCRToAugAssign()
    xform.setup_match(sdfg,
                      sdfg.cfg_id,
                      sdfg.node_id(state), {
                          WCRToAugAssign.inp: state.node_id(vnode),
                          WCRToAugAssign.output: state.node_id(hist),
                      },
                      expr_index=2)
    # Must return False without raising (pre-fix this raised the symbolic TypeError).
    assert xform.can_be_applied(state, 2, sdfg) is False


def test_mapexit_wcr_injective_reverts():
    """expr_index-4: WCR stranded on the OUTER ``map_exit -> output`` edge (over-approximated to
    the whole array), while the inner ``tasklet -> map_exit`` edge carries the precise per-iteration
    write with NO WCR. This is the shape ``AugAssignToWCR`` + ``LoopToMap`` leave for an injective
    in-place slice aug-assign whose loop only became a Map late (seidel's ``A[i, 1:-1] +=
    <neighbours>`` after its ``j``-loop is lifted). The write ``A[j]`` is injective over the map
    param and the tasklet already reads ``A[j]`` back, so the WCR is a spurious atomic over a
    conflict-free store and must revert to a plain indexed write."""
    N = 8
    sdfg = dace.SDFG('mapexit_wcr_inj')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    state = sdfg.add_state()
    me, mx = state.add_map('m', dict(j=f'0:{N}'))
    tasklet = state.add_tasklet('augassign', {'__in1', '__in2'}, {'__out'}, '__out = (__in1 + __in2)')
    a_read = state.add_read('A')
    b_read = state.add_read('B')
    a_write = state.add_write('A')
    # tasklet reads the destination element A[j] back (__in1) + the incoming operand B[j] (__in2)
    state.add_memlet_path(a_read, me, tasklet, dst_conn='__in1', memlet=dace.Memlet('A[j]'))
    state.add_memlet_path(b_read, me, tasklet, dst_conn='__in2', memlet=dace.Memlet('B[j]'))
    # inner edge: precise A[j], NO WCR
    mx.add_in_connector('IN_A')
    mx.add_out_connector('OUT_A')
    state.add_edge(tasklet, '__out', mx, 'IN_A', dace.Memlet('A[j]'))
    # outer edge: over-approximated A[0:N], the WCR lives HERE
    state.add_edge(mx, 'OUT_A', a_write, None, dace.Memlet(data='A', subset=f'0:{N}', wcr='lambda a, b: a + b'))
    sdfg.validate()

    applied = sdfg.apply_transformations(WCRToAugAssign)
    assert applied == 1, 'the injective map-exit WCR must revert'
    sdfg.validate()
    assert all(e.data.wcr is None for s in sdfg.all_states() for e in s.edges()), 'WCR must be gone after revert'

    rng = np.random.default_rng(3)
    A0 = rng.random(N)
    B = rng.random(N)
    ref = A0 + B  # A[j] = A[j] + B[j]
    got = A0.copy()
    sdfg(A=got, B=B)
    assert np.allclose(got, ref), f'A[j] += B[j]; got {got}, ref {ref}'


def test_mapexit_wcr_reduction_kept():
    """expr_index-4 must REFUSE a genuine reduction: a map-exit WCR whose per-iteration write is a
    CONSTANT target (``acc[0]`` does not vary with the map param) is a real cross-lane reduction,
    not an injective store. Reverting to a plain store would introduce a data race, so the
    injectivity gate keeps the WCR (later lowered to an OMP reduction / atomic)."""
    N = 8
    sdfg = dace.SDFG('mapexit_wcr_reduce')
    sdfg.add_array('acc', [1], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    state = sdfg.add_state()
    me, mx = state.add_map('m', dict(j=f'0:{N}'))
    tasklet = state.add_tasklet('augassign', {'__in1', '__in2'}, {'__out'}, '__out = (__in1 + __in2)')
    acc_read = state.add_read('acc')
    b_read = state.add_read('B')
    acc_write = state.add_write('acc')
    state.add_memlet_path(acc_read, me, tasklet, dst_conn='__in1', memlet=dace.Memlet('acc[0]'))
    state.add_memlet_path(b_read, me, tasklet, dst_conn='__in2', memlet=dace.Memlet('B[j]'))
    mx.add_in_connector('IN_A')
    mx.add_out_connector('OUT_A')
    state.add_edge(tasklet, '__out', mx, 'IN_A', dace.Memlet('acc[0]'))
    state.add_edge(mx, 'OUT_A', acc_write, None, dace.Memlet(data='acc', subset='0', wcr='lambda a, b: a + b'))
    sdfg.validate()

    applied = sdfg.apply_transformations(WCRToAugAssign)
    assert applied == 0, 'a constant-target (reduction) map-exit WCR must NOT revert'
    assert any(e.data.wcr is not None for s in sdfg.all_states() for e in s.edges()), 'reduction WCR must be kept'


def _nested_rmw_sdfg(n: int, extra_fold: bool) -> dace.SDFG:
    """``A[j] = A[j] * C[j]`` swept by a LoopRegion inside a NestedSDFG -- the shape a fissioned or
    chunked body has once ``MapToForLoop`` has lowered the sweep, which is where ``AugAssignToWCR``
    sees a FREE tasklet and lifts the in-place RMW. ``extra_fold`` adds a second, genuine reduction
    into ``A[0]`` under the SAME operator, so the array keeps a WCR writer of its own."""
    body = dace.SDFG('body')
    body.add_array('A', [n], dace.float64)
    body.add_array('C', [n], dace.float64)
    loop = LoopRegion('sweep', f'j < {n}', 'j', 'j = 0', 'j = j + 1')
    body.add_node(loop, is_start_block=True)
    sweep = loop.add_state('rmw', is_start_block=True)
    rmw = sweep.add_tasklet('augassign', {'__in1', '__in2'}, {'__out'}, '__out = (__in1 * __in2)')
    sweep.add_edge(sweep.add_read('A'), None, rmw, '__in1', dace.Memlet('A[j]'))
    sweep.add_edge(sweep.add_read('C'), None, rmw, '__in2', dace.Memlet('C[j]'))
    sweep.add_edge(rmw, '__out', sweep.add_write('A'), None, dace.Memlet('A[j]'))
    if extra_fold:
        fold_state = body.add_state('fold')
        body.add_edge(loop, fold_state, dace.InterstateEdge())
        fme, fmx = fold_state.add_map('fold', dict(k=f'0:{n}'))
        fold = fold_state.add_tasklet('fold', {'__in2'}, {'__out'}, '__out = __in2')
        fold_state.add_memlet_path(fold_state.add_read('C'), fme, fold, dst_conn='__in2', memlet=dace.Memlet('C[k]'))
        fold_state.add_memlet_path(fold,
                                   fmx,
                                   fold_state.add_write('A'),
                                   src_conn='__out',
                                   memlet=dace.Memlet(data='A', subset='0', wcr='lambda a, b: a * b'))

    sdfg = dace.SDFG(f'nested_rmw_{"fold" if extra_fold else "plain"}_{n}')
    sdfg.add_array('A', [n], dace.float64)
    sdfg.add_array('C', [n], dace.float64)
    state = sdfg.add_state('outer')
    nsdfg = state.add_nested_sdfg(body, {'A', 'C'}, {'A'})
    state.add_edge(state.add_read('A'), None, nsdfg, 'A', dace.Memlet(f'A[0:{n}]'))
    state.add_edge(state.add_read('C'), None, nsdfg, 'C', dace.Memlet(f'C[0:{n}]'))
    state.add_edge(nsdfg, 'A', state.add_write('A'), None, dace.Memlet(f'A[0:{n}]'))
    sdfg.validate()
    return sdfg


def _boundary_wcrs(sdfg: dace.SDFG, data: str) -> int:
    """WCR edges writing ``data`` in the OUTERMOST SDFG only -- the boundary a nested body's
    reduction is stamped onto, not the in-body edges that carry it."""
    return sum(1 for s in sdfg.states() for e in s.edges()
               if e.data is not None and e.data.wcr is not None and e.data.data == data)


def test_nested_boundary_wcr_is_cleared_on_revert():
    """``AugAssignToWCR`` stamps the reduction on EVERY enclosing nested-SDFG boundary, not only the
    state it rewrites. Reverting the in-body WCR must undo that outward stamp too, or the boundary
    keeps claiming an atomic no producer asks for -- residue memlet propagation erases, and that the
    multi-dim tile vectorizer refuses a whole kernel over (``loose WCR in the region to be tiled``,
    tsvc ``s212_d_single``, whose in-chunk sweep is nested)."""
    n = 16
    sdfg = _nested_rmw_sdfg(n, extra_fold=False)

    assert PatternMatchAndApplyRepeated([AugAssignToWCR()]).apply_pass(sdfg, {}), 'the in-place RMW must lift'
    assert _boundary_wcrs(sdfg, 'A') == 1, 'AugAssignToWCR stamps the enclosing NestedSDFG boundary'

    assert PatternMatchAndApplyRepeated([WCRToAugAssign()]).apply_pass(sdfg, {}), 'the injective WCR must revert'
    sdfg.validate()
    assert _boundary_wcrs(sdfg, 'A') == 0, 'the boundary stamp must go with the WCR it mirrors'
    assert not [
        e for sd in sdfg.all_sdfgs_recursive() for s in sd.states()
        for e in s.edges() if e.data is not None and e.data.wcr is not None
    ], 'no WCR survives the revert'

    rng = np.random.default_rng(7)
    a0, c = rng.random(n), rng.random(n)
    got = a0.copy()
    sdfg(A=got, C=c)
    assert np.allclose(got, a0 * c), f'A[j] *= C[j]; got {got}, ref {a0 * c}'


def test_nested_boundary_wcr_survives_a_second_reducer():
    """The outward clear stops at a level that still has a WCR writer for the array. Here the body
    both RMWs ``A`` elementwise (revertible once the sweep is a loop) and folds ``C`` into ``A[0]``
    across a parallel map (a real cross-lane reduction the injectivity gate keeps). Dropping the
    boundary would turn that fold's atomics into racing stores, so the stamp stays."""
    n = 16
    sdfg = _nested_rmw_sdfg(n, extra_fold=True)

    assert PatternMatchAndApplyRepeated([AugAssignToWCR()]).apply_pass(sdfg, {})
    assert _boundary_wcrs(sdfg, 'A') == 1

    PatternMatchAndApplyRepeated([WCRToAugAssign()]).apply_pass(sdfg, {})
    sdfg.validate()
    inner = [
        e for sd in sdfg.all_sdfgs_recursive() if sd is not sdfg for s in sd.states() for e in s.edges()
        if e.data is not None and e.data.wcr is not None and e.data.data == 'A'
    ]
    assert inner, 'the constant-target fold into A[0] is a real reduction and must keep its WCR'
    assert _boundary_wcrs(sdfg, 'A') == 1, 'the boundary still carries that reduction'

    rng = np.random.default_rng(11)
    a0, c = rng.random(n), rng.random(n)
    ref = a0 * c
    ref[0] *= c.prod()
    got = a0.copy()
    sdfg(A=got, C=c)
    assert np.allclose(got, ref), f'got {got}, ref {ref}'


def tasklet_to_tasklet_data_edges(sdfg: dace.SDFG) -> list:
    """Edges carrying a data memlet straight from one tasklet to another, at every nesting depth."""
    return [(sub.label, st.label, e.src.label, e.dst.label, e.data.data) for sub in sdfg.all_sdfgs_recursive()
            for st in sub.states() for e in st.edges() if isinstance(e.src, nodes.Tasklet)
            and isinstance(e.dst, nodes.Tasklet) and e.data is not None and e.data.data is not None]


def assert_every_memlet_is_carried(sdfg: dace.SDFG) -> None:
    """No data memlet runs tasklet-to-tasklet, and every container a memlet names still exists."""
    stray = tasklet_to_tasklet_data_edges(sdfg)
    assert not stray, f'data memlet between two tasklets: {stray}'
    for sub in sdfg.all_sdfgs_recursive():
        for st in sub.states():
            for e in st.edges():
                if e.data is not None and e.data.data is not None:
                    assert e.data.data in sub.arrays, f'{sub.label}: memlet names a missing container: {e.data.data}'
    sdfg.validate()


def test_augassign_operand_is_routed_through_an_access_node():
    """The materialised RMW reads its incoming operand from an AccessNode, not off the tasklet.

    A bare tasklet-to-tasklet memlet names a container that no AccessNode carries, and every scan that
    counts uses through access nodes then reads the container as dead: LoopToMap took minife's one such
    scalar for loop-unique data and removed the descriptor while the memlet still named it.
    """

    @dace.program
    def rmw():
        a = np.zeros((10, ))
        for i in dace.map[1:9]:
            a[i - 1] += 1
        return a

    sdfg = rmw.to_sdfg(simplify=False)
    assert sdfg.apply_transformations(WCRToAugAssign) == 1
    assert_every_memlet_is_carried(sdfg)
    assert np.allclose(sdfg(), rmw.f())


def test_augassign_operand_at_a_map_exit_is_routed_through_an_access_node():
    """Same invariant for the map-exit branch of ``apply``, which mints its own operand scalar."""
    N = 16
    sdfg = dace.SDFG('wcr_mapexit_operand')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('v', [N], dace.float64)
    state = sdfg.add_state()
    me, mx = state.add_map('m', dict(i=f'0:{N}'))
    tasklet = state.add_tasklet('t', {'inp': None}, {'out': None}, 'out = inp')
    v_node = state.add_read('v')
    a_node = state.add_write('a')
    state.add_memlet_path(v_node, me, tasklet, dst_conn='inp', memlet=dace.Memlet('v[i]'))
    state.add_memlet_path(tasklet,
                          mx,
                          a_node,
                          src_conn='out',
                          memlet=dace.Memlet(data='a', subset='i', wcr='lambda a, b: a - b'))

    assert sdfg.apply_transformations(WCRToAugAssign) == 1
    assert_every_memlet_is_carried(sdfg)

    rng = np.random.default_rng(0)
    a0, v0 = rng.random(N), rng.random(N)
    a = a0.copy()
    sdfg(a=a, v=v0.copy())
    assert np.allclose(a, a0 - v0), f'got {a[:3]}'


if __name__ == '__main__':
    test_tasklet()
    test_mapped_tasklet()
    test_noncommutative_operand_order()
    test_scalar_source_multidim_target_subset()
    test_slice_source_offset_wcr()
    test_symbolic_overapproximated_wcr_refused_no_typeerror()
    test_mapexit_wcr_injective_reverts()
    test_mapexit_wcr_reduction_kept()
    test_nested_boundary_wcr_is_cleared_on_revert()
    test_nested_boundary_wcr_survives_a_second_reducer()
    test_augassign_operand_is_routed_through_an_access_node()
    test_augassign_operand_at_a_map_exit_is_routed_through_an_access_node()
