# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests WCRToAugAssign. """

import dace
import numpy as np
from dace.transformation.dataflow import WCRToAugAssign


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
    xform.setup_match(sdfg, sdfg.cfg_id, sdfg.node_id(state), {
        WCRToAugAssign.inp: state.node_id(vnode),
        WCRToAugAssign.output: state.node_id(hist),
    }, expr_index=2)
    # Must return False without raising (pre-fix this raised the symbolic TypeError).
    assert xform.can_be_applied(state, 2, sdfg) is False


if __name__ == '__main__':
    test_tasklet()
    test_mapped_tasklet()
    test_noncommutative_operand_order()
    test_scalar_source_multidim_target_subset()
    test_slice_source_offset_wcr()
    test_symbolic_overapproximated_wcr_refused_no_typeerror()
