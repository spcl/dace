# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import copy

import numpy as np

from dace import SDFG, InterstateEdge, Memlet
from dace import dtypes
from dace.sdfg import nodes
from dace.transformation.interstate import StateFusionExtended

# ---------------------------------------------------------------------------
# Classical data hazards across the two fused states (first state -> second):
#   RAW (read-after-write, true dep)   : first writes X, second reads X
#   WAR (write-after-read, anti dep)   : first reads X, second writes X
#   WAW (write-after-write, output dep): first writes X, second writes X
#   RAR (read-after-read, NO hazard)   : both read X
# Whatever StateFusionExtended decides (add a dependency edge, merge nodes, or
# refuse), the fused SDFG must compute exactly what the un-fused one does. These
# tests pin that contract numerically against the un-transformed reference.
# ---------------------------------------------------------------------------

K = 3


def _run(sdfg: SDFG, tag: str, **arrays):
    """Run a private copy of ``sdfg`` on copies of ``arrays``; return the buffers.
    Any declared (non-transient) array not supplied is passed as zeros."""
    s = copy.deepcopy(sdfg)
    s.name = f'{sdfg.name}_{tag}'
    args = {k: v.copy() for k, v in arrays.items()}
    for name, desc in s.arrays.items():
        if not desc.transient and name not in args:
            args[name] = np.zeros([int(d) for d in desc.shape], dtype=np.float64)
    s(**args, k=K)
    return {k: args[k] for k in arrays}


def _assert_fusion_preserves_semantics(sdfg: SDFG, expected_states_after: int, **arrays):
    """Reference (un-fused) vs ``StateFusionExtended`` result must match bit-for-bit."""
    ref = _run(sdfg, 'ref', **arrays)
    fused = copy.deepcopy(sdfg)
    fused.apply_transformations_repeated(StateFusionExtended)
    assert fused.number_of_nodes() == expected_states_after, \
        f'expected {expected_states_after} state(s), got {fused.number_of_nodes()}'
    out = _run(fused, 'fused', **arrays)
    for name in arrays:
        assert np.allclose(out[name], ref[name], rtol=1e-13, atol=1e-13), \
            f'{name} diverges after fusion: {out[name]} vs ref {ref[name]}'


def _two_state(name: str, arrs=('A', 'B', 'C')):
    sdfg = SDFG(name)
    for arr in arrs:
        sdfg.add_array(arr, [8], dtypes.float64)
    sdfg.add_symbol('k', dtypes.int64)
    s1 = sdfg.add_state('s1', is_start_block=True)
    s2 = sdfg.add_state('s2')
    sdfg.add_edge(s1, s2, InterstateEdge())
    return sdfg, s1, s2


def test_raw_read_after_write():
    """first: A[k] = 5;  second: B[k] = A[k].  The read must see the write."""
    sdfg, s1, s2 = _two_state('fuse_raw')
    aw = s1.add_write('A')
    tw = s1.add_tasklet('w', {}, {'o'}, 'o = 5.0')
    s1.add_edge(tw, 'o', aw, None, Memlet('A[k]'))

    ar = s2.add_read('A')
    bw = s2.add_write('B')
    tr = s2.add_tasklet('r', {'i'}, {'o'}, 'o = i')
    s2.add_edge(ar, None, tr, 'i', Memlet('A[k]'))
    s2.add_edge(tr, 'o', bw, None, Memlet('B[k]'))
    sdfg.validate()

    a = np.arange(8, dtype=np.float64)
    b = np.zeros(8, dtype=np.float64)
    _assert_fusion_preserves_semantics(sdfg, 1, A=a, B=b)


def test_waw_write_after_write():
    """first: A[k] = 3;  second: A[k] = 7.  Final value must be 7."""
    sdfg, s1, s2 = _two_state('fuse_waw')
    aw1 = s1.add_write('A')
    t1 = s1.add_tasklet('w1', {}, {'o'}, 'o = 3.0')
    s1.add_edge(t1, 'o', aw1, None, Memlet('A[k]'))

    aw2 = s2.add_write('A')
    t2 = s2.add_tasklet('w2', {}, {'o'}, 'o = 7.0')
    s2.add_edge(t2, 'o', aw2, None, Memlet('A[k]'))
    sdfg.validate()

    a = np.zeros(8, dtype=np.float64)
    _assert_fusion_preserves_semantics(sdfg, 1, A=a)


def test_war_write_after_read():
    """first: B[k] = A[k];  second: A[k] = 9.  B must get the OLD A[k].

    Anti-dependency: the read must be emitted before the overwrite. For now both
    fusions REFUSE this (a correct dependency edge would have to target the read,
    and codegen ordering otherwise risks clobbering the still-pending read); the
    interstate edge then keeps the states ordered. The result must stay correct.
    """
    sdfg, s1, s2 = _two_state('fuse_war')
    ar = s1.add_read('A')
    bw = s1.add_write('B')
    tr = s1.add_tasklet('r', {'i'}, {'o'}, 'o = i')
    s1.add_edge(ar, None, tr, 'i', Memlet('A[k]'))
    s1.add_edge(tr, 'o', bw, None, Memlet('B[k]'))

    aw = s2.add_write('A')
    tw = s2.add_tasklet('w', {}, {'o'}, 'o = 9.0')
    s2.add_edge(tw, 'o', aw, None, Memlet('A[k]'))
    sdfg.validate()

    a = np.arange(8, dtype=np.float64) + 1.0  # A[k] == k+1
    b = np.zeros(8, dtype=np.float64)
    ref = _run(sdfg, 'ref', A=a, B=b)
    assert ref['B'][K] == K + 1.0 and ref['A'][K] == 9.0  # sanity of the reference
    # Refused for now -> the two states stay separate (ordered by the interstate
    # edge), and the result matches the un-fused reference.
    _assert_fusion_preserves_semantics(sdfg, 2, A=a, B=b)


def test_rar_read_after_read_no_hazard():
    """first: B[k] = A[k];  second: C[k] = A[k].  No hazard -> fuse + merge A."""
    sdfg, s1, s2 = _two_state('fuse_rar')
    ar1 = s1.add_read('A')
    bw = s1.add_write('B')
    t1 = s1.add_tasklet('r1', {'i'}, {'o'}, 'o = i')
    s1.add_edge(ar1, None, t1, 'i', Memlet('A[k]'))
    s1.add_edge(t1, 'o', bw, None, Memlet('B[k]'))

    ar2 = s2.add_read('A')
    cw = s2.add_write('C')
    t2 = s2.add_tasklet('r2', {'i'}, {'o'}, 'o = i')
    s2.add_edge(ar2, None, t2, 'i', Memlet('A[k]'))
    s2.add_edge(t2, 'o', cw, None, Memlet('C[k]'))
    sdfg.validate()

    a = np.arange(8, dtype=np.float64) + 0.5
    b = np.zeros(8, dtype=np.float64)
    c = np.zeros(8, dtype=np.float64)
    _assert_fusion_preserves_semantics(sdfg, 1, A=a, B=b, C=c)


def test_extended_fusion():
    """
    Test the extended state fusion transformation.
    It should fuse the two states into one and add a dependency between the two uses of tmp.
    """
    sdfg = SDFG('extended_state_fusion_test')
    sdfg.add_array('A', [20, 20], dtypes.float64)
    sdfg.add_array('B', [20, 20], dtypes.float64)
    sdfg.add_array('C', [20, 20], dtypes.float64)
    sdfg.add_array('D', [20, 20], dtypes.float64)
    sdfg.add_array('E', [20, 20], dtypes.float64)
    sdfg.add_array('F', [20, 20], dtypes.float64)

    sdfg.add_scalar('tmp', dtypes.float64)

    strt = sdfg.add_state("start")
    mid = sdfg.add_state("middle")

    sdfg.add_edge(strt, mid, InterstateEdge())

    acc_a = strt.add_read('A')
    acc_b = strt.add_read('B')
    acc_c = strt.add_write('C')
    acc_tmp = strt.add_access('tmp')

    acc2_d = mid.add_read('D')
    acc2_e = mid.add_read('E')
    acc2_f = mid.add_write('F')
    acc2_tmp = mid.add_access('tmp')

    t1 = strt.add_tasklet('t1', {'a', 'b'}, {
        'c',
    }, 'c = a + b')
    t2 = strt.add_tasklet('t2', {}, {
        'tmpa',
    }, 'tmpa=4')

    t3 = mid.add_tasklet('t3', {'d', 'e'}, {
        'f',
    }, 'f = e + d')
    t4 = mid.add_tasklet('t4', {}, {
        'tmpa',
    }, 'tmpa=7')

    strt.add_edge(acc_a, None, t1, 'a', Memlet.simple('A', '1,1'))
    strt.add_edge(acc_b, None, t1, 'b', Memlet.simple('B', '1,1'))
    strt.add_edge(t1, 'c', acc_c, None, Memlet.simple('C', '1,1'))
    strt.add_edge(t2, 'tmpa', acc_tmp, None, Memlet.simple('tmp', '0'))

    mid.add_edge(acc2_d, None, t3, 'd', Memlet.simple('D', '1,1'))
    mid.add_edge(acc2_e, None, t3, 'e', Memlet.simple('E', '1,1'))
    mid.add_edge(t3, 'f', acc2_f, None, Memlet.simple('F', '1,1'))
    mid.add_edge(t4, 'tmpa', acc2_tmp, None, Memlet.simple('tmp', '0'))
    sdfg.simplify()
    sdfg.apply_transformations_repeated(StateFusionExtended)
    assert sdfg.number_of_nodes() == 1


def test_extended_fusion_refuses_unsafe_write_after_read():
    """A read state followed by an in-place write of the same array must
    not be fused (write-after-read / anti-dependency).

    ``s1`` reads ``A[k]`` in two tasklets; the later ``s2`` does
    ``A[k] = A[k] + 1``. Safely fusing would need a dependency edge to
    every first-state sink reading ``A`` (arbitrarily fanned out), so
    ``StateFusionExtended`` must refuse and leave the two states
    separate; the interstate edge then keeps the write ordered after the
    reads. Previously it fused without that ordering and the write
    clobbered the still-pending reads.
    """
    sdfg = SDFG('state_fusion_war_ordering')
    sdfg.add_array('A', [8], dtypes.float64)
    sdfg.add_array('B', [8], dtypes.float64)
    sdfg.add_array('C', [8], dtypes.float64)
    sdfg.add_symbol('k', dtypes.int64)

    s1 = sdfg.add_state('read_A', is_start_block=True)
    s2 = sdfg.add_state('increment_A')
    sdfg.add_edge(s1, s2, InterstateEdge())

    ar_b = s1.add_read('A')
    bw = s1.add_write('B')
    tb = s1.add_tasklet('rb', {'_in'}, {'_out'}, '_out = _in')
    s1.add_edge(ar_b, None, tb, '_in', Memlet('A[k]'))
    s1.add_edge(tb, '_out', bw, None, Memlet('B[k]'))

    ar_c = s1.add_read('A')
    cw = s1.add_write('C')
    tc = s1.add_tasklet('rc', {'_in'}, {'_out'}, '_out = _in')
    s1.add_edge(ar_c, None, tc, '_in', Memlet('A[k]'))
    s1.add_edge(tc, '_out', cw, None, Memlet('C[k]'))

    ar2 = s2.add_read('A')
    aw2 = s2.add_write('A')
    ti = s2.add_tasklet('inc', {'_in'}, {'_out'}, '_out = _in + 1.0')
    s2.add_edge(ar2, None, ti, '_in', Memlet('A[k]'))
    s2.add_edge(ti, '_out', aw2, None, Memlet('A[k]'))
    sdfg.validate()

    applied = sdfg.apply_transformations_repeated(StateFusionExtended)
    assert applied == 0, 'write-after-read fusion must be refused'
    assert sdfg.number_of_nodes() == 2, 'the two states must remain separate'


if __name__ == '__main__':
    test_extended_fusion()
    test_extended_fusion_refuses_unsafe_write_after_read()
    test_raw_read_after_write()
    test_waw_write_after_write()
    test_war_write_after_read()
    test_rar_read_after_read_no_hazard()
