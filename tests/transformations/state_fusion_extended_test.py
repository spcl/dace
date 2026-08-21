# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np

from dace import SDFG, InterstateEdge, Memlet
from dace import dtypes
from dace.transformation.interstate import StateFusionExtended


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


def test_extended_fusion_orders_unsafe_write_after_read():
    """A read state followed by an in-place write of the same array fuses WITH a
    happens-before ordering, not by refusing.

    ``s1`` reads ``A[k]`` in two tasklets; the later ``s2`` does ``A[k] = A[k] + 1``.
    Fusing the two states drops the interstate edge that kept the write after the
    reads, so ``StateFusionExtended`` records the write-after-read anti-dependency and
    adds an ordering edge from every first-state ``A`` reader to the second-state write.
    The result is a single fused state that still computes the reference: ``B`` and ``C``
    see the ORIGINAL ``A[k]`` and ``A[k]`` is incremented after. (An earlier version
    refused this fusion outright; ordering it is sound and strictly more capable.)
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
    assert applied >= 1, 'write-after-read fusion should apply with an ordering edge'
    assert sdfg.number_of_nodes() == 1, 'the two states must fuse into one'

    # The ordering edge keeps the write after the reads: B, C get the original A[k]; A is incremented.
    A = np.arange(8, dtype=np.float64) + 10.0
    B = np.zeros(8)
    C = np.zeros(8)
    A0 = A.copy()
    sdfg(A=A, B=B, C=C, k=3)
    assert A[3] == A0[3] + 1.0
    assert B[3] == A0[3]
    assert C[3] == A0[3]


def test_extended_fusion_orders_reused_transient_producer_consumer():
    """Two states that each write AND read the same transient scalar (as loop-unroll clones
    do with reused index/slice buffers) fuse with a happens-before ordering, not a race.

    Each state does ``tmp = A[k]*2; B[k] = tmp`` for a distinct ``k``. Separate states
    serialize the shared ``tmp``; fusing them into one state must order state 1's read of
    ``tmp`` before state 2's write, or both reads see the last write. Regresses the pipeline
    bug where ShortLoopUnroll's local fusion collapsed every element to the last value.
    """
    sdfg = SDFG('state_fusion_reused_transient')
    sdfg.add_array('A', [4], dtypes.float64)
    sdfg.add_array('B', [4], dtypes.float64)
    sdfg.add_scalar('tmp', dtypes.float64, transient=True)

    s1 = sdfg.add_state('c0', is_start_block=True)
    s2 = sdfg.add_state('c1')
    sdfg.add_edge(s1, s2, InterstateEdge())
    for st, idx in ((s1, 0), (s2, 1)):
        r = st.add_read('A')
        tw = st.add_access('tmp')
        w = st.add_write('B')
        t1 = st.add_tasklet(f'load{idx}', {'_i'}, {'_o'}, '_o = _i * 2.0')
        t2 = st.add_tasklet(f'store{idx}', {'_i'}, {'_o'}, '_o = _i')
        st.add_edge(r, None, t1, '_i', Memlet(f'A[{idx}]'))
        st.add_edge(t1, '_o', tw, None, Memlet('tmp[0]'))
        st.add_edge(tw, None, t2, '_i', Memlet('tmp[0]'))
        st.add_edge(t2, '_o', w, None, Memlet(f'B[{idx}]'))
    sdfg.validate()

    sdfg.apply_transformations_repeated(StateFusionExtended)
    A = np.array([10.0, 20.0, 30.0, 40.0])
    B = np.zeros(4)
    sdfg(A=A, B=B)
    assert B[0] == 20.0 and B[1] == 40.0, f'reused-transient reads collapsed: {B}'


if __name__ == '__main__':
    test_extended_fusion()
    test_extended_fusion_orders_unsafe_write_after_read()
    test_extended_fusion_orders_reused_transient_producer_consumer()
