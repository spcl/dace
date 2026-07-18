# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import copy

import numpy as np

from dace import SDFG, InterstateEdge, Memlet
from dace import dtypes
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


def test_same_named_writer_transient_does_not_collapse():
    """Two states each have an intermediate scalar transient ``T`` written from
    a *different* source subset and read by a per-state tasklet. After fusion,
    the matcher must not merge the two ``T`` AccessNodes into one node (last-
    writer-wins would alias both readers to the same value); the empty memlet
    inserted via ``connections_to_make`` orders the two chains, and the second
    chain's reader sees the second chain's writer.

    Reproduces the compound-nest sibling-write shape (TSVC-style augassigns
    over a shared per-iteration intermediate buffer)."""
    sdfg = SDFG('fuse_same_named_writer_transient')
    sdfg.add_array('arr', [8], dtypes.float64)
    sdfg.add_array('out1', [1], dtypes.float64)
    sdfg.add_array('out2', [1], dtypes.float64)
    sdfg.add_transient('t', [1], dtypes.float64)
    sdfg.add_symbol('k', dtypes.int64)

    s1 = sdfg.add_state('s1', is_start_block=True)
    s2 = sdfg.add_state('s2')
    sdfg.add_edge(s1, s2, InterstateEdge())

    # state1: arr[k] -> t -> out1
    ar1 = s1.add_read('arr')
    tw1 = s1.add_access('t')
    ow1 = s1.add_write('out1')
    cp1 = s1.add_tasklet('cp1', {'i'}, {'o'}, 'o = i')
    rd1 = s1.add_tasklet('rd1', {'i'}, {'o'}, 'o = i')
    s1.add_edge(ar1, None, cp1, 'i', Memlet('arr[k]'))
    s1.add_edge(cp1, 'o', tw1, None, Memlet('t[0]'))
    s1.add_edge(tw1, None, rd1, 'i', Memlet('t[0]'))
    s1.add_edge(rd1, 'o', ow1, None, Memlet('out1[0]'))

    # state2: arr[k+1] -> t -> out2
    ar2 = s2.add_read('arr')
    tw2 = s2.add_access('t')
    ow2 = s2.add_write('out2')
    cp2 = s2.add_tasklet('cp2', {'i'}, {'o'}, 'o = i')
    rd2 = s2.add_tasklet('rd2', {'i'}, {'o'}, 'o = i')
    s2.add_edge(ar2, None, cp2, 'i', Memlet('arr[k+1]'))
    s2.add_edge(cp2, 'o', tw2, None, Memlet('t[0]'))
    s2.add_edge(tw2, None, rd2, 'i', Memlet('t[0]'))
    s2.add_edge(rd2, 'o', ow2, None, Memlet('out2[0]'))
    sdfg.validate()

    arr = np.arange(8, dtype=np.float64) * 1.0  # arr[k] = k
    out1 = np.zeros(1, dtype=np.float64)
    out2 = np.zeros(1, dtype=np.float64)
    _assert_fusion_preserves_semantics(sdfg, 1, arr=arr, out1=out1, out2=out2)


def test_peeled_iterations_then_remainder_map_keep_ordering():
    """Two peeled iterations + a remainder Map: the un-fused interstate
    ordering is load-bearing because the peeled writes alias the Map's
    write range. The first SFE fusion (peeled0 + peeled1) is safe -- their
    writes hit disjoint A subsets, and SFE adds a happens-before empty
    memlet from peeled0's A write to peeled1's B-side source. The second
    SFE fusion (merged + remainder) must be REFUSED: the remainder writes
    ``A[0:N]`` which overlaps the merged state's reads ``A[N-1]`` and
    ``A[N-2]``, and the remainder's write value does not flow from real
    first-state data. Earlier the empty-memlet edge to ``B`` fooled the
    ``flows_from_first`` exemption (``B`` was being counted as a producer
    of first state); the fix excludes empty-edge-only AccessNodes from
    ``first_out_data``. After the fix the merged + remainder fusion is
    refused, leaving the interstate edge to enforce the ordering."""
    N_SYM = 8
    sdfg = SDFG('peel_then_remainder')
    sdfg.add_array('A', [N_SYM], dtypes.float64)
    sdfg.add_array('B', [N_SYM], dtypes.float64)
    sdfg.add_transient('acc0', [1], dtypes.float64)
    sdfg.add_transient('acc1', [1], dtypes.float64)

    s_peel0 = sdfg.add_state('peeled_iter_0', is_start_block=True)
    s_peel1 = sdfg.add_state('peeled_iter_1')
    s_rem = sdfg.add_state('remainder_map')
    sdfg.add_edge(s_peel0, s_peel1, InterstateEdge())
    sdfg.add_edge(s_peel1, s_rem, InterstateEdge())

    def add_peel(state, dst_idx: str, acc_name: str):
        ar = state.add_access('A')
        acc_in = state.add_access(acc_name)
        acc_out = state.add_access(acc_name)
        aw = state.add_access('A')
        plus1 = state.add_tasklet(f'plus1_{dst_idx}', {'_in'}, {'_out'}, '_out = _in + 1.0')
        state.add_edge(ar, None, acc_in, None, Memlet(f'A[{dst_idx}] -> [0]'))
        state.add_edge(acc_in, None, plus1, '_in', Memlet(f'{acc_name}[0]'))
        state.add_edge(plus1, '_out', acc_out, None, Memlet(f'{acc_name}[0]'))
        state.add_edge(acc_out, None, aw, None, Memlet(f'{acc_name}[0] -> [{dst_idx}]'))

    add_peel(s_peel0, f'{N_SYM - 1}', 'acc0')  # A[N-1] += 1
    add_peel(s_peel1, f'{N_SYM - 2}', 'acc1')  # A[N-2] += 1

    # Remainder map: A[i] = B[i] * 2 for i in 0:N (overlaps A[N-1] and A[N-2]).
    br = s_rem.add_access('B')
    aw = s_rem.add_access('A')
    me, mx = s_rem.add_map('rem', {'i': f'0:{N_SYM}'})
    times2 = s_rem.add_tasklet('times2', {'_in'}, {'_out'}, '_out = _in * 2.0')
    s_rem.add_memlet_path(br, me, times2, dst_conn='_in', memlet=Memlet('B[i]'))
    s_rem.add_memlet_path(times2, mx, aw, src_conn='_out', memlet=Memlet('A[i]'))
    sdfg.validate()

    a = np.zeros(N_SYM, dtype=np.float64)
    b = np.arange(N_SYM, dtype=np.float64) + 0.5
    # peeled0+peeled1 fuses; merged+remainder is correctly refused -> 2 states left.
    _assert_fusion_preserves_semantics(sdfg, 2, A=a, B=b)


def test_same_cc_war_exempted_when_value_flows_from_first():
    """A WAR within one connected component is safely fusable when the
    second state's write VALUE flows from data the first state produced.
    Shape (cloudsc/TSVC scan-fission style): first state reads ``B`` and
    writes a transient ``T``; second state reads ``T`` and writes ``B``.
    The second's write to ``B`` is topologically downstream of the second's
    read of ``T``, which is downstream of the first's write of ``T``, so
    the read of ``B`` in first state is naturally ordered before the write
    of ``B`` in second state. Base ``StateFusion`` already accepts this
    shape; ``StateFusionExtended`` previously refused it because its
    extra same-CC WAR check lacked the ``flows_from_first`` exemption."""
    sdfg = SDFG('fuse_same_cc_war_flows_through_transient')
    sdfg.add_array('B', [8], dtypes.float64)
    sdfg.add_transient('T', [8], dtypes.float64)
    sdfg.add_symbol('k', dtypes.int64)

    s1 = sdfg.add_state('produce_T_read_B', is_start_block=True)
    s2 = sdfg.add_state('consume_T_write_B')
    sdfg.add_edge(s1, s2, InterstateEdge())

    # state1: T[k] = B[k] * 2
    b_in = s1.add_read('B')
    t_out = s1.add_write('T')
    t1 = s1.add_tasklet('produce', {'_in'}, {'_out'}, '_out = _in * 2.0')
    s1.add_edge(b_in, None, t1, '_in', Memlet('B[k]'))
    s1.add_edge(t1, '_out', t_out, None, Memlet('T[k]'))

    # state2: B[k] = T[k] + 1
    t_in = s2.add_read('T')
    b_out = s2.add_write('B')
    t2 = s2.add_tasklet('consume', {'_in'}, {'_out'}, '_out = _in + 1.0')
    s2.add_edge(t_in, None, t2, '_in', Memlet('T[k]'))
    s2.add_edge(t2, '_out', b_out, None, Memlet('B[k]'))
    sdfg.validate()

    b = np.arange(8, dtype=np.float64)
    _assert_fusion_preserves_semantics(sdfg, 1, B=b)


def test_peeled_maps_then_remainder_map_fuse_or_refuse_correctly():
    """Each peeled iter contains a parallel Map writing a boundary row of
    ``A`` (``A[0,:] = B[0,:]*2`` then ``A[1,:] = B[1,:]*2``), followed by a
    remainder Map over ``A[i,:] = B[i,:]*2`` for ``i in 2:N``. All writes
    are to disjoint rows of ``A``; the only shared data are reads of
    ``B``. ``StateFusionExtended`` must keep this value-preserving --
    either fusing the chain into one state (the disjoint writes are safe)
    or leaving the interstate edges to order them."""
    N_SYM, M_SYM = 6, 4
    sdfg = SDFG('peel_maps_then_rem')
    sdfg.add_array('A', [N_SYM, M_SYM], dtypes.float64)
    sdfg.add_array('B', [N_SYM, M_SYM], dtypes.float64)

    s_peel0 = sdfg.add_state('peeled_iter_0', is_start_block=True)
    s_peel1 = sdfg.add_state('peeled_iter_1')
    s_rem = sdfg.add_state('remainder_map')
    sdfg.add_edge(s_peel0, s_peel1, InterstateEdge())
    sdfg.add_edge(s_peel1, s_rem, InterstateEdge())

    def add_row_map(state, row_idx: str, j_range: str):
        b_in = state.add_access('B')
        a_out = state.add_access('A')
        me, mx = state.add_map(f'm_{row_idx}', {'j': j_range})
        t = state.add_tasklet(f't_{row_idx}', {'_in'}, {'_out'}, '_out = _in * 2.0')
        state.add_memlet_path(b_in, me, t, dst_conn='_in', memlet=Memlet(f'B[{row_idx}, j]'))
        state.add_memlet_path(t, mx, a_out, src_conn='_out', memlet=Memlet(f'A[{row_idx}, j]'))

    add_row_map(s_peel0, '0', f'0:{M_SYM}')  # peeled iter 0: full row 0
    add_row_map(s_peel1, '1', f'0:{M_SYM}')  # peeled iter 1: full row 1

    b_in = s_rem.add_access('B')
    a_out = s_rem.add_access('A')
    me, mx = s_rem.add_map('rem', {'i': f'2:{N_SYM}', 'j': f'0:{M_SYM}'})
    t = s_rem.add_tasklet('t_rem', {'_in'}, {'_out'}, '_out = _in * 2.0')
    s_rem.add_memlet_path(b_in, me, t, dst_conn='_in', memlet=Memlet('B[i, j]'))
    s_rem.add_memlet_path(t, mx, a_out, src_conn='_out', memlet=Memlet('A[i, j]'))
    sdfg.validate()

    A = np.zeros((N_SYM, M_SYM), dtype=np.float64)
    B = np.arange(N_SYM * M_SYM, dtype=np.float64).reshape(N_SYM, M_SYM) + 0.5
    # Disjoint writes -> fusing to 1 state is the cleanest outcome, but
    # refusing (keeping interstate ordering) would also be value-preserving.
    # Pin value preservation regardless of the topology choice.
    ref_sdfg = copy.deepcopy(sdfg)
    refA = A.copy()
    ref_sdfg(A=refA, B=B.copy())

    fused = copy.deepcopy(sdfg)
    fused.apply_transformations_repeated(StateFusionExtended)
    fused.validate()
    gotA = A.copy()
    fused(A=gotA, B=B.copy())
    assert np.allclose(refA, gotA), f'peeled-Map + remainder-Map diverges: ref={refA} got={gotA}'


def test_post_apply_structural_check_raises_on_orphaned_memlet():
    """The post-apply structural check on ``StateFusionExtended`` must
    raise ``InvalidSDFGEdgeError`` if the merger somehow leaves an edge
    whose ``memlet.data`` does not match either endpoint's AccessNode
    data. Construct a valid 2-state SDFG, fuse it (the apply runs
    cleanly), then PROACTIVELY corrupt a post-fusion edge to simulate
    the historical s118-class bug class, and assert the helper raises."""
    from dace.transformation.interstate import StateFusionExtended as _SFE
    sdfg = SDFG('post_apply_check')
    sdfg.add_array('A', [8], dtypes.float64)
    sdfg.add_array('B', [8], dtypes.float64)
    sdfg.add_symbol('k', dtypes.int64)

    s1 = sdfg.add_state('s1', is_start_block=True)
    s2 = sdfg.add_state('s2')
    sdfg.add_edge(s1, s2, InterstateEdge())

    aw = s1.add_write('A')
    t1 = s1.add_tasklet('w', {}, {'o'}, 'o = 3.0')
    s1.add_edge(t1, 'o', aw, None, Memlet('A[k]'))
    bw = s2.add_write('B')
    t2 = s2.add_tasklet('w2', {}, {'o'}, 'o = 5.0')
    s2.add_edge(t2, 'o', bw, None, Memlet('B[k]'))
    sdfg.validate()

    # Run the fusion -- this is a clean case, no real bug.
    sdfg.apply_transformations_repeated(_SFE)
    fused = next(iter(sdfg.states()))

    # Now PROACTIVELY corrupt: pick an A-write edge and overwrite its
    # ``memlet.data`` with a name that matches neither endpoint. The
    # post-apply check would have raised on this kind of damage if the
    # merger had produced it -- we verify the check helper catches it
    # synthetically.
    target = next((e for e in fused.edges() if e.data and e.data.data == 'A'), None)
    assert target is not None
    target.data.data = 'B'  # Now memlet says ``B`` but dst is ``A``.

    xform = _SFE()
    xform.first_state = fused
    xform.second_state = fused
    try:
        xform._post_apply_check(fused, sdfg)
    except Exception as ex:
        assert 'invalid edge' in str(ex).lower() or 'memlet.data' in str(ex).lower(), \
            f'unexpected exception text: {ex}'
        return
    raise AssertionError('post-apply check failed to flag a mismatched memlet.data')


def test_post_apply_strict_validate_runs_full_sdfg_validate():
    """The ``strict_validate`` knob runs ``sdfg.validate()`` after every
    apply. On a clean fusion this is a no-op; this test pins that the
    knob is wired and a clean fusion still passes when strict is on."""
    from dace.transformation.interstate import StateFusionExtended as _SFE
    sdfg = SDFG('strict_validate_clean')
    sdfg.add_array('A', [8], dtypes.float64)
    sdfg.add_array('B', [8], dtypes.float64)
    sdfg.add_symbol('k', dtypes.int64)

    s1 = sdfg.add_state('s1', is_start_block=True)
    s2 = sdfg.add_state('s2')
    sdfg.add_edge(s1, s2, InterstateEdge())

    aw = s1.add_write('A')
    t1 = s1.add_tasklet('w', {}, {'o'}, 'o = 3.0')
    s1.add_edge(t1, 'o', aw, None, Memlet('A[k]'))
    ar = s2.add_read('A')
    bw = s2.add_write('B')
    tr = s2.add_tasklet('r', {'i'}, {'o'}, 'o = i')
    s2.add_edge(ar, None, tr, 'i', Memlet('A[k]'))
    s2.add_edge(tr, 'o', bw, None, Memlet('B[k]'))
    sdfg.validate()

    sdfg.apply_transformations_repeated(_SFE, options={'strict_validate': True})
    sdfg.validate()
    assert sdfg.number_of_nodes() == 1, 'clean RAW fusion under strict_validate should still produce 1 state'


def test_reused_transient_multiple_producers_refuses_ambiguous_merge():
    """The SECOND state writes a transient ``t`` that ALREADY has two separate
    top-level producer AccessNodes in the FIRST state (canon's reused mean /
    scratch transient picks up one write per component across successive
    fusions). Merging would collapse the aliased instances and cross-bind a
    reader to the wrong producer -- a wrong-value binding that, unlike a missing
    happens-before edge, no ordering edge can repair. StateFusionExtended must
    refuse, keeping the states ordered by the interstate edge.

    Pins the nbody KE miscompile: the reused ``__rdo0`` collapsed to several
    crossed instances and the energy reduction read the wrong component (KE
    0.507 -> 1.536 after this fusion) until this shape was refused."""
    sdfg = SDFG('fuse_reused_transient_multi_producer')
    sdfg.add_array('arr', [8], dtypes.float64)
    sdfg.add_array('out1', [1], dtypes.float64)
    sdfg.add_array('out2', [1], dtypes.float64)
    sdfg.add_transient('t', [2], dtypes.float64)
    sdfg.add_symbol('k', dtypes.int64)

    s1 = sdfg.add_state('s1', is_start_block=True)
    s2 = sdfg.add_state('s2')
    sdfg.add_edge(s1, s2, InterstateEdge())

    # state1: TWO separate producers of t (t[0], t[1]); out1 reads both.
    a = s1.add_read('arr')
    ta = s1.add_access('t')
    tb = s1.add_access('t')
    o1 = s1.add_write('out1')
    w0 = s1.add_tasklet('w0', {'i'}, {'o'}, 'o = i')
    w1 = s1.add_tasklet('w1', {'i'}, {'o'}, 'o = i')
    rd = s1.add_tasklet('rd', {'i0', 'i1'}, {'o'}, 'o = i0 + i1')
    s1.add_edge(a, None, w0, 'i', Memlet('arr[0]'))
    s1.add_edge(w0, 'o', ta, None, Memlet('t[0]'))
    s1.add_edge(a, None, w1, 'i', Memlet('arr[1]'))
    s1.add_edge(w1, 'o', tb, None, Memlet('t[1]'))
    s1.add_edge(ta, None, rd, 'i0', Memlet('t[0]'))
    s1.add_edge(tb, None, rd, 'i1', Memlet('t[1]'))
    s1.add_edge(rd, 'o', o1, None, Memlet('out1[0]'))

    # state2: writes t again and reads it into out2.
    a2 = s2.add_read('arr')
    tc = s2.add_access('t')
    o2 = s2.add_write('out2')
    w2 = s2.add_tasklet('w2', {'i'}, {'o'}, 'o = i')
    r2 = s2.add_tasklet('r2', {'i'}, {'o'}, 'o = i')
    s2.add_edge(a2, None, w2, 'i', Memlet('arr[2]'))
    s2.add_edge(w2, 'o', tc, None, Memlet('t[0]'))
    s2.add_edge(tc, None, r2, 'i', Memlet('t[0]'))
    s2.add_edge(r2, 'o', o2, None, Memlet('out2[0]'))
    sdfg.validate()

    arr = np.arange(8, dtype=np.float64)  # arr[i] = i
    out1 = np.zeros(1, dtype=np.float64)
    out2 = np.zeros(1, dtype=np.float64)
    # Refused (2 states) -> matches the un-fused reference: out1=arr[0]+arr[1]=1, out2=arr[2]=2.
    _assert_fusion_preserves_semantics(sdfg, 2, arr=arr, out1=out1, out2=out2)


if __name__ == '__main__':
    test_extended_fusion()
    test_extended_fusion_refuses_unsafe_write_after_read()
    test_reused_transient_multiple_producers_refuses_ambiguous_merge()
    test_raw_read_after_write()
    test_waw_write_after_write()
    test_war_write_after_read()
    test_rar_read_after_read_no_hazard()
    test_same_named_writer_transient_does_not_collapse()
    test_peeled_iterations_then_remainder_map_keep_ordering()
    test_same_cc_war_exempted_when_value_flows_from_first()
    test_peeled_maps_then_remainder_map_fuse_or_refuse_correctly()
    test_post_apply_structural_check_raises_on_orphaned_memlet()
    test_post_apply_strict_validate_runs_full_sdfg_validate()
