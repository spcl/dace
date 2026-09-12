# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import copy

import networkx as nx
import numpy as np

from dace import SDFG, InterstateEdge, Memlet
from dace import dtypes
from dace.sdfg import nodes as dnodes
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

    Anti-dependency (false dep -- no data flows read->write): StateFusionExtended fuses
    the two states into one and adds a happens-before edge from the first-state reader of
    ``A`` to the second-state writer, so the read stays ordered before the overwrite and
    ``B`` still sees the OLD ``A[k]``.
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
    # Fused into one state; the happens-before edge keeps read-before-write.
    _assert_fusion_preserves_semantics(sdfg, 1, A=a, B=b)


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


def test_war_fanned_out_reads_then_inplace_write_fuses_with_dep_edges():
    """A read state followed by an in-place write of the same array fuses via
    happens-before edges (write-after-read / anti-dependency).

    ``s1`` reads ``A[k]`` in two tasklets (fanned out to ``B`` and ``C``); the later
    ``s2`` does ``A[k] = A[k] + 1``. StateFusionExtended fuses into one state and adds a
    happens-before edge from EACH first-state ``A`` reader to the second-state ``A``
    write chain, so both pending reads stay ordered before the overwrite. The fan-out is
    finite and every reader is edged, so the fusion is safe and stays correct.
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

    a = np.arange(8, dtype=np.float64) + 1.0
    b = np.zeros(8, dtype=np.float64)
    c = np.zeros(8, dtype=np.float64)
    _assert_fusion_preserves_semantics(sdfg, 1, A=a, B=b, C=c)


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
    """Two peeled iterations + a remainder Map: the peeled writes alias the Map's write
    range, so the ordering is load-bearing. The first SFE fusion (peeled0 + peeled1) is
    safe (disjoint A subsets). The second fusion (merged + remainder) is a WAR+WAW: the
    remainder writes ``A[0:N]`` which overlaps the merged state's reads/writes of
    ``A[N-1]``/``A[N-2]`` and its write value does not flow from first-state data. Rather
    than refuse, SFE adds happens-before edges from the merged state's ``A`` accesses to
    the remainder write chain, ordering them before the overwrite, and fuses the whole
    chain into one state. The result must match the un-fused reference (the remainder
    overwrite wins either way, so ``A`` ends at ``B*2``)."""
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
    # peeled0+peeled1 fuses; merged+remainder now also fuses via WAR/WAW happens-before
    # edges (accesses ordered before the remainder overwrite) -> 1 state, still correct.
    _assert_fusion_preserves_semantics(sdfg, 1, A=a, B=b)


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


def test_war_read_feeds_a_map_then_overwrite():
    """WAR where the first-state read is consumed by a MAP (not a bare tasklet):
    ``B[i] = A[i]*2`` over a map, then ``A[0:8] = 9``. The map's internal reads must all
    happen before the overwrite, so ordering must reach the map, not just the access
    node. ``B`` must hold the OLD ``A``."""
    sdfg = SDFG('war_map_reader')
    sdfg.add_array('A', [8], dtypes.float64)
    sdfg.add_array('B', [8], dtypes.float64)
    s1 = sdfg.add_state('read_map', is_start_block=True)
    s2 = sdfg.add_state('overwrite')
    sdfg.add_edge(s1, s2, InterstateEdge())

    ar = s1.add_access('A')
    bw = s1.add_access('B')
    me, mx = s1.add_map('rd', {'i': '0:8'})
    t = s1.add_tasklet('x2', {'_in'}, {'_out'}, '_out = _in * 2.0')
    s1.add_memlet_path(ar, me, t, dst_conn='_in', memlet=Memlet('A[i]'))
    s1.add_memlet_path(t, mx, bw, src_conn='_out', memlet=Memlet('B[i]'))

    aw = s2.add_access('A')
    me2, mx2 = s2.add_map('wr', {'i': '0:8'})
    t2 = s2.add_tasklet('set9', {}, {'_out'}, '_out = 9.0')
    s2.add_memlet_path(me2, t2, memlet=Memlet())
    s2.add_memlet_path(t2, mx2, aw, src_conn='_out', memlet=Memlet('A[i]'))
    sdfg.validate()

    a = np.arange(8, dtype=np.float64) + 1.0
    b = np.zeros(8, dtype=np.float64)
    _assert_fusion_preserves_semantics(sdfg, 1, A=a, B=b)


def test_war_multi_hop_read_chain_then_overwrite():
    """The A-read is consumed through a CHAIN ``A -> t1 -> mid -> t2 -> B`` before the
    second state overwrites ``A``. The actual read happens at ``t1``; every hop must stay
    ordered before the overwrite so ``B`` reflects the OLD ``A``."""
    sdfg = SDFG('war_multihop')
    sdfg.add_array('A', [8], dtypes.float64)
    sdfg.add_array('B', [8], dtypes.float64)
    sdfg.add_transient('mid', [8], dtypes.float64)
    sdfg.add_symbol('k', dtypes.int64)
    s1 = sdfg.add_state('chain', is_start_block=True)
    s2 = sdfg.add_state('overwrite')
    sdfg.add_edge(s1, s2, InterstateEdge())

    ar = s1.add_read('A')
    md = s1.add_access('mid')
    bw = s1.add_write('B')
    t1 = s1.add_tasklet('h1', {'_in'}, {'_out'}, '_out = _in + 1.0')
    t2 = s1.add_tasklet('h2', {'_in'}, {'_out'}, '_out = _in * 2.0')
    s1.add_edge(ar, None, t1, '_in', Memlet('A[k]'))
    s1.add_edge(t1, '_out', md, None, Memlet('mid[k]'))
    s1.add_edge(md, None, t2, '_in', Memlet('mid[k]'))
    s1.add_edge(t2, '_out', bw, None, Memlet('B[k]'))

    aw = s2.add_write('A')
    tw = s2.add_tasklet('set', {}, {'_out'}, '_out = 99.0')
    s2.add_edge(tw, '_out', aw, None, Memlet('A[k]'))
    sdfg.validate()

    a = np.arange(8, dtype=np.float64) + 1.0
    b = np.zeros(8, dtype=np.float64)
    _assert_fusion_preserves_semantics(sdfg, 1, A=a, B=b)


def test_war_and_waw_on_same_array():
    """First state BOTH reads and writes ``A`` (``A[k] = A[k]+1``, and ``B[k]=A[k]``);
    second state overwrites ``A[k]``. That is a WAR (first read) AND a WAW (first write)
    on the same array; both orderings must hold and the second write must win."""
    sdfg, s1, s2 = _two_state('war_and_waw')
    ar = s1.add_read('A')
    bw = s1.add_write('B')
    tb = s1.add_tasklet('cp', {'_in'}, {'_out'}, '_out = _in')
    s1.add_edge(ar, None, tb, '_in', Memlet('A[k]'))
    s1.add_edge(tb, '_out', bw, None, Memlet('B[k]'))
    aw1 = s1.add_write('A')
    ti = s1.add_tasklet('inc', {'_in'}, {'_out'}, '_out = _in + 1.0')
    s1.add_edge(ar, None, ti, '_in', Memlet('A[k]'))
    s1.add_edge(ti, '_out', aw1, None, Memlet('A[k]'))

    aw2 = s2.add_write('A')
    tw = s2.add_tasklet('set', {}, {'_out'}, '_out = 42.0')
    s2.add_edge(tw, '_out', aw2, None, Memlet('A[k]'))
    sdfg.validate()

    a = np.arange(8, dtype=np.float64) + 1.0
    b = np.zeros(8, dtype=np.float64)
    _assert_fusion_preserves_semantics(sdfg, 1, A=a, B=b)


def test_disjoint_subsets_no_hazard_still_fuses():
    """First reads/writes ``A[0]``, second writes ``A[4]`` -- provably disjoint, so no
    ordering edge is required. Must fuse and stay correct."""
    sdfg = SDFG('disjoint_subsets')
    sdfg.add_array('A', [8], dtypes.float64)
    sdfg.add_array('B', [8], dtypes.float64)
    s1 = sdfg.add_state('lo', is_start_block=True)
    s2 = sdfg.add_state('hi')
    sdfg.add_edge(s1, s2, InterstateEdge())

    ar = s1.add_read('A')
    bw = s1.add_write('B')
    t = s1.add_tasklet('cp', {'_in'}, {'_out'}, '_out = _in')
    s1.add_edge(ar, None, t, '_in', Memlet('A[0]'))
    s1.add_edge(t, '_out', bw, None, Memlet('B[0]'))

    aw = s2.add_write('A')
    tw = s2.add_tasklet('set', {}, {'_out'}, '_out = 7.0')
    s2.add_edge(tw, '_out', aw, None, Memlet('A[4]'))
    sdfg.validate()

    a = np.arange(8, dtype=np.float64) + 1.0
    b = np.zeros(8, dtype=np.float64)
    _assert_fusion_preserves_semantics(sdfg, 1, A=a, B=b)


def test_three_state_chain_war_then_raw():
    """Three states fused in sequence: s1 reads ``A`` -> ``B``; s2 overwrites ``A``
    (WAR vs s1); s3 reads ``A`` -> ``C`` (RAW vs s2). Repeated fusion must keep both
    orderings, so ``B`` holds the OLD ``A`` and ``C`` the NEW one."""
    sdfg = SDFG('three_state_chain')
    for arr in ('A', 'B', 'C'):
        sdfg.add_array(arr, [8], dtypes.float64)
    sdfg.add_symbol('k', dtypes.int64)
    s1 = sdfg.add_state('read', is_start_block=True)
    s2 = sdfg.add_state('overwrite')
    s3 = sdfg.add_state('reread')
    sdfg.add_edge(s1, s2, InterstateEdge())
    sdfg.add_edge(s2, s3, InterstateEdge())

    ar = s1.add_read('A')
    bw = s1.add_write('B')
    t1 = s1.add_tasklet('cp1', {'_in'}, {'_out'}, '_out = _in')
    s1.add_edge(ar, None, t1, '_in', Memlet('A[k]'))
    s1.add_edge(t1, '_out', bw, None, Memlet('B[k]'))

    aw = s2.add_write('A')
    t2 = s2.add_tasklet('set', {}, {'_out'}, '_out = 55.0')
    s2.add_edge(t2, '_out', aw, None, Memlet('A[k]'))

    ar3 = s3.add_read('A')
    cw = s3.add_write('C')
    t3 = s3.add_tasklet('cp2', {'_in'}, {'_out'}, '_out = _in')
    s3.add_edge(ar3, None, t3, '_in', Memlet('A[k]'))
    s3.add_edge(t3, '_out', cw, None, Memlet('C[k]'))
    sdfg.validate()

    a = np.arange(8, dtype=np.float64) + 1.0
    ref = _run(sdfg, 'ref3', A=a, B=np.zeros(8), C=np.zeros(8))
    assert ref['B'][K] == K + 1.0 and ref['C'][K] == 55.0  # sanity of the reference
    _assert_fusion_preserves_semantics(sdfg, 1, A=a, B=np.zeros(8), C=np.zeros(8))


def test_war_survives_downstream_simplify():
    """The reason the WAR happens-before edge exists: it must keep the ordering through
    LATER passes. Fuse the WAR, then run ``simplify()`` (which reorders / re-fuses), and
    the result must still match the untransformed reference. This is the cloudsc failure
    mode -- an unconstrained WAR that a downstream pass reorders."""
    sdfg, s1, s2 = _two_state('war_then_simplify')
    ar = s1.add_read('A')
    bw = s1.add_write('B')
    tr = s1.add_tasklet('r', {'i'}, {'o'}, 'o = i * 3.0')
    s1.add_edge(ar, None, tr, 'i', Memlet('A[k]'))
    s1.add_edge(tr, 'o', bw, None, Memlet('B[k]'))

    aw = s2.add_write('A')
    tw = s2.add_tasklet('w', {}, {'o'}, 'o = 11.0')
    s2.add_edge(tw, 'o', aw, None, Memlet('A[k]'))
    sdfg.validate()

    a = np.arange(8, dtype=np.float64) + 1.0
    ref = _run(sdfg, 'ref', A=a, B=np.zeros(8))

    fused = copy.deepcopy(sdfg)
    fused.apply_transformations_repeated(StateFusionExtended)
    fused.simplify()
    fused.validate()
    out = _run(fused, 'fused_simplified', A=a, B=np.zeros(8))
    for name in ('A', 'B'):
        assert np.allclose(out[name], ref[name], rtol=1e-13, atol=1e-13), \
            f'{name} diverges after fusion+simplify: {out[name]} vs ref {ref[name]}'


def test_war_from_dace_program_matches_reference():
    """Frontend-built (``@dace.program``) WAR fixture -- hand-built SDFGs can miss the
    shapes the frontend actually emits. ``B = A + 1`` then ``A[:] = 9``: ``B`` must hold
    the OLD ``A``. Compare an un-transformed run against a StateFusionExtended-fused
    run."""
    import dace

    @dace.program
    def war_prog(A: dace.float64[8], B: dace.float64[8]):
        B[:] = A[:] + 1.0
        A[:] = 9.0

    base = war_prog.to_sdfg(simplify=False)
    base.validate()

    a0 = np.arange(8, dtype=np.float64) + 1.0
    ref_a, ref_b = a0.copy(), np.zeros(8, dtype=np.float64)
    ref_sdfg = copy.deepcopy(base)
    ref_sdfg.name = 'war_prog_ref'
    ref_sdfg(A=ref_a, B=ref_b)

    fused = copy.deepcopy(base)
    fused.name = 'war_prog_fused'
    fused.apply_transformations_repeated(StateFusionExtended)
    fused.validate()
    got_a, got_b = a0.copy(), np.zeros(8, dtype=np.float64)
    fused(A=got_a, B=got_b)

    assert np.allclose(ref_b, a0 + 1.0), f'reference itself is wrong: {ref_b}'
    assert np.allclose(got_a, ref_a) and np.allclose(got_b, ref_b), \
        f'@dace.program WAR diverges after fusion: A={got_a} vs {ref_a}, B={got_b} vs {ref_b}'


# ---------------------------------------------------------------------------
# Structural pins. A numeric-only assertion is NOT sufficient for an ordering
# hazard: a fusion that forgot its happens-before edge still produces the right
# answer whenever codegen happens to emit the first state's components first.
# These tests assert the ordering PATH exists in the fused state (or that the
# fusion was refused), so an unwired dependency fails loudly.
# ---------------------------------------------------------------------------


def _fuse(sdfg: SDFG):
    """Fuse a private copy; return (fused_sdfg, single_state_or_None)."""
    fused = copy.deepcopy(sdfg)
    fused.apply_transformations_repeated(StateFusionExtended)
    fused.validate()
    states = list(fused.states())
    return fused, (states[0] if len(states) == 1 else None)


def _node_by(state, cls_or_data, want_write=None):
    """Find nodes by AccessNode data name or by node class."""
    out = []
    for n in state.nodes():
        if isinstance(cls_or_data, str):
            if isinstance(n, dnodes.AccessNode) and n.data == cls_or_data:
                if want_write is None or (state.in_degree(n) > 0) == want_write:
                    out.append(n)
        elif isinstance(n, cls_or_data):
            out.append(n)
    return out


def _assert_ordered_before(state, src, dst, what: str):
    assert nx.has_path(state._nx, src, dst), \
        f'{what}: no happens-before path {src} -> {dst} in the fused state (ordering edge missing)'


def test_war_ordering_edge_wired_when_second_source_fans_out():
    """Pins the ``all_nodes_between`` trap: the second-state source ``X`` feeds BOTH the
    hazardous ``A`` write and an unrelated ``C`` write. A reachability probe that bails on
    the first non-matching sink silently drops the ordering edge. Assert the WAR edge is
    really wired (and numerics hold)."""
    sdfg = SDFG('war_fanout_source')
    for arr in ('A', 'B', 'C', 'X'):
        sdfg.add_array(arr, [8], dtypes.float64)
    sdfg.add_symbol('k', dtypes.int64)
    s1 = sdfg.add_state('read_A', is_start_block=True)
    s2 = sdfg.add_state('fanout_write')
    sdfg.add_edge(s1, s2, InterstateEdge())

    ar = s1.add_read('A')
    bw = s1.add_write('B')
    t1 = s1.add_tasklet('rd', {'_in'}, {'_out'}, '_out = _in')
    s1.add_edge(ar, None, t1, '_in', Memlet('A[k]'))
    s1.add_edge(t1, '_out', bw, None, Memlet('B[k]'))

    xr = s2.add_read('X')
    aw = s2.add_write('A')
    cw = s2.add_write('C')
    ta = s2.add_tasklet('wa', {'_in'}, {'_out'}, '_out = _in + 1.0')
    tc = s2.add_tasklet('wc', {'_in'}, {'_out'}, '_out = _in + 2.0')
    s2.add_edge(xr, None, ta, '_in', Memlet('X[k]'))
    s2.add_edge(ta, '_out', aw, None, Memlet('A[k]'))
    s2.add_edge(xr, None, tc, '_in', Memlet('X[k]'))
    s2.add_edge(tc, '_out', cw, None, Memlet('C[k]'))
    sdfg.validate()

    fused, st = _fuse(sdfg)
    if st is not None:  # if it fused, the ordering must be explicit
        reader_consumers = [n for n in st.nodes() if isinstance(n, dnodes.Tasklet) and n.label == 'rd']
        a_writes = _node_by(st, 'A', want_write=True)
        assert reader_consumers and a_writes
        _assert_ordered_before(st, reader_consumers[0], a_writes[0], 'WAR with fanned-out second source')

    a = np.arange(8, dtype=np.float64) + 1.0
    _assert_fusion_preserves_semantics(sdfg,
                                       fused.number_of_nodes(),
                                       A=a,
                                       B=np.zeros(8),
                                       C=np.zeros(8),
                                       X=np.arange(8, dtype=np.float64))


def test_war_read_in_map_does_not_capture_second_state_into_scope():
    """Pins the scope-capture trap: when the first-state read feeds a Map, the ordering
    endpoint must be the map's EXIT, never its ENTRY. An edge leaving a MapEntry is inside
    that scope and drags the second-state subgraph into the map (``scope_dict`` recurses
    through every successor of an EntryNode) -- a silent miscompile ``validate()`` misses.
    Assert no second-state node ends up inside a map scope."""
    sdfg = SDFG('war_map_scope_capture')
    for arr in ('A', 'B', 'X'):
        sdfg.add_array(arr, [8], dtypes.float64)
    s1 = sdfg.add_state('map_read', is_start_block=True)
    s2 = sdfg.add_state('overwrite')
    sdfg.add_edge(s1, s2, InterstateEdge())

    ar = s1.add_access('A')
    bw = s1.add_access('B')
    me, mx = s1.add_map('rd', {'i': '0:8'})
    t = s1.add_tasklet('cp', {'_in'}, {'_out'}, '_out = _in')
    s1.add_memlet_path(ar, me, t, dst_conn='_in', memlet=Memlet('A[i]'))
    s1.add_memlet_path(t, mx, bw, src_conn='_out', memlet=Memlet('B[i]'))

    xr = s2.add_read('X')
    aw = s2.add_write('A')
    tw = s2.add_tasklet('wr', {'_in'}, {'_out'}, '_out = _in * 5.0')
    s2.add_edge(xr, None, tw, '_in', Memlet('X[0]'))
    s2.add_edge(tw, '_out', aw, None, Memlet('A[0]'))
    sdfg.validate()

    fused, st = _fuse(sdfg)
    if st is not None:
        scope = st.scope_dict()
        # The second-state nodes must stay at top level.
        for n in st.nodes():
            if isinstance(n, dnodes.Tasklet) and n.label == 'wr':
                assert scope[n] is None, 'second-state tasklet was captured into the map scope'
            if isinstance(n, dnodes.AccessNode) and n.data == 'X':
                assert scope[n] is None, 'second-state X was captured into the map scope'
        # No edge may leave a scope entry to a node outside that entry's scope.
        for e in st.edges():
            if isinstance(e.src, dnodes.EntryNode):
                assert scope[e.dst] is e.src, \
                    f'edge escapes scope: {e.src} -> {e.dst} (dst scope {scope[e.dst]})'

    a = np.arange(8, dtype=np.float64) + 1.0
    _assert_fusion_preserves_semantics(sdfg,
                                       fused.number_of_nodes(),
                                       A=a,
                                       B=np.zeros(8),
                                       X=np.arange(8, dtype=np.float64) + 3.0)


def test_war_not_exempted_when_read_is_in_a_different_component():
    """The ``value flows from first-produced data`` exemption must be a PATH property. Here
    the first state has TWO components: ``A -> t1 -> B`` (reads A) and ``t0 -> T``. The
    second state writes ``A`` from ``T``. ``T`` does order *itself*, but that path never
    touches the ``A`` reader, so the WAR is real: require either an ordering path or a
    refusal."""
    sdfg = SDFG('war_cross_component_exemption')
    for arr in ('A', 'B'):
        sdfg.add_array(arr, [8], dtypes.float64)
    sdfg.add_transient('T', [8], dtypes.float64)
    s1 = sdfg.add_state('two_ccs', is_start_block=True)
    s2 = sdfg.add_state('write_A_from_T')
    sdfg.add_edge(s1, s2, InterstateEdge())

    ar = s1.add_read('A')
    bw = s1.add_write('B')
    t1 = s1.add_tasklet('rd', {'_in'}, {'_out'}, '_out = _in')
    s1.add_edge(ar, None, t1, '_in', Memlet('A[0:8]'))
    s1.add_edge(t1, '_out', bw, None, Memlet('B[0:8]'))
    tw = s1.add_access('T')
    t0 = s1.add_tasklet('mk', {}, {'_out'}, '_out = 4.0')
    s1.add_edge(t0, '_out', tw, None, Memlet('T[0]'))

    tr = s2.add_read('T')
    aw = s2.add_write('A')
    t2 = s2.add_tasklet('wr', {'_in'}, {'_out'}, '_out = _in + 1.0')
    s2.add_edge(tr, None, t2, '_in', Memlet('T[0]'))
    s2.add_edge(t2, '_out', aw, None, Memlet('A[0]'))
    sdfg.validate()

    fused, st = _fuse(sdfg)
    if st is not None:
        rd = [n for n in st.nodes() if isinstance(n, dnodes.Tasklet) and n.label == 'rd']
        a_writes = _node_by(st, 'A', want_write=True)
        assert rd and a_writes
        _assert_ordered_before(st, rd[0], a_writes[0], 'cross-component WAR must not be exempted')

    a = np.arange(8, dtype=np.float64) + 1.0
    _assert_fusion_preserves_semantics(sdfg, fused.number_of_nodes(), A=a, B=np.zeros(8))


def test_war_not_exempted_when_transient_is_rewritten_by_second_state():
    """Second variant of the exemption trap: the transient ``T`` the write flows from is
    RE-WRITTEN inside the second state (``X -> w1 -> T -> w2 -> A``), so nothing actually
    flows from the first state. A name-only exemption wrongly clears the WAR."""
    sdfg = SDFG('war_second_state_rewrites_transient')
    for arr in ('A', 'B', 'X'):
        sdfg.add_array(arr, [8], dtypes.float64)
    sdfg.add_transient('T', [8], dtypes.float64)
    s1 = sdfg.add_state('read_A_and_make_T', is_start_block=True)
    s2 = sdfg.add_state('rewrite_T_then_A')
    sdfg.add_edge(s1, s2, InterstateEdge())

    ar = s1.add_read('A')
    bw = s1.add_write('B')
    t1 = s1.add_tasklet('rd', {'_in'}, {'_out'}, '_out = _in')
    s1.add_edge(ar, None, t1, '_in', Memlet('A[0:8]'))
    s1.add_edge(t1, '_out', bw, None, Memlet('B[0:8]'))
    tprod = s1.add_access('T')
    p0 = s1.add_tasklet('mk', {}, {'_out'}, '_out = 2.0')
    s1.add_edge(p0, '_out', tprod, None, Memlet('T[0]'))

    xr = s2.add_read('X')
    tmid = s2.add_access('T')
    aw = s2.add_write('A')
    w1 = s2.add_tasklet('w1', {'_in'}, {'_out'}, '_out = _in')
    w2 = s2.add_tasklet('w2', {'_in'}, {'_out'}, '_out = _in + 1.0')
    s2.add_edge(xr, None, w1, '_in', Memlet('X[0]'))
    s2.add_edge(w1, '_out', tmid, None, Memlet('T[0]'))
    s2.add_edge(tmid, None, w2, '_in', Memlet('T[0]'))
    s2.add_edge(w2, '_out', aw, None, Memlet('A[0]'))
    sdfg.validate()

    fused, st = _fuse(sdfg)
    if st is not None:
        rd = [n for n in st.nodes() if isinstance(n, dnodes.Tasklet) and n.label == 'rd']
        a_writes = _node_by(st, 'A', want_write=True)
        assert rd and a_writes
        _assert_ordered_before(st, rd[0], a_writes[0], 'WAR exempted by a second-state-rewritten transient')

    a = np.arange(8, dtype=np.float64) + 1.0
    _assert_fusion_preserves_semantics(sdfg,
                                       fused.number_of_nodes(),
                                       A=a,
                                       B=np.zeros(8),
                                       X=np.arange(8, dtype=np.float64) + 5.0)


def test_war_detected_through_single_sided_copy_memlet():
    """An AccessNode->AccessNode copy written single-sided (``Memlet('Tm[0:8]')`` on
    ``A -> Tm``, naming only the destination) leaves ``src_subset`` unset. Without the
    other-side fallback the first-state READ of ``A`` is never recorded and the WAR is
    missed entirely."""
    sdfg = SDFG('war_single_sided_copy')
    sdfg.add_array('A', [8], dtypes.float64)
    sdfg.add_array('X', [8], dtypes.float64)
    sdfg.add_transient('Tm', [8], dtypes.float64)
    sdfg.add_array('B', [8], dtypes.float64)
    s1 = sdfg.add_state('copy_A', is_start_block=True)
    s2 = sdfg.add_state('overwrite_A')
    sdfg.add_edge(s1, s2, InterstateEdge())

    ar = s1.add_read('A')
    tm = s1.add_access('Tm')
    bw = s1.add_write('B')
    s1.add_edge(ar, None, tm, None, Memlet('Tm[0:8]'))  # single-sided: names the DST only
    tt = s1.add_tasklet('use', {'_in'}, {'_out'}, '_out = _in')
    s1.add_edge(tm, None, tt, '_in', Memlet('Tm[0]'))
    s1.add_edge(tt, '_out', bw, None, Memlet('B[0]'))

    xr = s2.add_read('X')
    aw = s2.add_write('A')
    tw = s2.add_tasklet('wr', {'_in'}, {'_out'}, '_out = _in * 2.0')
    s2.add_edge(xr, None, tw, '_in', Memlet('X[0]'))
    s2.add_edge(tw, '_out', aw, None, Memlet('A[0]'))
    sdfg.validate()

    fused, st = _fuse(sdfg)
    if st is not None:
        tms = _node_by(st, 'Tm', want_write=True)
        a_writes = _node_by(st, 'A', want_write=True)
        assert tms and a_writes
        _assert_ordered_before(st, tms[0], a_writes[0], 'WAR through a single-sided copy memlet')

    a = np.arange(8, dtype=np.float64) + 1.0
    _assert_fusion_preserves_semantics(sdfg,
                                       fused.number_of_nodes(),
                                       A=a,
                                       B=np.zeros(8),
                                       X=np.arange(8, dtype=np.float64) + 7.0)


# ---------------------------------------------------------------------------
# Determinism. The pass reads several decisions out of `set`s -- of AccessNodes, which
# hash by ``id()``, and of data names, whose hash is salted per process. Both iterate in a
# different order on a different run, so the same SDFG used to fuse on one run and not on
# the next (TSVC ``s253``). These tests pin that the answer depends on the SDFG only.
# ---------------------------------------------------------------------------


def _structural_fingerprint(sdfg: SDFG) -> str:
    """Node order, edge order and subsets -- everything a set-order flip would perturb."""
    parts = []
    for state in sdfg.states():
        parts.append(f'STATE {state.label}')
        for idx, node in enumerate(state.nodes()):
            parts.append(f'  N{idx} {type(node).__name__} {node}')
        for edge in state.edges():
            parts.append(f'  E {state.node_id(edge.src)}[{edge.src_conn}] -> '
                         f'{state.node_id(edge.dst)}[{edge.dst_conn}] '
                         f'{"EMPTY" if edge.data.is_empty() else edge.data}')
    return '\n'.join(parts)


def _two_ordering_candidates() -> SDFG:
    """First state: ``d = A + 1``; ``t1 = d * 2``; ``t2 = d * 3``. Second: ``d = t1 - 1``;
    ``E = t2 / 2``.

    Both ``t1`` and ``t2`` are match nodes (written by the first state, read by the second),
    and the write-write hazard on ``d`` reaches both of them in the first state. But only
    ``t1`` carries the ordering through in the second state: ``t1 -> d`` exists there,
    ``t2 -> d`` does not. So the two candidates give OPPOSITE verdicts, and picking "the
    first one" means picking whichever the set yielded -- the ``s253`` failure mode.
    ``t1`` proves the two writes of ``d`` are ordered, so the correct answer is to fuse.
    """
    sdfg = SDFG('two_ordering_candidates')
    for arr in ('A', 'd', 't1', 't2', 'E'):
        sdfg.add_array(arr, [8], dtypes.float64)
    s1 = sdfg.add_state('first', is_start_block=True)
    s2 = sdfg.add_state('second')
    sdfg.add_edge(s1, s2, InterstateEdge())

    def chain(state, label, src_node, dst, expr):
        entry, exit_ = state.add_map(label, {'i': '0:8'})
        tasklet = state.add_tasklet(label, {'_in'}, {'_out'}, expr)
        write = state.add_access(dst)
        state.add_memlet_path(src_node, entry, tasklet, dst_conn='_in', memlet=Memlet(f'{src_node.data}[i]'))
        state.add_memlet_path(tasklet, exit_, write, src_conn='_out', memlet=Memlet(f'{dst}[i]'))
        return write

    dw = chain(s1, 'mk_d', s1.add_access('A'), 'd', '_out = _in + 1.0')
    chain(s1, 'mk_t1', dw, 't1', '_out = _in * 2.0')
    chain(s1, 'mk_t2', dw, 't2', '_out = _in * 3.0')
    chain(s2, 'use_t1', s2.add_access('t1'), 'd', '_out = _in - 1.0')
    chain(s2, 'use_t2', s2.add_access('t2'), 'E', '_out = _in / 2.0')
    sdfg.validate()
    return sdfg


def test_ordering_is_proven_by_any_candidate_not_just_the_first():
    """One match node proving the ordering is enough, whichever position it sits in.

    Deciding on the first candidate alone made this SDFG fuse or not fuse depending on the
    process' string-hash seed. ``t1`` proves the two writes to ``d`` are ordered, so the
    states must fuse -- on every run."""
    sdfg = _two_ordering_candidates()
    arrays = {'A': np.arange(8, dtype=np.float64) + 1.0}
    _assert_fusion_preserves_semantics(sdfg, 1, **arrays)


def test_fusion_result_is_reproducible_across_heap_layouts():
    """Deep copies put structurally identical nodes at different addresses, which is exactly
    what makes an ``id()``-keyed set iterate differently between two runs. Every copy has to
    fuse into the same graph, down to node and edge order."""
    base = _two_ordering_candidates()
    fingerprints = set()
    for idx in range(8):
        churn = [object() for _ in range(500 * (idx + 1))]  # move the next allocations around
        work = copy.deepcopy(base)
        work.name = f'{base.name}_{idx}'
        work.apply_transformations_repeated(StateFusionExtended)
        work.validate()
        fingerprints.add(_structural_fingerprint(work))
        del churn
    assert len(fingerprints) == 1, f'StateFusionExtended produced {len(fingerprints)} different results'


def test_happens_before_edges_are_recorded_in_a_stable_order():
    """Several WAR/WAW hazards at once: the edges they produce must be inserted in the same
    order every time, since that order ends up in the fused state and downstream passes read
    it."""

    def build():
        sdfg = SDFG('many_hazards')
        for arr in ('a', 'b', 'c', 'd'):
            sdfg.add_array(arr, [8], dtypes.float64)
        s1 = sdfg.add_state('s1', is_start_block=True)
        s2 = sdfg.add_state('s2')
        sdfg.add_edge(s1, s2, InterstateEdge())
        for state, pairs, expr in ((s1, (('a', 'c'), ('b', 'c'), ('a', 'd')), '_out = _in + 1.0'),
                                   (s2, (('c', 'a'), ('d', 'b'), ('c', 'b')), '_out = _in * 2.0')):
            for idx, (src, dst) in enumerate(pairs):
                read, write = state.add_access(src), state.add_access(dst)
                entry, exit_ = state.add_map(f'm{idx}_{src}{dst}', {'i': '0:8'})
                tasklet = state.add_tasklet(f't{idx}', {'_in'}, {'_out'}, expr)
                state.add_memlet_path(read, entry, tasklet, dst_conn='_in', memlet=Memlet(f'{src}[i]'))
                state.add_memlet_path(tasklet, exit_, write, src_conn='_out', memlet=Memlet(f'{dst}[i]'))
        sdfg.validate()
        return sdfg

    fingerprints = set()
    for idx in range(6):
        churn = [object() for _ in range(400 * (idx + 1))]
        work = build()
        work.name = f'many_hazards_{idx}'
        work.apply_transformations_repeated(StateFusionExtended)
        work.validate()
        fingerprints.add(_structural_fingerprint(work))
        del churn
    assert len(fingerprints) == 1, f'the happens-before edges landed in {len(fingerprints)} different arrangements'


if __name__ == '__main__':
    test_extended_fusion()
    test_ordering_is_proven_by_any_candidate_not_just_the_first()
    test_fusion_result_is_reproducible_across_heap_layouts()
    test_happens_before_edges_are_recorded_in_a_stable_order()
    test_war_fanned_out_reads_then_inplace_write_fuses_with_dep_edges()
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
    test_war_read_feeds_a_map_then_overwrite()
    test_war_multi_hop_read_chain_then_overwrite()
    test_war_and_waw_on_same_array()
    test_disjoint_subsets_no_hazard_still_fuses()
    test_three_state_chain_war_then_raw()
    test_war_survives_downstream_simplify()
    test_war_from_dace_program_matches_reference()
