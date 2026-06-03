"""Unit tests for CleanAccessNodeToScalarSliceToTaskletPattern.

Covers the two cases the pass distinguishes for the frontend
``A -> A_slice(scalar transient) -> tasklet`` pattern:

- not reused -> the scalar is removed and ``A`` is wired straight into
  the tasklet;
- reused (in another state or an interstate edge) -> the scalar is kept
  but the AccessNode -> AccessNode copy becomes an assignment tasklet
  (no map), so a dtype-mismatched copy compiles via the tasklet's
  implicit cast instead of failing in ``CopyNDDynamic``.
"""
import dace
import numpy as np

from dace.transformation.passes.clean_access_node_to_scalar_slice_to_tasklet_pattern import (
    CleanAccessNodeToScalarSliceToTaskletPattern, )


def test_scalar_slice_removed():
    """``A[5] -> tmp[0] -> tasklet -> B[5]``: tmp is dead elsewhere, so the
    pass removes it and wires ``A`` straight into the tasklet."""
    sdfg = dace.SDFG('test_clean_scalar')
    sdfg.add_array('A', [10], dace.float64)
    sdfg.add_array('tmp', [1], dace.float64, transient=True)
    sdfg.add_array('B', [10], dace.float64)

    state = sdfg.add_state()
    a = state.add_access('A')
    tmp = state.add_access('tmp')
    b = state.add_access('B')
    tasklet = state.add_tasklet('double', {'inp'}, {'out'}, 'out = inp * 2.0')
    state.add_edge(a, None, tmp, None, dace.Memlet('A[5]'))
    state.add_edge(tmp, None, tasklet, 'inp', dace.Memlet('tmp[0]'))
    state.add_edge(tasklet, 'out', b, None, dace.Memlet('B[5]'))

    assert any(n.data == 'tmp' for n in state.data_nodes())
    CleanAccessNodeToScalarSliceToTaskletPattern().apply_pass(sdfg, None)

    assert not any(n.data == 'tmp' for n in state.data_nodes())
    tasklet_node = next(n for n in state.nodes() if isinstance(n, dace.nodes.Tasklet))
    in_edges = state.in_edges(tasklet_node)
    assert len(in_edges) == 1
    assert in_edges[0].data.data == 'A'
    assert str(in_edges[0].data.subset) == '5'

    sdfg.validate()
    A = np.random.rand(10)
    B = np.zeros(10)
    sdfg(A=A, B=B)
    assert np.isclose(B[5], A[5] * 2.0)


def _build_two_state_sdfg():
    """State s0: ``A[5] -> tmp[0] -> t0 -> B[5]``; state s1 reads ``tmp[0]``."""
    sdfg = dace.SDFG('test_reused')
    sdfg.add_array('A', [10], dace.float64)
    sdfg.add_array('tmp', [1], dace.float64, transient=True)
    sdfg.add_array('B', [10], dace.float64)
    sdfg.add_array('C', [10], dace.float64)

    s0 = sdfg.add_state('s0')
    a = s0.add_access('A')
    tmp0 = s0.add_access('tmp')
    b = s0.add_access('B')
    t0 = s0.add_tasklet('double', {'inp'}, {'out'}, 'out = inp * 2.0')
    s0.add_edge(a, None, tmp0, None, dace.Memlet('A[5]'))
    s0.add_edge(tmp0, None, t0, 'inp', dace.Memlet('tmp[0]'))
    s0.add_edge(t0, 'out', b, None, dace.Memlet('B[5]'))

    s1 = sdfg.add_state('s1')
    tmp1 = s1.add_access('tmp')
    c = s1.add_access('C')
    t1 = s1.add_tasklet('triple', {'inp'}, {'out'}, 'out = inp * 3.0')
    s1.add_edge(tmp1, None, t1, 'inp', dace.Memlet('tmp[0]'))
    s1.add_edge(t1, 'out', c, None, dace.Memlet('C[3]'))

    sdfg.add_edge(s0, s1, dace.InterstateEdge())
    return sdfg, s0


def test_scalar_reused_gets_assign_tasklet():
    """When ``tmp`` is read in another state, the pass keeps it and turns
    the ``A -> tmp`` copy into an assignment tasklet (no map)."""
    sdfg, s0 = _build_two_state_sdfg()
    CleanAccessNodeToScalarSliceToTaskletPattern().apply_pass(sdfg, None)

    # tmp kept; the copy is now a `_out = _in` assign tasklet feeding tmp.
    assert any(n.data == 'tmp' for n in s0.data_nodes())
    cast = next(
        (n for n in s0.nodes() if isinstance(n, dace.nodes.Tasklet) and n.label.startswith('_assign_in_A_to_tmp')),
        None)
    assert cast is not None, "expected an assignment tasklet on the A -> tmp copy"
    # No map was introduced.
    assert not any(isinstance(n, dace.nodes.MapEntry) for n in s0.nodes())
    # The assign reads A[5] and writes tmp.
    in_e = s0.in_edges(cast)[0]
    assert in_e.data.data == 'A' and str(in_e.data.subset) == '5'

    sdfg.validate()
    A = np.random.rand(10)
    B = np.zeros(10)
    C = np.zeros(10)
    sdfg(A=A, B=B, C=C)
    assert np.isclose(B[5], A[5] * 2.0)
    assert np.isclose(C[3], A[5] * 3.0)


def test_permissive_removes_even_if_reused():
    """``permissive=True`` removes the scalar regardless of reuse."""
    sdfg, s0 = _build_two_state_sdfg()
    CleanAccessNodeToScalarSliceToTaskletPattern(permissive=True).apply_pass(sdfg, None)
    assert not any(n.data == 'tmp' for n in s0.data_nodes())


def test_dtype_mismatch_not_reused_compiles():
    """The motivating case: a double->float32 ``A -> A_slice`` copy fails
    in CopyNDDynamic. Removing the scalar (not reused) wires the f64 read
    straight into the tasklet, so the cast happens in the tasklet body."""
    sdfg = dace.SDFG('test_dtype_fold')
    sdfg.add_array('A', [10], dace.float64)
    sdfg.add_array('tmp', [1], dace.float32, transient=True)
    sdfg.add_array('B', [10], dace.float32)

    state = sdfg.add_state()
    a = state.add_access('A')
    tmp = state.add_access('tmp')
    b = state.add_access('B')
    t = state.add_tasklet('scale', {'inp'}, {'out'}, 'out = inp * 2.0')
    state.add_edge(a, None, tmp, None, dace.Memlet('A[5]'))
    state.add_edge(tmp, None, t, 'inp', dace.Memlet('tmp[0]'))
    state.add_edge(t, 'out', b, None, dace.Memlet('B[5]'))

    CleanAccessNodeToScalarSliceToTaskletPattern().apply_pass(sdfg, None)
    assert not any(n.data == 'tmp' for n in state.data_nodes())

    sdfg.validate()
    A = np.random.rand(10)
    B = np.zeros(10, dtype=np.float32)
    sdfg(A=A, B=B)
    assert np.isclose(B[5], np.float32(A[5] * 2.0), atol=1e-6)


def test_dtype_mismatch_reused_compiles():
    """Same dtype mismatch but the scalar is reused: the assign tasklet
    carries the f64->f32 cast and the copy compiles."""
    sdfg = dace.SDFG('test_dtype_assign')
    sdfg.add_array('A', [10], dace.float64)
    sdfg.add_array('tmp', [1], dace.float32, transient=True)
    sdfg.add_array('B', [10], dace.float32)
    sdfg.add_array('C', [10], dace.float32)

    s0 = sdfg.add_state('s0')
    a = s0.add_access('A')
    tmp0 = s0.add_access('tmp')
    b = s0.add_access('B')
    t0 = s0.add_tasklet('scale', {'inp'}, {'out'}, 'out = inp * 2.0')
    s0.add_edge(a, None, tmp0, None, dace.Memlet('A[5]'))
    s0.add_edge(tmp0, None, t0, 'inp', dace.Memlet('tmp[0]'))
    s0.add_edge(t0, 'out', b, None, dace.Memlet('B[5]'))

    s1 = sdfg.add_state('s1')
    tmp1 = s1.add_access('tmp')
    c = s1.add_access('C')
    t1 = s1.add_tasklet('triple', {'inp'}, {'out'}, 'out = inp * 3.0')
    s1.add_edge(tmp1, None, t1, 'inp', dace.Memlet('tmp[0]'))
    s1.add_edge(t1, 'out', c, None, dace.Memlet('C[3]'))
    sdfg.add_edge(s0, s1, dace.InterstateEdge())

    CleanAccessNodeToScalarSliceToTaskletPattern().apply_pass(sdfg, None)
    assert any(n.data == 'tmp' for n in s0.data_nodes())

    sdfg.validate()
    A = np.random.rand(10)
    B = np.zeros(10, dtype=np.float32)
    C = np.zeros(10, dtype=np.float32)
    sdfg(A=A, B=B, C=C)
    assert np.isclose(B[5], np.float32(A[5] * 2.0), atol=1e-6)
    assert np.isclose(C[3], np.float32(A[5] * 3.0), atol=1e-6)


def test_scalar_reused_in_interstate_edge_is_kept():
    """A scalar referenced in an interstate-edge condition counts as
    reused, so the pass keeps it (assign tasklet) rather than removing."""
    sdfg = dace.SDFG('test_interstate_reuse')
    sdfg.add_array('A', [10], dace.float64)
    sdfg.add_scalar('tmp', dace.float64, transient=True)
    sdfg.add_array('B', [10], dace.float64)

    s0 = sdfg.add_state('s0')
    a = s0.add_access('A')
    tmp = s0.add_access('tmp')
    b = s0.add_access('B')
    t = s0.add_tasklet('double', {'inp'}, {'out'}, 'out = inp * 2.0')
    s0.add_edge(a, None, tmp, None, dace.Memlet('A[5]'))
    s0.add_edge(tmp, None, t, 'inp', dace.Memlet('tmp[0]'))
    s0.add_edge(t, 'out', b, None, dace.Memlet('B[5]'))

    s1 = sdfg.add_state('s1')
    s2 = sdfg.add_state('s2')
    sdfg.add_edge(s0, s1, dace.InterstateEdge(condition='tmp > 0'))
    sdfg.add_edge(s0, s2, dace.InterstateEdge(condition='tmp <= 0'))

    CleanAccessNodeToScalarSliceToTaskletPattern().apply_pass(sdfg, None)
    assert any(n.data == 'tmp' for n in s0.data_nodes()), "scalar used in interstate edge must be kept"


def test_refuses_when_slice_read_edge_has_wcr():
    """The fold must refuse when the ``A_slice -> tasklet`` edge
    carries WCR (atomic / reduction semantics). Folding would replace
    the WCR-bearing scalar-read edge with a plain array memlet and
    drop the atomic semantics. Symmetric with the inverse-fold WCR
    guard on the tasklet -> scalar edge."""
    sdfg = dace.SDFG('rmw_wcr')
    sdfg.add_array('A', (16, ), dace.float64)
    sdfg.add_array('B', (16, ), dace.float64)
    sdfg.add_scalar('A_slice', dace.float64, transient=True)
    sdfg.add_symbol('k', dace.int64)

    s = sdfg.add_state('rmw', is_start_block=True)
    a_read = s.add_access('A')
    a_slice = s.add_access('A_slice')
    s.add_edge(a_read, None, a_slice, None, dace.Memlet('A[k]'))
    t = s.add_tasklet('dbl', {'_in'}, {'_o'}, '_o = _in * 2.0')
    # Outbound WCR on the A_slice -> tasklet edge -- this is the case
    # the guard protects.
    wcr_memlet = dace.Memlet('A_slice[0]')
    wcr_memlet.wcr = 'lambda old, new: old + new'
    s.add_edge(a_slice, None, t, '_in', wcr_memlet)
    b_write = s.add_access('B')
    s.add_edge(t, '_o', b_write, None, dace.Memlet('B[k]'))

    CleanAccessNodeToScalarSliceToTaskletPattern().apply_pass(sdfg, None)
    assert any(n.data == 'A_slice' for n in s.data_nodes()), \
        'A_slice must survive when the A_slice->tasklet edge carries WCR'


def test_refuses_when_other_subset_missing_and_source_is_not_an_accessnode():
    """``_slice_read`` cannot recover the read subset when the memlet
    is named after the scalar (``data == an2.data``), ``other_subset``
    is unset, AND the source is not a plain AccessNode whose own
    ``data`` we can fall back on. The fold must refuse rather than
    synthesise an incorrect subscript (the original code asserted and
    aborted the whole canonicalize call).

    We reproduce by attaching a MapEntry source -- not an AccessNode --
    with a memlet whose ``data`` matches the scalar and ``other_subset
    is None``.
    """
    sdfg = dace.SDFG('rmw_other_subset_none')
    sdfg.add_array('A', (16, ), dace.float64)
    sdfg.add_array('B', (16, ), dace.float64)
    sdfg.add_scalar('A_slice', dace.float64, transient=True)

    s = sdfg.add_state('rmw', is_start_block=True)
    a_read = s.add_read('A')
    me, mx = s.add_map('m', dict(i='0:16'))
    a_slice = s.add_access('A_slice')
    # The MapEntry -> A_slice edge has ``data='A_slice'`` (matches
    # scalar) and no ``other_subset`` -- the case ``_slice_read``
    # cannot recover from.
    bad_memlet = dace.Memlet('A_slice[0]')
    s.add_edge(a_read, None, me, 'IN_A', dace.Memlet('A[0:16]'))
    me.add_in_connector('IN_A')
    me.add_out_connector('OUT_A')
    s.add_edge(me, 'OUT_A', a_slice, None, bad_memlet)
    t = s.add_tasklet('dbl', {'_in'}, {'_o'}, '_o = _in * 2.0')
    s.add_edge(a_slice, None, t, '_in', dace.Memlet('A_slice[0]'))
    b_w = s.add_access('B')
    s.add_edge(t, '_o', mx, 'IN_B', dace.Memlet('B[i]'))
    mx.add_in_connector('IN_B')
    mx.add_out_connector('OUT_B')
    s.add_edge(mx, 'OUT_B', b_w, None, dace.Memlet('B[0:16]'))

    # The fold must not raise; it must simply leave A_slice alone.
    CleanAccessNodeToScalarSliceToTaskletPattern().apply_pass(sdfg, None)
    assert any(n.data == 'A_slice' for n in s.data_nodes()), \
        'A_slice must survive when the read subset cannot be recovered'


def test_refuses_when_source_array_also_written_in_same_state():
    """The fold must NOT collapse ``A -> A_slice -> tasklet`` when the
    source array ``A`` is ALSO written somewhere in the same state.
    The intermediate scalar is load-bearing: it makes the gather +
    update sequence explicit, and folding it exposes a self-RMW
    pattern (``A[k] = A[k] + ...``) that downstream matchers (e.g.
    ``LoopToReduce``) then mis-classify as a single-cell reduction.

    TSVC ``s141`` is the canonical case: a triangular
    ``flat_2d_array[k] += bb[j, i]`` with a carried-scalar ``k``
    update. Without this guard, ``LoopToReduce`` mis-fires on the
    folded form and the canonicalised SDFG diverges from the numpy
    reference (regression seen end-to-end on ``s141_d_single``).
    """
    sdfg = dace.SDFG('rmw_same_state')
    sdfg.add_array('A', (16, ), dace.float64)
    sdfg.add_array('B', (16, ), dace.float64)
    sdfg.add_scalar('A_slice', dace.float64, transient=True)
    sdfg.add_scalar('B_slice', dace.float64, transient=True)
    sdfg.add_scalar('add_out', dace.float64, transient=True)
    sdfg.add_symbol('k', dace.int64)

    s = sdfg.add_state('rmw', is_start_block=True)
    # Gather A[k] -> A_slice (the fold's target).
    a_read = s.add_access('A')
    a_slice = s.add_access('A_slice')
    s.add_edge(a_read, None, a_slice, None, dace.Memlet('A[k]'))
    # Gather B[k] -> B_slice (a sibling gather, not the fold's
    # target -- only here to make the body realistic).
    b_read = s.add_access('B')
    b_slice = s.add_access('B_slice')
    s.add_edge(b_read, None, b_slice, None, dace.Memlet('B[k]'))
    # Tasklet adds them into a scratch transient.
    t = s.add_tasklet('add', {'_a', '_b'}, {'_o'}, '_o = _a + _b')
    s.add_edge(a_slice, None, t, '_a', dace.Memlet('A_slice[0]'))
    s.add_edge(b_slice, None, t, '_b', dace.Memlet('B_slice[0]'))
    out = s.add_access('add_out')
    s.add_edge(t, '_o', out, None, dace.Memlet('add_out[0]'))
    # And finally write back into ``A[k]`` -- the same-state write
    # of the source array that the guard must detect.
    a_write = s.add_access('A')
    s.add_edge(out, None, a_write, None, dace.Memlet('A[k]'))

    CleanAccessNodeToScalarSliceToTaskletPattern().apply_pass(sdfg, None)
    # The fold must have refused: ``A_slice`` is still present in
    # the state because the source array ``A`` is also written
    # somewhere in the same state.
    assert any(n.data == 'A_slice' for n in s.data_nodes()), \
        ('A_slice must survive when A is read+written in the same state '
         '(self-RMW exposes load-bearing intermediate scalar)')
    # ``B_slice``, in contrast, has no same-state write of its
    # source array ``B`` -- the fold IS allowed to drop it.
    assert not any(n.data == 'B_slice' for n in s.data_nodes()), \
        'B_slice should fold normally; only A_slice is protected by the same-state-write guard'


if __name__ == '__main__':
    test_scalar_slice_removed()
    test_scalar_reused_gets_assign_tasklet()
    test_permissive_removes_even_if_reused()
    test_dtype_mismatch_not_reused_compiles()
    test_dtype_mismatch_reused_compiles()
    test_scalar_reused_in_interstate_edge_is_kept()
    test_refuses_when_slice_read_edge_has_wcr()
    test_refuses_when_other_subset_missing_and_source_is_not_an_accessnode()
    test_refuses_when_source_array_also_written_in_same_state()
