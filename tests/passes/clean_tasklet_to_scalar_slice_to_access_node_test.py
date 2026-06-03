"""Unit tests for CleanTaskletToScalarSliceToAccessNodePattern.

Mirrors the input-side ``CleanAccessNodeToScalarSliceToTaskletPattern``
tests. Covers the two folding cases (not-reused -> drop scalar; reused
-> insert assignment tasklet) plus the new D1 in-state safety refusal
(introduced to prevent the fold from exposing AAW's
``expr_index == 0`` latent ``out_degree(input) == 1`` hole on the
TSVC s3112 shape ``sum += a[i]; b[i] = sum``).
"""
import dace
import numpy as np

from dace.transformation.passes.clean_tasklet_to_scalar_slice_to_access_node_pattern import (
    CleanTaskletToScalarSliceToAccessNodePattern,
)


def test_scalar_slice_removed_when_not_reused():
    """``tasklet -> tmp[0] -> A[5]``: tmp is dead elsewhere, so the pass
    removes it and wires the tasklet output straight into ``A``."""
    sdfg = dace.SDFG('test_clean_scalar_out')
    sdfg.add_array('A', [10], dace.float64)
    sdfg.add_array('tmp', [1], dace.float64, transient=True)
    sdfg.add_array('B', [10], dace.float64)

    state = sdfg.add_state()
    b = state.add_access('B')
    tmp = state.add_access('tmp')
    a = state.add_access('A')
    tasklet = state.add_tasklet('mul', {'inp'}, {'out'}, 'out = inp * 2.0')
    state.add_edge(b, None, tasklet, 'inp', dace.Memlet('B[5]'))
    state.add_edge(tasklet, 'out', tmp, None, dace.Memlet('tmp[0]'))
    state.add_edge(tmp, None, a, None, dace.Memlet('A[5]'))

    assert any(n.data == 'tmp' for n in state.data_nodes())
    CleanTaskletToScalarSliceToAccessNodePattern().apply_pass(sdfg, None)

    assert not any(n.data == 'tmp' for n in state.data_nodes())
    tasklet_node = next(n for n in state.nodes() if isinstance(n, dace.nodes.Tasklet))
    out_edges = state.out_edges(tasklet_node)
    assert len(out_edges) == 1
    assert out_edges[0].data.data == 'A'
    assert str(out_edges[0].data.subset) == '5'

    sdfg.validate()
    B = np.random.rand(10)
    A = np.zeros(10)
    sdfg(A=A, B=B)
    assert np.isclose(A[5], B[5] * 2.0)


def test_scalar_reused_in_other_state_gets_assign_tasklet():
    """If ``tmp`` is used in another state, the pass must keep it and
    splice in an ``_out = _in`` assignment tasklet instead of dropping."""
    sdfg = dace.SDFG('test_reused_out')
    sdfg.add_array('A', [10], dace.float64)
    sdfg.add_scalar('tmp', dace.float64, transient=True)
    sdfg.add_array('B', [10], dace.float64)

    s0 = sdfg.add_state('s0')
    b = s0.add_access('B')
    tmp = s0.add_access('tmp')
    a = s0.add_access('A')
    t = s0.add_tasklet('mul', {'inp'}, {'out'}, 'out = inp * 2.0')
    s0.add_edge(b, None, t, 'inp', dace.Memlet('B[5]'))
    s0.add_edge(t, 'out', tmp, None, dace.Memlet('tmp[0]'))
    s0.add_edge(tmp, None, a, None, dace.Memlet('A[5]'))

    # s1 reuses ``tmp``.
    s1 = sdfg.add_state('s1')
    sdfg.add_edge(s0, s1, dace.InterstateEdge())
    tmp_r = s1.add_access('tmp')
    a2 = s1.add_access('A')
    cp = s1.add_tasklet('cp', {'i'}, {'o'}, 'o = i')
    s1.add_edge(tmp_r, None, cp, 'i', dace.Memlet('tmp[0]'))
    s1.add_edge(cp, 'o', a2, None, dace.Memlet('A[6]'))

    CleanTaskletToScalarSliceToAccessNodePattern().apply_pass(sdfg, None)
    # The scalar must be kept (used in s1); s0 must contain an additional
    # assignment tasklet between tmp and A.
    assert any(n.data == 'tmp' for n in s0.data_nodes()), 'scalar used in another state must be kept'
    assign_tasklets = [n for n in s0.nodes() if isinstance(n, dace.nodes.Tasklet) and 'assign_out' in n.label]
    assert len(assign_tasklets) >= 1, 'output-side assign tasklet must be inserted'


def test_d1_refuses_when_destination_is_also_read_through_other_an():
    """**D1 in-state safety refusal**: when the destination array ``A`` is
    ALSO read in this state through a DIFFERENT AccessNode with
    ``out_degree > 0``, the pass must refuse the fold. This prevents
    exposing AAW's ``expr_index == 0`` latent hole. Concrete shape: the
    TSVC s3112 pattern where ``sum`` is read both by the RMW Add and by
    the subsequent assignment.
    """
    sdfg = dace.SDFG('test_d1_refusal')
    sdfg.add_array('sum', [1], dace.float64, transient=True)
    sdfg.add_array('a', [10], dace.float64)
    sdfg.add_array('b', [10], dace.float64)
    sdfg.add_scalar('tmp', dace.float64, transient=True)
    sdfg.add_symbol('i', dace.int64)

    state = sdfg.add_state('only')
    # Read sum (the OUT_DEGREE > 0 sibling reader of ``sum``).
    sum_r = state.add_access('sum')
    a_r = state.add_access('a')
    add = state.add_tasklet('add', {'_a', '_s'}, {'_o'}, '_o = _a + _s')
    state.add_edge(a_r, None, add, '_a', dace.Memlet('a[i]'))
    state.add_edge(sum_r, None, add, '_s', dace.Memlet('sum[0]'))
    # Write back to sum via tmp -> sum (the chain D1 would otherwise fold).
    tmp = state.add_access('tmp')
    sum_w = state.add_access('sum')
    state.add_edge(add, '_o', tmp, None, dace.Memlet('tmp[0]'))
    state.add_edge(tmp, None, sum_w, None, dace.Memlet('sum[0]'))

    pre_an_count = sum(1 for n in state.data_nodes() if n.data == 'tmp')
    assert pre_an_count == 1

    CleanTaskletToScalarSliceToAccessNodePattern().apply_pass(sdfg, None)

    post_an_count = sum(1 for n in state.data_nodes() if n.data == 'tmp')
    assert post_an_count == 1, ('D1 safety refusal must keep tmp in place when the destination ``sum`` '
                                'is read through a sibling AccessNode with out_degree > 0.')


def test_d1_allows_fold_when_sibling_an_is_pure_sink():
    """A sibling AccessNode of the destination's data is fine if it's a
    pure sink (``out_degree == 0``). The check refuses only when a sibling
    is being READ (``out_degree > 0``)."""
    sdfg = dace.SDFG('test_d1_allowed')
    sdfg.add_array('A', [10], dace.float64)
    sdfg.add_array('tmp', [1], dace.float64, transient=True)
    sdfg.add_array('B', [10], dace.float64)

    state = sdfg.add_state('only')
    b = state.add_access('B')
    t = state.add_tasklet('mul', {'inp'}, {'out'}, 'out = inp * 2.0')
    state.add_edge(b, None, t, 'inp', dace.Memlet('B[5]'))
    tmp = state.add_access('tmp')
    state.add_edge(t, 'out', tmp, None, dace.Memlet('tmp[0]'))
    a_w = state.add_access('A')  # the fold target
    state.add_edge(tmp, None, a_w, None, dace.Memlet('A[5]'))
    # A sibling AccessNode of ``A`` that is a pure sink (gets an empty edge
    # for example) -- this should NOT refuse the fold.
    a_sib = state.add_access('A')
    state.add_nedge(a_w, a_sib, dace.Memlet())

    CleanTaskletToScalarSliceToAccessNodePattern().apply_pass(sdfg, None)
    # The fold must have happened: tmp removed.
    assert not any(n.data == 'tmp' for n in state.data_nodes()), \
        'sink-only sibling AccessNode of the destination must not block the fold'


if __name__ == '__main__':
    test_scalar_slice_removed_when_not_reused()
    test_scalar_reused_in_other_state_gets_assign_tasklet()
    test_d1_refuses_when_destination_is_also_read_through_other_an()
    test_d1_allows_fold_when_sibling_an_is_pure_sink()
