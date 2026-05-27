# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for ``CleanScalarAssignToMapExit``.

The pass folds a redundant in-scope chain::

    ... -> assign_tasklet(__out = __inp) -> scalar(thread-local transient) -> MapExit

When the scalar is written once, read once and reused nowhere else, the
scalar AccessNode and the assign tasklet are removed and the producer is
rewired straight to the MapExit, preserving the destination memlet and the
MapExit connector. The tests cover:

1. the pattern is cleaned and stays numerically equivalent to the un-cleaned
   SDFG (both compiled and run);
2. the pass does NOT fire when the scalar is reused in another state;
3. idempotency: a second run is a no-op.
"""
import copy

import dace
import numpy as np

from dace.transformation.passes.vectorization.clean_scalar_assign_to_map_exit import CleanScalarAssignToMapExit


def _build_chain_sdfg(name: str = 'clean_scalar_assign'):
    """Build ``B[i] = A[i] * 2`` with the redundant assign -> scalar -> MapExit chain.

    The map body is ``A[i] -> compute(__out = __inp * 2) -> tmp[0] (scalar
    transient) -> assign(__out = __inp) -> s[0] (scalar transient) -> MapExit
    -> B[i]``. Here ``s`` is the thread-local scalar the pass targets, fed by
    the trivial assign tasklet, flowing straight into the MapExit.

    :param name: SDFG name.
    :returns: ``(sdfg, state, map_exit, compute_tasklet)``.
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [16], dace.float64)
    sdfg.add_array('B', [16], dace.float64)
    sdfg.add_scalar('tmp', dace.float64, transient=True)
    sdfg.add_scalar('s', dace.float64, transient=True)

    state = sdfg.add_state()
    a = state.add_access('A')
    b = state.add_access('B')
    me, mx = state.add_map('m', {'i': '0:16'})

    compute = state.add_tasklet('compute', {'__inp'}, {'__out'}, '__out = __inp * 2.0')
    assign = state.add_tasklet('assign', {'__inp'}, {'__out'}, '__out = __inp')
    s = state.add_access('s')

    # A -> MapEntry -> compute
    state.add_memlet_path(a, me, compute, dst_conn='__inp', memlet=dace.Memlet('A[i]'))
    # compute -> assign (the producer of the assign tasklet). Carries its own
    # transient ``tmp`` — distinct from ``s`` so the un-cleaned reference SDFG
    # does not materialise ``s`` on two edge groups (a double C++ declaration).
    state.add_edge(compute, '__out', assign, '__inp', dace.Memlet('tmp[0]'))
    # assign -> s (scalar transient)
    state.add_edge(assign, '__out', s, None, dace.Memlet('s[0]'))
    # s -> MapExit -> B   (creates the IN_/OUT_ connector pair on the exit)
    state.add_memlet_path(s, mx, b, memlet=dace.Memlet('B[i]'))

    sdfg.validate()
    return sdfg, state, mx, compute


def _run(sdfg: dace.SDFG, a: np.ndarray) -> np.ndarray:
    """Compile and run ``sdfg`` on input ``a``, returning the ``B`` output.

    :param sdfg: SDFG to execute (deep-copied so the caller's graph is untouched).
    :param a: Input array bound to ``A``.
    :returns: The ``B`` output array.
    """
    b = np.zeros_like(a)
    copy.deepcopy(sdfg)(A=a.copy(), B=b)
    return b


def test_pattern_is_cleaned_and_numerically_equivalent():
    """The assign + scalar are removed, the producer is rewired to the MapExit,
    and the cleaned SDFG produces the same result as the un-cleaned one."""
    sdfg, state, map_exit, compute = _build_chain_sdfg()
    uncleaned = copy.deepcopy(sdfg)

    assert any(n.data == 's' for n in state.data_nodes())
    assert sum(1 for n in state.nodes() if isinstance(n, dace.nodes.Tasklet)) == 2

    count = CleanScalarAssignToMapExit().apply_pass(sdfg, None)
    assert count == 1

    # Scalar AccessNode gone; assign tasklet gone (only compute remains).
    assert not any(n.data == 's' for n in state.data_nodes())
    tasklets = [n for n in state.nodes() if isinstance(n, dace.nodes.Tasklet)]
    assert len(tasklets) == 1 and tasklets[0] is compute

    # The producer (compute) now writes straight to the MapExit, and the
    # destination memlet/connector were preserved (writes B[i]).
    rewired = [e for e in state.out_edges(compute) if e.dst is map_exit]
    assert len(rewired) == 1
    assert rewired[0].data.data == 'B'
    assert str(rewired[0].data.subset) == 'i'
    assert rewired[0].dst_conn is not None and rewired[0].dst_conn.startswith('IN_')

    sdfg.validate()

    a = np.random.rand(16)
    np.testing.assert_allclose(_run(sdfg, a), _run(uncleaned, a), rtol=0, atol=0)
    np.testing.assert_allclose(_run(sdfg, a), a * 2.0, rtol=1e-12)


def test_does_not_fire_when_scalar_reused_in_other_state():
    """When the scalar is read in another state it is not thread-local-dead,
    so the chain must be left intact."""
    sdfg, state, _map_exit, _compute = _build_chain_sdfg('reused_scalar')

    # Add a second state that reads ``s`` -> the scalar is now reused.
    sdfg.add_array('C', [1], dace.float64)
    s1 = sdfg.add_state('s1')
    s_read = s1.add_access('s')
    c = s1.add_access('C')
    t = s1.add_tasklet('use', {'__inp'}, {'__out'}, '__out = __inp')
    s1.add_edge(s_read, None, t, '__inp', dace.Memlet('s[0]'))
    s1.add_edge(t, '__out', c, None, dace.Memlet('C[0]'))
    sdfg.add_edge(state, s1, dace.InterstateEdge())

    count = CleanScalarAssignToMapExit().apply_pass(sdfg, None)
    assert count is None

    # Chain untouched: scalar + both tasklets still present.
    assert any(n.data == 's' for n in state.data_nodes())
    assert sum(1 for n in state.nodes() if isinstance(n, dace.nodes.Tasklet)) == 2


def test_idempotent_second_run_is_noop():
    """A second run after the first fold finds nothing to do."""
    sdfg, _state, _map_exit, _compute = _build_chain_sdfg('idempotent')

    first = CleanScalarAssignToMapExit().apply_pass(sdfg, None)
    assert first == 1
    sdfg.validate()

    second = CleanScalarAssignToMapExit().apply_pass(sdfg, None)
    assert second is None


if __name__ == '__main__':
    test_pattern_is_cleaned_and_numerically_equivalent()
    test_does_not_fire_when_scalar_reused_in_other_state()
    test_idempotent_second_run_is_noop()
