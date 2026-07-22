# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``TrivialTaskletElimination`` must keep the copies around a carried accumulator.

A transient scalar staged from an array element, reduced via a WCR across states,
and written back (``w = A[i,j]``; ``w (+)= ...``; ``A[i,j] = w``) is a loop-carried
reduction accumulator. The staging / write-back copies sequence that cross-state
carry; eliminating either splices the array element straight onto the WCR-written
scalar and drops the reduction (polybench ludcmp's LU update collapses to zero).
"""
import copy

import numpy as np
import pytest

import dace
from dace.sdfg.sdfg import InterstateEdge
from dace.transformation.dataflow.trivial_tasklet_elimination import (TrivialTaskletElimination,
                                                                      _is_carried_reduction_accumulator)

N = 8


def _carried_accumulator_sdfg():
    """``for i: (w = A[i]); (w (+)= B[i,k] over k); (A[i] = w)`` across states."""
    sdfg = dace.SDFG('carried')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('B', [N, N], dace.float64)
    sdfg.add_transient('w', [1], dace.float64)

    stage = sdfg.add_state('stage', is_start_block=True)
    at = stage.add_tasklet('cp', {'i'}, {'o'}, 'o = i')
    stage.add_edge(stage.add_access('A'), None, at, 'i', dace.Memlet('A[0]'))
    stage.add_edge(at, 'o', stage.add_access('w'), None, dace.Memlet('w[0]'))

    reduce_st = sdfg.add_state('reduce')
    me, mx = reduce_st.add_map('k', dict(k=f'0:{N}'))
    rt = reduce_st.add_tasklet('acc', {'b'}, {'o'}, 'o = b')
    reduce_st.add_memlet_path(reduce_st.add_access('B'), me, rt, dst_conn='b', memlet=dace.Memlet('B[0, k]'))
    wnode = reduce_st.add_access('w')
    reduce_st.add_memlet_path(rt, mx, wnode, src_conn='o', memlet=dace.Memlet('w[0]', wcr='lambda x, y: x + y'))

    writeback = sdfg.add_state('writeback')
    wt = writeback.add_tasklet('cp', {'i'}, {'o'}, 'o = i')
    writeback.add_edge(writeback.add_access('w'), None, wt, 'i', dace.Memlet('w[0]'))
    writeback.add_edge(wt, 'o', writeback.add_access('A'), None, dace.Memlet('A[0]'))

    sdfg.add_edge(stage, reduce_st, InterstateEdge())
    sdfg.add_edge(reduce_st, writeback, InterstateEdge())
    return sdfg


def test_classifier_flags_the_accumulator():
    """The helper recognises ``w`` (cross-state, WCR target) but not a plain array."""
    sdfg = _carried_accumulator_sdfg()
    assert _is_carried_reduction_accumulator(sdfg, 'w') is True
    assert _is_carried_reduction_accumulator(sdfg, 'A') is False
    assert _is_carried_reduction_accumulator(sdfg, 'B') is False


def test_staging_and_writeback_copies_survive():
    """Neither the staging nor the write-back copy may be eliminated."""
    sdfg = _carried_accumulator_sdfg()
    applied = sdfg.apply_transformations_repeated(TrivialTaskletElimination, validate=False)
    assert applied == 0, 'the carried-accumulator copies must not be eliminated'


def test_plain_cross_state_relay_is_still_eliminated():
    """A cross-state scalar WITHOUT a WCR is an ordinary relay -- still eliminable,
    so the guard is narrow and does not needlessly block optimization."""
    sdfg = dace.SDFG('relay')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('C', [N], dace.float64)
    sdfg.add_transient('t', [1], dace.float64)
    s1 = sdfg.add_state('s1', is_start_block=True)
    t1 = s1.add_tasklet('cp', {'i'}, {'o'}, 'o = i')
    s1.add_edge(s1.add_access('A'), None, t1, 'i', dace.Memlet('A[0]'))
    s1.add_edge(t1, 'o', s1.add_access('t'), None, dace.Memlet('t[0]'))
    s2 = sdfg.add_state('s2')
    t2 = s2.add_tasklet('cp', {'i'}, {'o'}, 'o = i')
    s2.add_edge(s2.add_access('t'), None, t2, 'i', dace.Memlet('t[0]'))
    s2.add_edge(t2, 'o', s2.add_access('C'), None, dace.Memlet('C[0]'))
    sdfg.add_edge(s1, s2, InterstateEdge())

    assert _is_carried_reduction_accumulator(sdfg, 't') is False
    assert sdfg.apply_transformations_repeated(TrivialTaskletElimination, validate=False) >= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
