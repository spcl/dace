# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import os
from copy import deepcopy

import numpy as np

import dace
from dace import SDFG, Memlet, InterstateEdge
from dace.transformation.interstate import StateReplication


def make_branched_sdfg_with_replicable_branch_terminal():
    """
    Construct a simple SDFG of the structure:
               guard_state
               /        \\
        branch_1         branch_2
              \\        /
             terminal_state
              (+ interior)
    """
    g = SDFG('prog')
    st0 = g.add_state("guard_state", is_start_block=True)
    st1 = g.add_state("branch_1")
    st2 = g.add_state("branch_2")
    st3 = g.add_state("terminal_state")
    g.add_array('A', (2,), dace.float32)
    g.add_symbol('flag', dace.bool)

    # Do something on the branches.
    t = st1.add_tasklet('write_1', {}, {'__out'}, '__out = 1')
    A = st1.add_access('A')
    st1.add_edge(t, '__out', A, None, Memlet(expr='A[0]'))
    t = st2.add_tasklet('write_2', {}, {'__out'}, '__out = 2')
    A = st2.add_access('A')
    st2.add_edge(t, '__out', A, None, Memlet(expr='A[0]'))

    # Do something in the terminal state.
    t = st3.add_tasklet('write_0', {}, {'__out'}, '__out = 3')
    A = st3.add_access('A')
    st3.add_edge(t, '__out', A, None, Memlet(expr='A[1]'))

    # Connect the states.
    g.add_edge(st0, st1, InterstateEdge(condition='(flag)'))
    g.add_edge(st0, st2, InterstateEdge(condition='(not flag)'))
    g.add_edge(st1, st3, InterstateEdge())
    g.add_edge(st2, st3, InterstateEdge())

    g.fill_scope_connectors()

    return g


def test_replicable_branch_terminal():
    origA = np.zeros((2,), np.float32)

    g = make_branched_sdfg_with_replicable_branch_terminal()
    g.save(os.path.join('_dacegraphs', 'simple-0.sdfg'))
    g.validate()
    g.compile()

    # Get the expected values.
    wantA_1 = deepcopy(origA)
    wantA_2 = deepcopy(origA)
    g(A=wantA_1, flag=True)
    g(A=wantA_2, flag=False)

    # Before, the outer graph had four states.
    assert len(g.nodes()) == 4

    assert g.apply_transformations_repeated([StateReplication]) == 1

    g.save(os.path.join('_dacegraphs', 'simple-1.sdfg'))
    g.validate()
    g.compile()

    # But now, the graph have six states: the terminal state spawned two additional states on the branches.
    assert len(g.nodes()) == 6

    # Get the values from transformed program.
    gotA_1 = deepcopy(origA)
    gotA_2 = deepcopy(origA)
    g(A=gotA_1, flag=True)
    g(A=gotA_2, flag=False)

    # Verify numerically.
    assert all(np.equal(wantA_1, gotA_1))
    assert all(np.equal(wantA_2, gotA_2))


def make_sdfg_with_loop():
    g = SDFG('prog')
    g.add_symbol('i', dace.int32)
    g.add_array('A', (1,), dace.float32)
    st0 = g.add_state('head', is_start_block=True)
    st1 = g.add_state('body', is_start_block=True)
    st2 = g.add_state('tail', is_start_block=True)

    g.add_loop(st0, st1, st2, 'i', '0', 'i < 10', 'i + 1')
    t = st1.add_tasklet('write_1', {}, {'__out'}, '__out = i')
    A = st1.add_access('A')
    st1.add_edge(t, '__out', A, None, Memlet(expr='A[0]'))

    return g


def test_does_not_replicate_loop_terminal_guard():
    g = make_sdfg_with_loop()
    g.save(os.path.join('_dacegraphs', 'loop-0.sdfg'))
    g.validate()
    g.compile()

    assert g.apply_transformations_repeated([StateReplication]) == 0


if __name__ == '__main__':
    test_replicable_branch_terminal()
