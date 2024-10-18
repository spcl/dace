# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.


import numpy as np
import dace
from dace.sdfg.state import ConditionalBlock
from dace.transformation.passes.simplification.prune_empty_conditional_branches import PruneEmptyConditionalBranches


def test_prune_empty_else():
    N = dace.symbol('N')

    @dace.program
    def prune_empty_else(A: dace.int32[N]):
        A[:] = 0
        if N == 32:
            for i in range(N):
                A[i] = 1
        else:
            A[:] = 0

    sdfg = prune_empty_else.to_sdfg(simplify=False)

    conditional: ConditionalBlock = None
    for n in sdfg.nodes():
        if isinstance(n, ConditionalBlock):
            conditional = n
            break

    assert len(conditional.branches) == 2

    conditional._branches[-1][0] = None
    else_branch = conditional._branches[-1][1]
    else_branch.remove_nodes_from(else_branch.nodes())
    else_branch.add_state('empty')

    res = PruneEmptyConditionalBranches().apply_pass(sdfg, {})

    assert res[conditional.cfg_id] == 1
    assert len(conditional.branches) == 1

    N1 = 32
    N2 = 31
    A1 = np.zeros((N1,), dtype=np.int32)
    A2 = np.zeros((N2,), dtype=np.int32)
    verif1 = np.full((N1,), 1, dtype=np.int32)
    verif2 = np.zeros((N2,), dtype=np.int32)

    sdfg(A1, N=N1)
    sdfg(A2, N=N2)

    assert np.allclose(A1, verif1)
    assert np.allclose(A2, verif2)


def test_prune_empty_if_with_else():
    N = dace.symbol('N')

    @dace.program
    def prune_empty_if_with_else(A: dace.int32[N]):
        A[:] = 0
        if N == 32:
            for i in range(N):
                A[i] = 2
        else:
            A[:] = 1

    sdfg = prune_empty_if_with_else.to_sdfg(simplify=False)

    conditional: ConditionalBlock = None
    for n in sdfg.nodes():
        if isinstance(n, ConditionalBlock):
            conditional = n
            break

    assert len(conditional.branches) == 2

    conditional._branches[-1][0] = None
    if_branch = conditional._branches[0][1]
    if_branch.remove_nodes_from(if_branch.nodes())
    if_branch.add_state('empty')

    res = PruneEmptyConditionalBranches().apply_pass(sdfg, {})

    assert res[conditional.cfg_id] == 1
    assert len(conditional.branches) == 1
    assert conditional.branches[0][0] is not None

    N1 = 32
    N2 = 31
    A1 = np.zeros((N1,), dtype=np.int32)
    A2 = np.zeros((N2,), dtype=np.int32)
    verif1 = np.zeros((N1,), dtype=np.int32)
    verif2 = np.full((N2,), 1, dtype=np.int32)

    sdfg(A1, N=N1)
    sdfg(A2, N=N2)

    assert np.allclose(A1, verif1)
    assert np.allclose(A2, verif2)


if __name__ == '__main__':
    test_prune_empty_else()
    test_prune_empty_if_with_else()
