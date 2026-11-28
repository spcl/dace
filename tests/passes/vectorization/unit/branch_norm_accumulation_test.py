# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``BranchNormalization`` keeps a guarded accumulation an accumulation.

CloudSC's species loop subtracts from ``tendency_loc_t[jk, jl]`` under ``if iphase(jm) == 1``. The gate turned the
WCR write ``T (+)= -x`` into the masked store ``if (c) T = -x``: the old value was lost, and so was the atomic the
parallel species map needs.
"""

import numpy as np

import dace
from dace.sdfg import nodes
from dace.transformation.passes.vectorization.branch_normalization import BranchNormalization

N, S = (dace.symbol(name) for name in "NS")


@dace.program
def guarded_accumulation(flag: dace.int32[S], x: dace.float64[S, N], T: dace.float64[N]):
    for s, i in dace.map[0:S, 0:N]:
        if flag[s] == 1:
            T[i] += -x[s, i]


def test_a_guarded_accumulation_keeps_its_wcr():
    sdfg = guarded_accumulation.to_sdfg(simplify=True)
    BranchNormalization().apply_pass(sdfg, {})
    sdfg.validate()
    gates = [
        (node, parent)
        for node, parent in sdfg.all_nodes_recursive()
        if isinstance(node, nodes.Tasklet) and node.label.startswith("bn_ite_T")
    ]
    for gate, state in gates:
        assert all(edge.data.wcr is not None for edge in state.out_edges(gate)), "the gated write lost its WCR"
    rng = np.random.default_rng(0)
    flag = np.array([1, 0, 1, 1], dtype=np.int32)
    x, T = rng.random((4, 6)), rng.random(6)
    want = T - x[flag == 1].sum(axis=0)
    sdfg(flag=flag, x=x, T=T, N=6, S=4)
    assert np.allclose(T, want)


if __name__ == "__main__":
    test_a_guarded_accumulation_keeps_its_wcr()
