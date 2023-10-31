# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
from sympy import core as sympy_core

import dace
from dace import subsets


def test_double_index_bug():

    sdfg = dace.SDFG('test_')
    state = sdfg.add_state()

    sdfg.add_array('A', shape=(10, ), dtype=dace.float64)
    sdfg.add_array('table', shape=(10, 2), dtype=dace.int64)
    sdfg.add_array('B', shape=(10, ), dtype=dace.float64)
    sdfg.add_scalar('idx', dace.int64, transient=True)
    idx_node = state.add_access('idx')
    set_tlet = state.add_tasklet('set_idx', code="_idx=0", inputs={}, outputs={"_idx"})
    state.add_mapped_tasklet('map',
                             map_ranges={'i': "0:10"},
                             inputs={
                                 'inp': dace.Memlet("A[0:10]"),
                                 '_idx': dace.Memlet('idx[0]'),
                                 'indices': dace.Memlet('table[0:10, 0:2]')
                             },
                             code="out = inp[indices[i,_idx]]",
                             outputs={'out': dace.Memlet("B[i]")},
                             external_edges=True,
                             input_nodes={'idx': idx_node})

    state.add_edge(set_tlet, '_idx', idx_node, None, dace.Memlet('idx[0]'))

    sdfg.simplify()

    # Check that `indices` (which is an array) is not used in a memlet subset
    for state in sdfg.states():
        for memlet in state.edges():
            subset = memlet.data.subset
            if not isinstance(subset, subsets.Range):
                continue
            for range in subset.ranges:
                for part in range:
                    for sympy_node in sympy_core.preorder_traversal(part):
                        assert getattr(sympy_node, "name", None) != "indices"


if __name__ == '__main__':
    test_double_index_bug()
