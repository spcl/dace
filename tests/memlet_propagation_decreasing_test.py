# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from dace import nodes


def test_decreasing_propagation():

    p = np.random.randn(19, 19)
    q = np.random.randn(19, 19)
    ref = q.copy()

    @dace.program
    def copy_nw_corner(p: dace.float64[19, 19], q: dace.float64[19, 19]):
        for j in dace.map[15:19]:
            for i in dace.map[0:3]:
                q[i, j] = p[j - 12, 17 - i]

    sdfg = copy_nw_corner.to_sdfg()
    sdfg.view()
    me = None
    state = None
    for n, s in sdfg.all_nodes_recursive():
        if isinstance(n, nodes.MapEntry) and n.map.params[0] == 'j':
            me = n
            state = s
            break
    assert (me)
    assert (state)
    edges = state.in_edges(me)
    assert (len(edges) == 1)
    subset = edges[0].data.src_subset
    assert (subset.ranges == [(3, 6, 1), (15, 17, 1)])

    copy_nw_corner(p, q)
    copy_nw_corner.f(p, ref)
    assert (np.allclose(q, ref))


if __name__ == '__main__':
    test_decreasing_propagation()
