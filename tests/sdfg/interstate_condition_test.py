# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

import dace


def test_interstate_multidim_access():
    sdfg = dace.SDFG('ima')
    sdfg.add_array('A', [2, 2], dace.float32)
    s = sdfg.add_state()
    s1 = sdfg.add_state()
    s2 = sdfg.add_state()
    sdfg.add_edge(s, s1, dace.InterstateEdge('(1.0 - A[1, 0]) > 0'))
    sdfg.add_edge(s, s2, dace.InterstateEdge('(1.0 - A[1, 0]) <= 0'))

    sdfg.compile()



if __name__ == '__main__':
    test_interstate_multidim_access()