# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import numpy as np


def test_key_replacement_same_name():

    sdfg = dace.SDFG('key_replacement_same_name')
    sdfg.add_array('inp', [1], dace.int32)
    sdfg.add_array('out', [1], dace.int32)

    first = sdfg.add_state('first_state')
    second = sdfg.add_state('second_state')
    edge = sdfg.add_edge(first, second, dace.InterstateEdge(assignments={'s': 'inp[0]'}))

    task = second.add_tasklet('t', {}, {'__out'}, '__out = s')
    access = second.add_access('out')
    second.add_edge(task, '__out', access, None, dace.Memlet('out[0]'))

    sdfg.replace('s', 's')
    assert 's' in edge.data.assignments
    sdfg.replace_dict({'s': 's'})
    assert 's' in edge.data.assignments

    rng = np.random.default_rng()
    inp = rng.integers(1, 100, 1)
    inp = np.array(inp, dtype=np.int32)
    out = np.zeros([1], dtype=np.int32)

    sdfg(inp=inp, out=out)
    assert out[0] == inp[0]


if __name__ == '__main__':
    test_key_replacement_same_name()
