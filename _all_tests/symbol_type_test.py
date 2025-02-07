# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.memlet import Memlet
import numpy as np


def test_nested_symbol_type():
    test_sdfg = dace.SDFG("test_nested_symbol_type")
    test_state = test_sdfg.add_state("test_state")
    test_sdfg.add_symbol("s", dace.float32)
    test_sdfg.add_array('output', shape=[1], dtype=dace.float32)

    out = test_state.add_write('output')
    tasklet = test_state.add_tasklet('bugs', [], ['out'], 'out = s')

    test_state.add_memlet_path(tasklet, out, src_conn='out', memlet=Memlet.simple(out.data, "0"))

    outer_sdfg = dace.SDFG("nested_symbol_type")
    outer_state = outer_sdfg.add_state("outer_state")

    outer_sdfg.add_symbol("s", dace.float32)
    outer_sdfg.add_array('data', shape=[1], dtype=dace.float32)

    data = outer_state.add_write('data')
    nested = outer_state.add_nested_sdfg(test_sdfg, outer_sdfg, {}, {'output'})

    outer_state.add_memlet_path(nested, data, src_conn='output', memlet=Memlet.simple(data.data, "0"))

    compiledSDFG = outer_sdfg.compile()

    res = np.zeros(1, dtype=np.float32)
    compiledSDFG(data=res, s=np.float32(1.5))

    print("res:", res[0])
    assert res[0] == np.float32(1.5)


if __name__ == '__main__':
    test_nested_symbol_type()
