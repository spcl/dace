# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import numpy as np
from dace.sdfg.state import ConditionalBlock


def test_dataflow_if_check():

    @dace.program
    def dataflow_if_check(A: dace.int32[10], i: dace.int64):
        if A[i] < 10:
            return 0
        elif A[i] == 10:
            return 10
        return 100

    dataflow_if_check.use_explicit_cf = True
    sdfg = dataflow_if_check.to_sdfg()

    assert any(isinstance(x, ConditionalBlock) for x in sdfg.nodes())

    A = np.zeros((10, ), np.int32)
    A[4] = 10
    A[5] = 100
    assert sdfg(A, 0)[0] == 0
    assert sdfg(A, 4)[0] == 10
    assert sdfg(A, 5)[0] == 100
    assert sdfg(A, 6)[0] == 0


def test_nested_if_chain():

    @dace.program
    def nested_if_chain(i: dace.int64):
        if i < 2:
            return 0
        else:
            if i < 4:
                return 1
            else:
                if i < 6:
                    return 2
                else:
                    if i < 8:
                        return 3
                    else:
                        return 4

    nested_if_chain.use_explicit_cf = True
    sdfg = nested_if_chain.to_sdfg()

    assert any(isinstance(x, ConditionalBlock) for x in sdfg.nodes())

    assert nested_if_chain(0)[0] == 0
    assert nested_if_chain(2)[0] == 1
    assert nested_if_chain(4)[0] == 2
    assert nested_if_chain(7)[0] == 3
    assert nested_if_chain(15)[0] == 4


def test_elif_chain():

    @dace.program
    def elif_chain(i: dace.int64):
        if i < 2:
            return 0
        elif i < 4:
            return 1
        elif i < 6:
            return 2
        elif i < 8:
            return 3
        else:
            return 4

    elif_chain.use_explicit_cf = True
    sdfg = elif_chain.to_sdfg()

    assert any(isinstance(x, ConditionalBlock) for x in sdfg.nodes())

    assert elif_chain(0)[0] == 0
    assert elif_chain(2)[0] == 1
    assert elif_chain(4)[0] == 2
    assert elif_chain(7)[0] == 3
    assert elif_chain(15)[0] == 4


if __name__ == '__main__':
    test_dataflow_if_check()
    test_nested_if_chain()
    test_elif_chain()
