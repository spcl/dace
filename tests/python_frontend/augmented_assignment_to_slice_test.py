# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import dace

def test_augmented_assignment_to_indirect_access():

    N = dace.symbol('N')
    M = dace.symbol('M')

    @dace.program
    def _test_prog(A: dace.int32[M], ind: dace.int32[N], B: dace.int32[N]):
        return A[ind] + B

    sdfg = _test_prog.to_sdfg()
    assert sdfg.is_valid()

def test_augmented_assignment_to_indirect_access_regression():

    N = dace.symbol('N')
    M = dace.symbol('M')

    @dace.program
    def _test_prog(A: dace.int32[M], ind: dace.int32[N], B: dace.int32[N]):
        A[ind] += B

    sdfg = _test_prog.to_sdfg()
    assert sdfg.is_valid()


if __name__ == '__main__':
    test_augmented_assignment_to_indirect_access()
    test_augmented_assignment_to_indirect_access_regression()
