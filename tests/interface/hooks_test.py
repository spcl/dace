# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests hooks that can be used to extend DaCe functionality.
"""

import dace
import numpy as np

def test_profile():
    @dace.program
    def test1(A: dace.float64[20]):
        return A + 1

    @dace.program
    def test2(A: dace.float64[20]):
        return A + 2

    A = np.random.rand(20)
    expected1 = A + 1
    expected2 = A + 2
    
    with dace.profile(repetitions=10) as prof:
        r2 = test2(A)
        r1 = test1(A)
    
    assert np.allclose(r1, expected1)
    assert np.allclose(r2, expected2)

    assert len(prof.times) == 2
    assert len(prof.times[0][1]) == 10
    assert len(prof.times[1][1]) == 10
    assert prof.times[0][0] == 'test2'
    assert prof.times[1][0] == 'test1'


if __name__ == '__main__':
    test_profile()
