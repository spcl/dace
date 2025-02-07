# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

n = 1024


@dace.program
def transients(A: dace.float32[n]):
    ostream = dace.define_stream(dace.float32, n)
    oscalar = dace.define_local_scalar(dace.int32)
    oarray = dace.define_local([n], dace.float32)
    oarray[:] = 0
    oscalar = 0
    for i in dace.map[0:n]:
        if A[i] >= 0.5:
            A[i] >> ostream(-1)
            with dace.tasklet:
                out >> oscalar(1, lambda a, b: a + b)
                out = 1
    ostream >> oarray
    return oscalar, oarray


def test_transients():
    A = np.random.rand(n).astype(np.float32)
    scal, arr = transients(A)
    if scal[0] > 0:
        assert (arr[0:scal[0]] >= 0.5).all()
    assert (arr[scal[0]:] == 0).all()


if __name__ == "__main__":
    test_transients()
