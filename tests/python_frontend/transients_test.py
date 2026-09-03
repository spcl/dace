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


@dace.program
def push_every_element(A: dace.float32[n]):
    ostream = dace.define_stream(dace.float32, n)
    oarray = dace.define_local([n], dace.float32)
    for i in dace.map[0:n]:
        A[i] >> ostream(-1)
    ostream >> oarray
    return oarray


def test_stream_push_names_the_element():
    """What a push moves is the element the memlet names, not the start of the container.

    The map pushes every element once, in no particular order, so the result is a permutation of the
    input -- which it is not if every iteration pushes the same element.
    """
    A = np.arange(n, dtype=np.float32)
    pushed = push_every_element(A)
    assert np.array_equal(np.sort(pushed), A)


if __name__ == "__main__":
    test_transients()
    test_stream_push_names_the_element()
