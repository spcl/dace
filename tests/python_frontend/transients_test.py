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
            ostream.push(A[i])
            with dace.tasklet:
                out >> oscalar(1, lambda a, b: a + b)
                out = 1
    oarray[:] = ostream.pop()
    return oscalar, oarray


def test_transients():
    A = np.random.rand(n).astype(np.float32)
    scal, arr = transients(A)
    if scal[0] > 0:
        assert (arr[0:scal[0]] >= 0.5).all()
    # The pop writes only the elements the stream held, into the array the
    # program zeroed itself.
    assert (arr[scal[0]:] == 0).all()


@dace.program
def filtered(A: dace.float32[n], out: dace.float32[n]):
    ostream = dace.define_stream(dace.float32, n)
    for i in dace.map[0:n]:
        with dace.tasklet:
            a << A[i]
            if a >= 0.5:
                b = a
            b >> ostream(-1)
    out[:] = ostream.pop()


def test_drain_into_an_argument_is_a_zero_copy_view():
    """A pop assigned into an array IN FULL drains into that array itself,
    with no buffer and no initialization of its own in between -- which leaves
    the pattern the code generator emits as a zero-copy stream-array view
    (``samples/explicit/filter.py``)."""
    sdfg = filtered.to_sdfg()
    assert 'dace::ArrayStreamView' in sdfg.generate_code()[0].clean_code

    A = np.random.rand(n).astype(np.float32)
    out = np.zeros(n, np.float32)
    sdfg(A=A, out=out)
    expected = A[A >= 0.5]
    assert np.allclose(np.sort(out[:expected.size]), np.sort(expected))


if __name__ == "__main__":
    test_transients()
    test_drain_into_an_argument_is_a_zero_copy_view()
