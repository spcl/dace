# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

pair = dace.struct('pair', idx=dace.int32, val=dace.float64)


@dace.program
def argmax(x: dace.float64[1024]):
    result = np.ndarray([1], dtype=pair)
    with dace.tasklet:
        init >> result[0]
        init.idx = -1
        init.val = -1e38

    for i in dace.map[0:1024]:
        with dace.tasklet:
            inp << x[i]
            out >> result(1, lambda x, y: pair(val=max(x.val, y.val), idx=(x.idx if x.val > y.val else y.idx)))
            out = pair(idx=i, val=inp)

    return result


def test_argmax():
    A = np.random.rand(1024)
    result = argmax(A)
    assert result[0][0] == np.argmax(A)


if __name__ == '__main__':
    test_argmax()
