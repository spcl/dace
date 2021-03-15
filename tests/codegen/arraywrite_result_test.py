# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

N = dace.symbol('N')


@dace.program
def writeresult(output: dace.float64[10], values: dace.float64[N]):
    for i in dace.map[0:N]:
        with dace.tasklet:
            o >> output(-1, lambda a, b: a + b)[:]
            v >> values[i]
            # Add one to output and write old value to v
            v = o[5] = 1


def test_arraywrite():
    output = np.zeros([10], dtype=np.float64)
    values = np.zeros([100], dtype=np.float64)
    writeresult(output, values)

    reference = np.array([i for i in range(100)]).astype(np.float64)
    assert np.allclose(np.array(sorted(values)), reference)


@dace.program
def writeresult2(output: dace.float64[2], values: dace.float64[N]):
    for i in dace.map[0:N]:
        with dace.tasklet:
            o >> output(2, lambda a, b: a + b)[:]
            v << values[i]
            o[0] = v
            o[1] = 2 * v


def test_arraywcr():
    A = np.random.rand(20)
    o = np.random.rand(2)
    reference = np.copy(o)
    reference[0] += np.sum(A)
    reference[1] += np.sum(A) * 2
    writeresult2(o, A)

    assert np.allclose(o, reference)


if __name__ == '__main__':
    test_arraywrite()
    test_arraywcr()
