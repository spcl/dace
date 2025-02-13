# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


def test_nested_symbol_partial():
    W = dace.symbol()

    @dace.program
    def nested(out: dace.float64[1]):
        tmp = np.ndarray([W], dtype=np.float64)
        for i in dace.map[0:W]:
            with dace.tasklet:
                o >> tmp[i]
                o = i
        dace.reduce(lambda a, b: a + b, tmp, out, identity=0)

    @dace.program
    def nested_symbol_partial(A: dace.float64[1]):
        nested(A, W=W)

    expected = np.arange(0, 20, dtype=np.float64).sum()
    out = np.ndarray([1], dtype=np.float64)
    nested_symbol_partial(A=out, sym_0=20)
    assert np.allclose(out, expected)


if __name__ == '__main__':
    test_nested_symbol_partial()
