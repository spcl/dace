# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

value = dace.symbol('value', dtype=dace.float32)


@dace.program
def symintasklet_numpy(out: dace.float32[1]):
    out[0] = value


@dace.program
def symintasklet_explicit(out: dace.float32[1]):
    with dace.tasklet:
        o = value
        o >> out[0]


def test_numpy():
    out = np.zeros(1).astype(np.float32)
    symintasklet_numpy(out, value=np.float32(1.5))
    assert out[0] == np.float32(1.5)


def test_explicit():
    out = np.zeros(1).astype(np.float32)
    symintasklet_explicit(out, value=np.float32(1.5))
    assert out[0] == np.float32(1.5)


if __name__ == '__main__':
    test_numpy()
    test_explicit()
