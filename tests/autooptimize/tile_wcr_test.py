# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests write-conflict resolution tiling """
import dace
from dace.transformation import auto_optimize as aopt
import numpy as np
import pytest

N = dace.symbol('N')


def _runtest(sdfg: dace.SDFG, n: int, add_symbol: bool = True):
    A = np.random.rand(n).astype(np.float32)
    out = np.zeros([1], dtype=np.float32)
    if add_symbol:
        sdfg(A=A, out=out, N=n)
    else:
        sdfg(A=A, out=out)
    assert np.allclose(out, np.sum(A))


def test_shortmap():
    @dace.program
    def sum(A: dace.float32[4], out: dace.float32[1]):
        for i in dace.map[0:4]:
            out += A[i]

    sdfg = sum.to_sdfg()
    aopt.auto_optimize(sdfg, dace.DeviceType.CPU)
    assert 'atomic' not in sdfg.generate_code()[0].code
    _runtest(sdfg, 4, False)


def test_symmap():
    @dace.program
    def sum(A: dace.float32[N], out: dace.float32[1]):
        for i in dace.map[0:N]:
            out += A[i]

    sdfg = sum.to_sdfg()
    aopt.auto_optimize(sdfg, dace.DeviceType.CPU)
    code: str = sdfg.generate_code()[0].code
    assert 'reduce(' in code and code.count('atomic') == 1
    _runtest(sdfg, 257)


@pytest.mark.skip
def test_libnode():
    @dace.program
    def sum(A: dace.float32[N], out: dace.float32[1]):
        dace.reduce(lambda a, b: a + b, A, out, identity=0)

    sdfg = sum.to_sdfg()
    aopt.auto_optimize(sdfg, dace.DeviceType.CPU)
    code: str = sdfg.generate_code()[0].code
    assert 'reduce(' in code and code.count('atomic') == 1
    _runtest(sdfg, 257)


if __name__ == '__main__':
    test_symmap()
    test_shortmap()
    # test_libnode() # Not yet supported
