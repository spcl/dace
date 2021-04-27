# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests write-conflict resolution tiling """
import dace
from dace.transformation.auto import auto_optimize as aopt
import numpy as np

N = dace.symbol('N')


def _runtest(sdfg: dace.SDFG, n: int, add_symbol: bool = True):
    A = np.random.rand(n).astype(np.float32)
    out = np.zeros([1], dtype=np.float32)
    if add_symbol:
        sdfg(A=A, out=out, N=n)
    else:
        sdfg(A=A, out=out)
    assert np.allclose(out, np.sum(A))


def _runtest2d(sdfg: dace.SDFG, n: int, m: int):
    A = np.random.rand(n, m).astype(np.float32)
    out = np.zeros([m], dtype=np.float32)
    sdfg(A=A, out=out, N=n)
    assert np.allclose(out, np.sum(A, axis=0))


def test_shortmap():
    @dace.program
    def sum(A: dace.float32[4], out: dace.float32[1]):
        for i in dace.map[0:4]:
            out += A[i]

    sdfg = sum.to_sdfg()
    aopt.auto_optimize(sdfg, dace.DeviceType.CPU)
    assert 'atomic' not in sdfg.generate_code()[0].code
    _runtest(sdfg, 4, False)
    del sdfg


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
    del sdfg


def test_libnode():
    @dace.program
    def sum(A: dace.float32[N], out: dace.float32[1]):
        dace.reduce(lambda a, b: a + b, A, out, identity=0)

    sdfg = sum.to_sdfg()
    sdfg.expand_library_nodes()
    aopt.auto_optimize(sdfg, dace.DeviceType.CPU)
    code: str = sdfg.generate_code()[0].code
    assert 'reduce(' in code and code.count('atomic') == 1
    _runtest(sdfg, 257)
    del sdfg


def test_block_reduction():
    @dace.program
    def sum(A: dace.float32[N, N], out: dace.float32[N]):
        for i, j in dace.map[0:N, 0:N]:
            out[j] += A[i, j]

    sdfg = sum.to_sdfg()
    aopt.auto_optimize(sdfg, dace.DeviceType.CPU)
    code: str = sdfg.generate_code()[0].code
    if dace.Config.get_bool('optimizer', 'autotile_partial_parallelism'):
        assert 'reduce(' in code and code.count('atomic') == 0
    _runtest2d(sdfg, 257, 257)
    del sdfg


def test_block_reduction_short():
    @dace.program
    def sum(A: dace.float32[N, 2], out: dace.float32[2]):
        for i, j in dace.map[0:N, 0:2]:
            out[j] += A[i, j]

    sdfg = sum.to_sdfg()
    aopt.auto_optimize(sdfg, dace.DeviceType.CPU)
    code: str = sdfg.generate_code()[0].code
    assert 'reduce(' in code and code.count('atomic') == 1
    _runtest2d(sdfg, 257, 2)
    del sdfg


if __name__ == '__main__':
    test_symmap()
    test_shortmap()
    test_libnode()
    test_block_reduction()
    test_block_reduction_short()
