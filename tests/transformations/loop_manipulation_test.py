# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from dace.transformation.interstate.loop_unroll import LoopUnroll
from dace.transformation.interstate.loop_peeling import LoopPeeling


@dace.program
def tounroll(A: dace.float64[20], B: dace.float64[20]):
    for i in range(5):
        for j in dace.map[0:20]:
            with dace.tasklet:
                a << A[j]
                b_in << B[j]
                b_out >> B[j]
                b_out = b_in + a * i


def regression(A, B):
    result = np.zeros_like(B)
    result[:] = B
    for i in range(5):
        result += A * i
    return result


def test_unroll():
    sdfg: dace.SDFG = tounroll.to_sdfg()
    sdfg.simplify()
    assert len(sdfg.nodes()) == 4
    sdfg.apply_transformations(LoopUnroll)
    assert len(sdfg.nodes()) == (5 + 2)
    sdfg.simplify()
    assert len(sdfg.nodes()) == 1
    A = np.random.rand(20)
    B = np.random.rand(20)
    reg = regression(A, B)

    # HACK: Workaround to deal with bug in frontend (See PR #161)
    if 'i' in sdfg.symbols:
        del sdfg.symbols['i']

    sdfg(A=A, B=B)
    assert np.allclose(B, reg)


def test_peeling_start():
    sdfg: dace.SDFG = tounroll.to_sdfg()
    sdfg.simplify()
    assert len(sdfg.nodes()) == 4
    sdfg.apply_transformations(LoopPeeling, dict(count=2))
    assert len(sdfg.nodes()) == 6
    sdfg.simplify()
    assert len(sdfg.nodes()) == 4
    A = np.random.rand(20)
    B = np.random.rand(20)
    reg = regression(A, B)

    # HACK: Workaround to deal with bug in frontend (See PR #161)
    if 'i' in sdfg.symbols:
        del sdfg.symbols['i']

    sdfg(A=A, B=B)
    assert np.allclose(B, reg)


def test_peeling_end():
    sdfg: dace.SDFG = tounroll.to_sdfg()
    sdfg.simplify()
    assert len(sdfg.nodes()) == 4
    sdfg.apply_transformations(LoopPeeling, dict(count=2, begin=False))
    assert len(sdfg.nodes()) == 6
    sdfg.simplify()
    assert len(sdfg.nodes()) == 4
    A = np.random.rand(20)
    B = np.random.rand(20)
    reg = regression(A, B)

    # HACK: Workaround to deal with bug in frontend (See PR #161)
    if 'i' in sdfg.symbols:
        del sdfg.symbols['i']

    sdfg(A=A, B=B)
    assert np.allclose(B, reg)


if __name__ == '__main__':
    test_unroll()
    test_peeling_start()
    test_peeling_end()
