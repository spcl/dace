# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest
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
    assert len(sdfg.nodes()) == 1
    sdfg.apply_transformations(LoopUnroll)
    # +1: the empty predecessor LoopUnroll prepends when the loop is the graph's start block.
    assert len(sdfg.nodes()) == 5 * 2 + 1
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
    assert len(sdfg.nodes()) == 1
    sdfg.apply_transformations(LoopPeeling, dict(count=2))
    assert len(sdfg.nodes()) == 3
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
    assert len(sdfg.nodes()) == 1
    sdfg.apply_transformations(LoopPeeling, dict(count=2, begin=False))
    assert len(sdfg.nodes()) == 3
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


@pytest.mark.parametrize('start, condition, step, expected', [
    (0, 'i < 10', 'i + 1', list(range(0, 10, 1))),
    (0, 'i < 10', 'i + 3', list(range(0, 10, 3))),
    (9, 'i > -1', 'i - 1', list(range(9, -1, -1))),
    (9, 'i > -1', 'i - 3', list(range(9, -1, -3))),
    (9, 'i >= 0', 'i - 1', list(range(9, -1, -1))),
    (5, 'i <= 5', 'i + 1', [5]),
])
def test_unroll_covers_every_iteration(start, condition, step, expected):
    """Every iteration is unrolled, counting up or down, at any stride."""
    sdfg = dace.SDFG(f'unroll_{start}_{step.replace(" ", "").replace("-", "m").replace("+", "p")}')
    sdfg.add_array('A', [10], dace.float64)
    loop = dace.sdfg.state.LoopRegion('l', condition, 'i', f'i = {start}', f'i = {step}')
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state('body', is_start_block=True)
    tasklet = body.add_tasklet('write', {}, {'o'}, 'o = 1.0')
    body.add_edge(tasklet, 'o', body.add_write('A'), None, dace.Memlet('A[i]'))

    assert sdfg.apply_transformations_repeated([LoopUnroll]) == 1

    A = np.zeros([10])
    sdfg(A=A)
    written = sorted(int(i) for i, v in enumerate(A) if v != 0)
    assert written == sorted(expected)
