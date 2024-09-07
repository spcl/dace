# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import pytest

from dace.transformation.interstate.loop_detection import DetectLoop
from dace.transformation import transformation as xf


class CountLoops(DetectLoop, xf.MultiStateTransformation):

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        return super().can_be_applied(graph, expr_index, sdfg, permissive)


def test_pyloop():

    @dace.program
    def tester(a: dace.float64[20]):
        for i in range(1, 20):
            a[i] = a[i - 1] + 1

    sdfg = tester.to_sdfg()
    xform = CountLoops()
    assert sdfg.apply_transformations(xform) == 1
    itvar, rng, _ = xform.loop_information()
    assert itvar == 'i'
    assert rng == (1, 19, 1)


def test_loop_rotated():
    sdfg = dace.SDFG('tester')
    sdfg.add_symbol('N', dace.int32)

    entry = sdfg.add_state('entry', is_start_block=True)
    body = sdfg.add_state('body')
    latch = sdfg.add_state('latch')
    exitstate = sdfg.add_state('exitstate')

    sdfg.add_edge(entry, body, dace.InterstateEdge(assignments=dict(i=0)))
    sdfg.add_edge(body, latch, dace.InterstateEdge())
    sdfg.add_edge(latch, body, dace.InterstateEdge('i < N', assignments=dict(i='i + 2')))
    sdfg.add_edge(latch, exitstate, dace.InterstateEdge('i >= N'))

    xform = CountLoops()
    assert sdfg.apply_transformations(xform) == 1
    itvar, rng, _ = xform.loop_information()
    assert itvar == 'i'
    assert rng == (0, dace.symbol('N') - 1, 2)


@pytest.mark.skip('Extra incrementation states should not be supported by loop detection')
def test_loop_rotated_extra_increment():
    sdfg = dace.SDFG('tester')
    sdfg.add_symbol('N', dace.int32)

    entry = sdfg.add_state('entry', is_start_block=True)
    body = sdfg.add_state('body')
    latch = sdfg.add_state('latch')
    increment = sdfg.add_state('increment')
    exitstate = sdfg.add_state('exitstate')

    sdfg.add_edge(entry, body, dace.InterstateEdge(assignments=dict(i=0)))
    sdfg.add_edge(latch, increment, dace.InterstateEdge('i < N'))
    sdfg.add_edge(increment, body, dace.InterstateEdge(assignments=dict(i='i + 1')))
    sdfg.add_edge(latch, exitstate, dace.InterstateEdge('i >= N'))

    xform = CountLoops()
    assert sdfg.apply_transformations(xform) == 1
    itvar, rng, _ = xform.loop_information()
    assert itvar == 'i'
    assert rng == (0, dace.symbol('N') - 1, 1)


def test_self_loop():
    # Tests a single-state loop
    sdfg = dace.SDFG('tester')
    sdfg.add_symbol('N', dace.int32)

    entry = sdfg.add_state('entry', is_start_block=True)
    body = sdfg.add_state('body')
    exitstate = sdfg.add_state('exitstate')

    sdfg.add_edge(entry, body, dace.InterstateEdge(assignments=dict(i=2)))
    sdfg.add_edge(body, body, dace.InterstateEdge('i < N', assignments=dict(i='i + 3')))
    sdfg.add_edge(body, exitstate, dace.InterstateEdge('i >= N'))

    xform = CountLoops()
    assert sdfg.apply_transformations(xform) == 1
    itvar, rng, _ = xform.loop_information()
    assert itvar == 'i'
    assert rng == (2, dace.symbol('N') - 1, 3)


def test_loop_llvm_canonical():
    sdfg = dace.SDFG('tester')
    sdfg.add_symbol('N', dace.int32)

    entry = sdfg.add_state('entry', is_start_block=True)
    guard = sdfg.add_state_after(entry, 'guard')
    preheader = sdfg.add_state('preheader')
    body = sdfg.add_state('body')
    latch = sdfg.add_state('latch')
    loopexit = sdfg.add_state('loopexit')
    exitstate = sdfg.add_state('exitstate')

    sdfg.add_edge(guard, exitstate, dace.InterstateEdge('N <= 0'))
    sdfg.add_edge(guard, preheader, dace.InterstateEdge('N > 0'))
    sdfg.add_edge(preheader, body, dace.InterstateEdge(assignments=dict(i=0)))
    sdfg.add_edge(body, latch, dace.InterstateEdge())
    sdfg.add_edge(latch, body, dace.InterstateEdge('i < N', assignments=dict(i='i + 1')))
    sdfg.add_edge(latch, loopexit, dace.InterstateEdge('i >= N'))
    sdfg.add_edge(loopexit, exitstate, dace.InterstateEdge())

    xform = CountLoops()
    assert sdfg.apply_transformations(xform) == 1
    itvar, rng, _ = xform.loop_information()
    assert itvar == 'i'
    assert rng == (0, dace.symbol('N') - 1, 1)


@pytest.mark.skip('Extra incrementation states should not be supported by loop detection')
@pytest.mark.parametrize('with_bounds_check', (False, True))
def test_loop_llvm_canonical_with_extras(with_bounds_check):
    sdfg = dace.SDFG('tester')
    sdfg.add_symbol('N', dace.int32)

    entry = sdfg.add_state('entry', is_start_block=True)
    guard = sdfg.add_state_after(entry, 'guard')
    preheader = sdfg.add_state('preheader')
    body = sdfg.add_state('body')
    latch = sdfg.add_state('latch')
    increment1 = sdfg.add_state('increment1')
    increment2 = sdfg.add_state('increment2')
    loopexit = sdfg.add_state('loopexit')
    exitstate = sdfg.add_state('exitstate')

    if with_bounds_check:
        sdfg.add_edge(guard, exitstate, dace.InterstateEdge('N <= 0'))
        sdfg.add_edge(guard, preheader, dace.InterstateEdge('N > 0'))
    else:
        sdfg.add_edge(guard, preheader, dace.InterstateEdge())
    sdfg.add_edge(preheader, body, dace.InterstateEdge(assignments=dict(i=0)))
    sdfg.add_edge(body, latch, dace.InterstateEdge())
    sdfg.add_edge(latch, increment1, dace.InterstateEdge('i < N'))
    sdfg.add_edge(increment1, increment2, dace.InterstateEdge(assignments=dict(i='i + 1')))
    sdfg.add_edge(increment2, body, dace.InterstateEdge())
    sdfg.add_edge(latch, loopexit, dace.InterstateEdge('i >= N'))
    sdfg.add_edge(loopexit, exitstate, dace.InterstateEdge())

    xform = CountLoops()
    assert sdfg.apply_transformations(xform) == 1
    itvar, rng, _ = xform.loop_information()
    assert itvar == 'i'
    assert rng == (0, dace.symbol('N') - 1, 1)


if __name__ == '__main__':
    test_pyloop()
    test_loop_rotated()
    # test_loop_rotated_extra_increment()
    test_self_loop()
    test_loop_llvm_canonical()
    # test_loop_llvm_canonical_with_extras(False)
    # test_loop_llvm_canonical_with_extras(True)
