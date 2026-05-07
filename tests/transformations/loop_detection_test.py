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

    tester.use_explicit_cf = False
    sdfg = tester.to_sdfg(simplify=False)
    xform = CountLoops()
    assert sdfg.apply_transformations(xform) == 1
    itvar, rng, _ = xform.loop_information()
    assert itvar == 'i'
    assert rng == (1, 19, 1)


@pytest.mark.parametrize('increment_before_condition', (True, False))
def test_loop_rotated(increment_before_condition):
    sdfg = dace.SDFG('tester')
    sdfg.add_symbol('N', dace.int32)

    entry = sdfg.add_state('entry', is_start_block=True)
    body = sdfg.add_state('body')
    latch = sdfg.add_state('latch')
    exitstate = sdfg.add_state('exitstate')

    sdfg.add_edge(entry, body, dace.InterstateEdge(assignments=dict(i=0)))
    if increment_before_condition:
        sdfg.add_edge(body, latch, dace.InterstateEdge(assignments=dict(i='i + 2')))
        sdfg.add_edge(latch, body, dace.InterstateEdge('i < N'))
    else:
        sdfg.add_edge(body, latch, dace.InterstateEdge())
        sdfg.add_edge(latch, body, dace.InterstateEdge('i < N', assignments=dict(i='i + 2')))
    sdfg.add_edge(latch, exitstate, dace.InterstateEdge('i >= N'))

    xform = CountLoops()
    assert sdfg.apply_transformations(xform) == 1
    itvar, rng, _ = xform.loop_information()
    assert itvar == 'i'
    assert rng == (0, dace.symbol('N') - 1, 2)


def test_loop_rotated_extra_increment():
    # Extra incrementation states (i.e., something more than a single edge between the latch and the body) should not
    # be allowed and consequently not be detected as loops.
    sdfg = dace.SDFG('tester')
    sdfg.add_symbol('N', dace.int32)

    entry = sdfg.add_state('entry', is_start_block=True)
    body = sdfg.add_state('body')
    latch = sdfg.add_state('latch')
    increment = sdfg.add_state('increment')
    exitstate = sdfg.add_state('exitstate')

    sdfg.add_edge(entry, body, dace.InterstateEdge(assignments=dict(i=0)))
    sdfg.add_edge(body, latch, dace.InterstateEdge())
    sdfg.add_edge(latch, increment, dace.InterstateEdge('i < N'))
    sdfg.add_edge(increment, body, dace.InterstateEdge(assignments=dict(i='i + 1')))
    sdfg.add_edge(latch, exitstate, dace.InterstateEdge('i >= N'))

    xform = CountLoops()
    assert sdfg.apply_transformations(xform) == 0


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


@pytest.mark.parametrize('increment_before_condition', (True, False))
def test_loop_llvm_canonical(increment_before_condition):
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
    if increment_before_condition:
        sdfg.add_edge(body, latch, dace.InterstateEdge(assignments=dict(i='i + 1')))
        sdfg.add_edge(latch, body, dace.InterstateEdge('i < N'))
    else:
        sdfg.add_edge(body, latch, dace.InterstateEdge())
        sdfg.add_edge(latch, body, dace.InterstateEdge('i < N', assignments=dict(i='i + 1')))
    sdfg.add_edge(latch, loopexit, dace.InterstateEdge('i >= N'))
    sdfg.add_edge(loopexit, exitstate, dace.InterstateEdge())

    xform = CountLoops()
    assert sdfg.apply_transformations(xform) == 1
    itvar, rng, _ = xform.loop_information()
    assert itvar == 'i'
    assert rng == (0, dace.symbol('N') - 1, 1)


@pytest.mark.parametrize('with_bounds_check', (False, True))
def test_loop_llvm_canonical_with_extras(with_bounds_check):
    # Extra incrementation states (i.e., something more than a single edge between the latch and the body) should not
    # be allowed and consequently not be detected as loops.
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
    assert sdfg.apply_transformations(xform) == 0


if __name__ == '__main__':
    test_pyloop()
    test_loop_rotated(True)
    test_loop_rotated(False)
    test_loop_rotated_extra_increment()
    test_self_loop()
    test_loop_llvm_canonical(True)
    test_loop_llvm_canonical(False)
    test_loop_llvm_canonical_with_extras(False)
    test_loop_llvm_canonical_with_extras(True)
