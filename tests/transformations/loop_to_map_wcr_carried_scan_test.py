# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""LoopToMap must refuse a loop whose WCR accumulator is read back inside the body.

A WCR memlet is a read-modify-write against its destination (``dace/runtime/include/dace/reduction.h``
does ``*ptr = wcr(*ptr, value)``), not a pure write. LoopToMap exempts WCR writes from both the
unique-write-index test and the pairwise write-overlap test, which is sound only for an associative
reduction over the whole loop. ``for i: out[i] = acc[0]; acc[0] += B[i]`` -- a serial prefix scan --
reads the partial sum back, so its value depends on how many iterations already ran.

Reachable from a plain ``@dace.program``: the frontend emits no WCR, but ``AugAssignToWCR`` turns
the augmented assignment into one, and LoopToMap then lifted the scan into a map. The result still
validated and miscompiled -- 4095 of 4096 elements wrong at 8 threads
(``observed out[:8]: [0. 5. 23. 79. 80. 84. 85. 90.]``, expected ``[0. 1. 2. 3. 4. 5. 6. 7.]``).
"""
import numpy as np
import pytest

import dace
from dace import Memlet
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.dataflow import AugAssignToWCR
from dace.transformation.interstate import LoopToMap

N = 4
Nsym = dace.symbol('Nsym', dtype=dace.int64)


@dace.program
def prefix_scan(B: dace.float64[Nsym], out: dace.float64[Nsym], acc: dace.float64[1]):
    for i in range(Nsym):
        out[i] = acc[0]
        acc[0] += B[i]


@dace.program
def reduction(B: dace.float64[Nsym], acc: dace.float64[1]):
    for i in range(Nsym):
        acc[0] += B[i]


def build_prefix_scan(use_wcr: bool) -> dace.SDFG:
    """Hand-built twin of :func:`prefix_scan`, with and without the WCR on the accumulation."""
    sdfg = dace.SDFG('prefix_scan_' + ('wcr' if use_wcr else 'plain'))
    sdfg.add_array('B', [N], dace.float64)
    sdfg.add_array('acc', [1], dace.float64)
    sdfg.add_array('out', [N], dace.float64)
    sdfg.add_symbol('i', dace.int64)

    loop = LoopRegion('loop', f'i < {N}', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)

    read = loop.add_state('read', is_start_block=True)
    copy_out = read.add_tasklet('copy_out', {'a'}, {'o'}, 'o = a')
    read.add_edge(read.add_access('acc'), None, copy_out, 'a', Memlet('acc[0]'))
    read.add_edge(copy_out, 'o', read.add_access('out'), None, Memlet('out[i]'))

    accumulate = loop.add_state('accumulate')
    loop.add_edge(read, accumulate, dace.InterstateEdge())
    if use_wcr:
        add = accumulate.add_tasklet('add', {'b'}, {'o'}, 'o = b')
        accumulate.add_edge(accumulate.add_access('B'), None, add, 'b', Memlet('B[i]'))
        accumulate.add_edge(add, 'o', accumulate.add_access('acc'), None,
                            Memlet(data='acc', subset='0', wcr='lambda a, b: a + b'))
    else:
        add = accumulate.add_tasklet('add', {'a': None, 'b': None}, {'o'}, 'o = a + b')
        accumulate.add_edge(accumulate.add_access('B'), None, add, 'b', Memlet('B[i]'))
        accumulate.add_edge(accumulate.add_access('acc'), None, add, 'a', Memlet('acc[0]'))
        accumulate.add_edge(add, 'o', accumulate.add_access('acc'), None, Memlet('acc[0]'))
    sdfg.validate()
    return sdfg


def _loops(sdfg: dace.SDFG):
    return [b for b in sdfg.all_control_flow_blocks() if isinstance(b, LoopRegion)]


def _maps_over(sdfg: dace.SDFG, itervar: str):
    return [
        n for state in sdfg.all_states() for n in state.nodes()
        if isinstance(n, nodes.MapEntry) and itervar in n.map.params
    ]


def _wcr_memlets(sdfg: dace.SDFG):
    return [e.data for state in sdfg.all_states() for e in state.edges() if e.data is not None and e.data.wcr]


def test_loop_to_map_refuses_a_wcr_carried_scan_end_to_end():
    """The reachable path: ``@dace.program`` -> ``AugAssignToWCR`` -> ``LoopToMap``."""
    sdfg = prefix_scan.to_sdfg(simplify=True)
    assert sdfg.apply_transformations_repeated(AugAssignToWCR, validate=False) == 1
    assert _wcr_memlets(sdfg), 'the test needs AugAssignToWCR to have produced the WCR accumulation'

    applied = sdfg.apply_transformations_repeated(LoopToMap, validate=False)

    assert applied == 0, 'LoopToMap parallelized a prefix scan whose WCR accumulator is read back'
    assert len(_loops(sdfg)) == 1, 'the sequential loop must survive as a LoopRegion'
    assert not _maps_over(sdfg, 'i'), 'no map may range over the carrying iteration variable'


@pytest.mark.parametrize('use_wcr', [False, True])
def test_loop_to_map_refuses_a_hand_built_wcr_carried_scan(use_wcr):
    sdfg = build_prefix_scan(use_wcr)
    before = sdfg.to_json()

    applied = sdfg.apply_transformations_repeated(LoopToMap, validate=False)

    assert applied == 0, 'LoopToMap parallelized a prefix scan whose accumulator is read back'
    assert len(_loops(sdfg)) == 1, 'the sequential loop must survive as a LoopRegion'
    assert not _maps_over(sdfg, 'i'), 'no map may range over the carrying iteration variable'
    assert sdfg.to_json() == before, 'a transformation that does not apply must not mutate the SDFG'


def test_loop_to_map_still_lifts_a_plain_wcr_reduction():
    """Positive control: a whole-loop reduction nobody reads back must still become a map."""
    sdfg = reduction.to_sdfg(simplify=True)
    assert sdfg.apply_transformations_repeated(AugAssignToWCR, validate=False) == 1

    applied = sdfg.apply_transformations_repeated(LoopToMap, validate=False)

    assert applied == 1, 'LoopToMap must still parallelize a plain WCR reduction'
    assert not _loops(sdfg), 'the loop must be gone'
    assert _maps_over(sdfg, 'i'), 'a map over i must have replaced it'


def test_wcr_prefix_scan_computes_the_sequential_result():
    n = 512
    sdfg = prefix_scan.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(AugAssignToWCR, validate=False)
    sdfg.apply_transformations_repeated(LoopToMap, validate=False)
    b = np.ones(n, dtype=np.float64)
    acc = np.zeros(1, dtype=np.float64)
    out = np.zeros(n, dtype=np.float64)

    sdfg(B=b, out=out, acc=acc, Nsym=n)

    assert np.allclose(out, np.concatenate(([0.0], np.cumsum(b)[:-1])))
    assert np.allclose(acc, b.sum())


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
