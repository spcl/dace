# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Canonicalization gives every interstate-symbol web its own name (``SymbolSSA`` at the ``ssa`` stage)."""
import copy

import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N', dtype=dace.int64)


def gather(state: dace.SDFGState, index: str):
    tasklet = state.add_tasklet('gather', {'inp'}, {'o'}, 'o = inp + 1.0')
    state.add_edge(state.add_read('b'), None, tasklet, 'inp', dace.Memlet('b[s]'))
    state.add_edge(tasklet, 'o', state.add_write('out'), None, dace.Memlet(f'out[{index}]'))


def peeled_gather_sdfg() -> dace.SDFG:
    """``out[i] = b[idx[i]] + 1`` with iteration 0 peeled in front: peel and body both assign ``s = idx[.]``."""
    sdfg = dace.SDFG('peeled_gather')
    sdfg.add_array('idx', [N], dace.int64)
    sdfg.add_array('b', [N], dace.float64)
    sdfg.add_array('out', [N], dace.float64)
    sdfg.add_symbol('s', dace.int64)
    init = sdfg.add_state('init', is_start_block=True)
    peel = sdfg.add_state('peel')
    sdfg.add_edge(init, peel, dace.InterstateEdge(assignments={'s': 'idx[0]'}))
    gather(peel, '0')
    loop = LoopRegion('body_loop', 'i < N', 'i', 'i = 1', 'i = i + 1')
    sdfg.add_node(loop)
    sdfg.add_edge(peel, loop, dace.InterstateEdge())
    head = loop.add_state('head', is_start_block=True)
    comp = loop.add_state('comp')
    loop.add_edge(head, comp, dace.InterstateEdge(assignments={'s': 'idx[i]'}))
    gather(comp, 'i')
    sdfg.validate()
    return sdfg


def gather_symbols(sdfg: dace.SDFG):
    return [(k, v) for nested in sdfg.all_sdfgs_recursive() for e in nested.all_interstate_edges()
            for k, v in e.data.assignments.items() if 'idx[' in v]


def test_peel_and_body_gather_symbols_get_distinct_names():
    """The peel's ``s = idx[0]`` and the body's ``s = idx[i]`` leave canonicalize under two names."""
    sdfg = peeled_gather_sdfg()
    reference = copy.deepcopy(sdfg)
    canonicalize(sdfg, validate=True)

    defs = gather_symbols(sdfg)
    assert len(defs) == 2, defs
    assert len({name for name, _ in defs}) == 2, f'peel and body still share a symbol name: {defs}'

    n = 9
    rng = np.random.default_rng(0)
    idx = rng.permutation(n).astype(np.int64)
    b = rng.random(n)
    expected, got = np.zeros(n), np.zeros(n)
    reference(idx=idx, b=b, out=expected, N=n)
    sdfg(idx=idx, b=b, out=got, N=n)
    assert np.array_equal(expected, b[idx] + 1.0)
    assert np.array_equal(got, expected)
