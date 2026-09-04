# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""RedundantArrayCopyingIn must refuse a chain whose ``in_array`` is written in another state.

``apply`` deletes ``in_array`` and ``med_array`` and redirects the writers of ``in_array`` it finds
IN THIS STATE. A producer in a different state has no edge here to redirect, so the whole copy is
dropped: the consumer of ``out_array`` then reads whatever was there before. The SDFG still
validates, so the failure is silent wrong numbers -- before the fix the reproducer below returned
``res: [0. 0. 0. 0. 0. 0. 0. 0.]`` where ``[1. 2. 3. 4. 5. 6. 7. 8.]`` was expected.
"""
import numpy as np
import pytest

import dace
from dace import Memlet
from dace.sdfg import nodes
from dace.transformation.dataflow import RedundantArrayCopyingIn

N = 8


def full(name: str) -> Memlet:
    return Memlet(f'{name}[0:{N}] -> [0:{N}]')


def build(cross_state: bool) -> dace.SDFG:
    """``A -> B -> out -> res``, with ``A`` produced either in a previous state or in the same one."""
    sdfg = dace.SDFG('rac_in_' + ('cross_state' if cross_state else 'same_state'))
    sdfg.add_array('inp', [N], dace.float64)
    sdfg.add_array('out', [N], dace.float64)
    sdfg.add_array('res', [N], dace.float64)
    sdfg.add_transient('A', [N], dace.float64)
    sdfg.add_transient('B', [N], dace.float64)

    state = sdfg.add_state('copy', is_start_block=True)
    if cross_state:
        producer = sdfg.add_state_before(state, 'produce')
        producer.add_nedge(producer.add_access('inp'), producer.add_access('A'), full('inp'))
        a = state.add_access('A')
    else:
        a = state.add_access('A')
        state.add_nedge(state.add_access('inp'), a, full('inp'))
    b, o, r = state.add_access('B'), state.add_access('out'), state.add_access('res')
    state.add_nedge(a, b, full('A'))
    state.add_nedge(b, o, full('B'))
    state.add_nedge(o, r, full('out'))
    sdfg.validate()
    return sdfg


def _access_names(sdfg: dace.SDFG):
    return {n.data for state in sdfg.all_states() for n in state.nodes() if isinstance(n, nodes.AccessNode)}


def run(sdfg: dace.SDFG) -> np.ndarray:
    inp = np.arange(1, N + 1, dtype=np.float64)
    out = np.zeros(N, dtype=np.float64)
    res = np.zeros(N, dtype=np.float64)
    sdfg(inp=inp, out=out, res=res)
    return res


def test_refuses_a_chain_whose_producer_is_in_another_state():
    sdfg = build(cross_state=True)
    before = sdfg.to_json()

    applied = sdfg.apply_transformations_repeated(RedundantArrayCopyingIn, validate=False)

    assert applied == 0, 'RedundantArrayCopyingIn dropped a copy whose producer is in another state'
    assert {'A', 'B'} <= _access_names(sdfg), 'the chain that carries the value must survive'
    assert sdfg.to_json() == before, 'a transformation that does not apply must not mutate the SDFG'


def test_cross_state_chain_still_computes_the_right_numbers():
    sdfg = build(cross_state=True)
    sdfg.apply_transformations_repeated(RedundantArrayCopyingIn, validate=False)
    sdfg.validate()

    assert np.allclose(run(sdfg), np.arange(1, N + 1, dtype=np.float64))


def test_still_folds_a_chain_wholly_inside_one_state():
    """Positive control: with the producer in the same state every writer is redirected."""
    sdfg = build(cross_state=False)

    applied = sdfg.apply_transformations_repeated(RedundantArrayCopyingIn, validate=False)
    sdfg.validate()

    assert applied == 1, 'RedundantArrayCopyingIn must still fold a chain it can fully account for'
    assert not {'A', 'B'} & _access_names(sdfg), 'the folded chain must be gone'
    assert np.allclose(run(sdfg), np.arange(1, N + 1, dtype=np.float64))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
