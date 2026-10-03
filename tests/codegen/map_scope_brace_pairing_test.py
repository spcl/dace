# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A map's encapsulating C scope must open and close as a pair.

The readable CPU generator elides that scope when it would bound nothing
(:meth:`ExperimentalCPUCodeGen.map_scope_needs_brace`), so the ``MapExit`` has to learn what the
matching ``MapEntry`` decided. Keying that on ``id(node.map)`` assumed ``entry.map is exit.map``; a
transformation that deep-copies the two nodes against separate memos breaks the assumption, the
exit's lookup misses, and the fallback emits a ``}`` with no ``{`` -- unbalanced C++ that only
surfaces as a compiler syntax error far from its cause (LoopPeeling did exactly this).
"""
import copy

import numpy as np
import pytest

import dace
from dace.codegen.targets.cpu import CPUCodeGen
from dace.config import set_temporary
from dace.sdfg import nodes

IMPLEMENTATIONS = ('legacy', 'experimental_readable')


def _map_sdfg(name: str) -> dace.SDFG:
    """``B[i] = A[i] + 1`` over one OpenMP map -- a map whose scope declares nothing, so the readable
    generator elides its braces and the pairing is actually exercised."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_array('B', [20], dace.float64)
    state = sdfg.add_state()
    tasklet, _, _ = state.add_mapped_tasklet('m',
                                             dict(i='0:20'),
                                             dict(a=dace.Memlet('A[i]')),
                                             'b = a + 1',
                                             dict(b=dace.Memlet('B[i]')),
                                             external_edges=True)
    return sdfg


def _balance(code: str) -> int:
    return code.count('{') - code.count('}')


@pytest.mark.parametrize('implementation', IMPLEMENTATIONS)
def test_map_scope_braces_balanced(implementation):
    with set_temporary('compiler', 'cpu', 'implementation', value=implementation):
        code = _map_sdfg(f'balanced_{implementation}').generate_code()[0].clean_code
    assert _balance(code) == 0
    assert 'Indentation failure' not in code


def _split_map_identity(sdfg: dace.SDFG) -> None:
    """Give the MapExit its own copy of the Map -- exactly what a per-node deepcopy produces."""
    state = sdfg.nodes()[0]
    exit_node = next(n for n in state.nodes() if isinstance(n, nodes.MapExit))
    exit_node.map = copy.deepcopy(state.entry_node(exit_node).map)
    assert state.entry_node(exit_node).map is not exit_node.map


def test_split_map_identity_is_rejected_by_validation():
    """First line of defence: the SDFG is invalid, and validate_state says so by name."""
    sdfg = _map_sdfg('split_identity_validation')
    sdfg.validate()  # sound before the split
    _split_map_identity(sdfg)
    with pytest.raises(dace.sdfg.validation.InvalidSDFGNodeError, match='same scope object'):
        sdfg.validate()


def test_map_scope_key_is_independent_of_map_object_identity():
    """Second line of defence: the brace pairing must not read ``id(node.map)``.

    Validation now rejects a split identity before codegen can see it, so this asserts the mechanism
    rather than the emitted text: the key an exit computes for its scope is derived from the block
    and node ids, so it still matches its entry's key when the two stop sharing a Map object.
    """
    sdfg = _map_sdfg('scope_key')
    state = sdfg.nodes()[0]
    exit_node = next(n for n in state.nodes() if isinstance(n, nodes.MapExit))
    entry_node = state.entry_node(exit_node)

    key = CPUCodeGen.map_scope_key(sdfg, state.block_id, state, entry_node)
    _split_map_identity(sdfg)
    assert CPUCodeGen.map_scope_key(sdfg, state.block_id, state, state.entry_node(exit_node)) == key


def test_map_scope_key_distinguishes_sibling_maps():
    """...and two maps in one state must not collide onto one key."""
    sdfg = dace.SDFG('scope_key_siblings')
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_array('B', [20], dace.float64)
    sdfg.add_array('C', [20], dace.float64)
    state = sdfg.add_state()
    state.add_mapped_tasklet('m1',
                             dict(i='0:20'),
                             dict(a=dace.Memlet('A[i]')),
                             'b = a + 1',
                             dict(b=dace.Memlet('B[i]')),
                             external_edges=True)
    state.add_mapped_tasklet('m2',
                             dict(i='0:20'),
                             dict(b=dace.Memlet('B[i]')),
                             'c = b * 2',
                             dict(c=dace.Memlet('C[i]')),
                             external_edges=True)

    entries = [n for n in state.nodes() if isinstance(n, nodes.MapEntry)]
    assert len(entries) == 2
    keys = {CPUCodeGen.map_scope_key(sdfg, state.block_id, state, e) for e in entries}
    assert len(keys) == 2


@pytest.mark.parametrize('implementation', IMPLEMENTATIONS)
def test_sibling_maps_in_one_state_stay_balanced(implementation):
    """Two maps in one state: the second must not inherit the first's brace record."""
    sdfg = dace.SDFG(f'siblings_{implementation}')
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_array('B', [20], dace.float64)
    sdfg.add_array('C', [20], dace.float64)
    state = sdfg.add_state()
    state.add_mapped_tasklet('m1',
                             dict(i='0:20'),
                             dict(a=dace.Memlet('A[i]')),
                             'b = a + 1',
                             dict(b=dace.Memlet('B[i]')),
                             external_edges=True)
    state.add_mapped_tasklet('m2',
                             dict(i='0:20'),
                             dict(b=dace.Memlet('B[i]')),
                             'c = b * 2',
                             dict(c=dace.Memlet('C[i]')),
                             external_edges=True)

    with set_temporary('compiler', 'cpu', 'implementation', value=implementation):
        code = sdfg.generate_code()[0].clean_code
    assert _balance(code) == 0
    assert 'Indentation failure' not in code


@pytest.mark.parametrize('implementation', IMPLEMENTATIONS)
def test_map_compiles_and_runs(implementation):
    """The balance check above is structural; this one proves the emitted code is real C++."""
    with set_temporary('compiler', 'cpu', 'implementation', value=implementation):
        csdfg = _map_sdfg(f'runs_{implementation}').compile()
    A = np.random.rand(20)
    B = np.zeros(20)
    csdfg(A=A, B=B)
    assert np.allclose(B, A + 1)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
