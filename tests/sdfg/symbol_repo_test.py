# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import json

import pytest

import dace
from dace import symbolic
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion, SymbolResolver
from dace.sdfg.validation import InvalidSDFGError

POSITIVE = frozenset({symbolic.Predicate.POSITIVE})


def reloaded(sdfg: dace.SDFG) -> dace.SDFG:
    return dace.SDFG.from_json(json.loads(json.dumps(sdfg.to_json())))


def relation(kind: symbolic.RelationKind, lhs: str, rhs: str) -> symbolic.Relation:
    return symbolic.Relation(kind, symbolic.pystr_to_symbolic(lhs), symbolic.pystr_to_symbolic(rhs))


def loop_in_loop_sdfg() -> dace.SDFG:
    """ ``for i in 0:N: (k = i + 1) for i in 0:k: map j in 0:i`` -- the inner loop shadows ``i``. """
    sdfg = dace.SDFG('loop_in_loop')
    sdfg.add_symbol('N', dace.int64, predicates=POSITIVE)
    sdfg.add_array('A', ['N'], dace.float64)
    outer = LoopRegion('outer', 'i < N', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(outer, is_start_block=True)
    first = outer.add_state('first', is_start_block=True)
    inner = LoopRegion('inner', 'i < k', 'i', 'i = 0', 'i = i + 1')
    outer.add_node(inner)
    edge = outer.add_edge(first, inner, dace.InterstateEdge(assignments={'k': 'i + 1'}))
    body = inner.add_state('body', is_start_block=True)
    _, entry, _ = body.add_mapped_tasklet('m', {'j': '0:i'}, {},
                                          'a = 1.0', {'a': dace.Memlet('A[j]')},
                                          external_edges=True)

    repo = sdfg.symbol_repo
    repo.add('i', dace.int64, at=outer)
    repo.add_relation(relation(symbolic.RelationKind.LT, 'i', 'N'), at=outer)
    repo.add('i', dace.int32, at=inner)
    repo.add_relation(relation(symbolic.RelationKind.LT, 'i', 'k'), at=inner)
    repo.add('j', dace.int16, at=entry)
    sdfg.validate()
    return sdfg


def scope_owners(sdfg: dace.SDFG):
    outer = next(block for block in sdfg.nodes() if isinstance(block, LoopRegion))
    inner = next(block for block in outer.nodes() if isinstance(block, LoopRegion))
    entry = next(node for node in inner.start_block.nodes() if isinstance(node, nodes.MapEntry))
    return outer, inner, entry


def test_params_and_facts_round_trip():
    sdfg = dace.SDFG('params_round_trip')
    sdfg.add_state(is_start_block=True)
    sdfg.add_symbol('N', dace.int64, predicates=POSITIVE)
    sdfg.add_symbol('K', dace.uint32)
    sdfg.add_symbol_relation(relation(symbolic.RelationKind.LE, 'K', 'N'))
    loaded = reloaded(sdfg)
    assert loaded.symbols == {'K': dace.uint32, 'N': dace.int64}
    assert loaded.symbol_repo.params.predicates == {'N': POSITIVE}
    assert list(loaded.symbol_repo.params.relations) == [relation(symbolic.RelationKind.LE, 'K', 'N')]
    assert json.dumps(loaded.to_json()) == json.dumps(sdfg.to_json())


def test_symbols_is_a_read_only_view_of_the_params():
    sdfg = dace.SDFG('read_only_symbols')
    sdfg.add_symbol('N', dace.int64)
    with pytest.raises(TypeError):
        sdfg.symbols['M'] = dace.int64
    sdfg.symbol_repo.add('M', dace.int32)
    assert sdfg.symbols == {'N': dace.int64, 'M': dace.int32}


def test_scopes_round_trip_to_their_owners():
    sdfg = loop_in_loop_sdfg()
    loaded = reloaded(sdfg)
    outer, inner, entry = scope_owners(loaded)
    repo = loaded.symbol_repo
    assert list(repo.scopes) == [outer, inner, entry]
    assert repo.scopes[outer].types == {'i': dace.int64}
    assert repo.scopes[inner].types == {'i': dace.int32}
    assert repo.scopes[entry].types == {'j': dace.int16}
    assert json.dumps(loaded.to_json()) == json.dumps(sdfg.to_json())


def test_inner_scope_shadows_outer_after_round_trip():
    loaded = reloaded(loop_in_loop_sdfg())
    outer, inner, entry = scope_owners(loaded)
    resolver = SymbolResolver()
    i_below_n = relation(symbolic.RelationKind.LE, 'i', 'N - 1')
    # The outer ``i < N`` holds in the outer loop, but not in the inner one, which reopens ``i``
    assert symbolic.ask(i_below_n, resolver.facts_at(outer.start_block)) is symbolic.Truth.TRUE
    at_map = resolver.facts_at(inner.start_block, entry)
    assert symbolic.ask(i_below_n, at_map) is symbolic.Truth.UNKNOWN
    assert symbolic.ask(relation(symbolic.RelationKind.LE, 'i', 'k - 1'), at_map) is symbolic.Truth.TRUE


def test_deepcopy_keys_scopes_by_the_copied_owners():
    sdfg = loop_in_loop_sdfg()
    copied = copy.deepcopy(sdfg)
    assert list(copied.symbol_repo.scopes) == list(scope_owners(copied))
    copied.validate()


def test_nested_sdfg_round_trips_its_own_repo():
    inner = loop_in_loop_sdfg()
    sdfg = dace.SDFG('nesting')
    sdfg.add_symbol('M', dace.int64, predicates=POSITIVE)
    sdfg.add_array('A', ['M'], dace.float64)
    state = sdfg.add_state(is_start_block=True)
    nsdfg = state.add_nested_sdfg(inner, {}, {'A': None}, symbol_mapping={'N': 'M'})
    state.add_edge(nsdfg, 'A', state.add_write('A'), None, dace.Memlet('A[0:M]'))
    sdfg.validate()
    loaded = reloaded(sdfg)
    loaded_inner = loaded.start_block.nodes()[0].sdfg
    assert loaded.symbol_repo.params.types == {'M': dace.int64}
    assert not loaded.symbol_repo.scopes
    assert list(loaded_inner.symbol_repo.scopes) == list(scope_owners(loaded_inner))
    assert json.dumps(loaded.to_json()) == json.dumps(sdfg.to_json())


def test_validation_rejects_scopes_that_do_not_match_their_owners():
    sdfg = loop_in_loop_sdfg()
    outer, inner, entry = scope_owners(sdfg)
    with pytest.raises(ValueError, match='does not bind'):
        sdfg.symbol_repo.add('k2', dace.int64, at=entry)
    outer.remove_node(inner)
    with pytest.raises(InvalidSDFGError, match='no owner in the SDFG'):
        sdfg.validate()
    # Saving leaves out the scopes of removed owners
    loaded = reloaded(sdfg)
    assert list(loaded.symbol_repo.scopes) == [loaded.start_block]


def test_edge_scopes_declare_no_facts():
    sdfg = loop_in_loop_sdfg()
    edge = scope_owners(sdfg)[0].edges()[0].data
    with pytest.raises(NotImplementedError):
        sdfg.symbol_repo.add('k', dace.int64, POSITIVE, at=edge)


if __name__ == '__main__':
    test_params_and_facts_round_trip()
    test_symbols_is_a_read_only_view_of_the_params()
    test_scopes_round_trip_to_their_owners()
    test_inner_scope_shadows_outer_after_round_trip()
    test_deepcopy_keys_scopes_by_the_copied_owners()
    test_nested_sdfg_round_trips_its_own_repo()
    test_validation_rejects_scopes_that_do_not_match_their_owners()
    test_edge_scopes_declare_no_facts()
