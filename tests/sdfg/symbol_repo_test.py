# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import json

import pytest

import dace
from dace import symbolic
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.sdfg.symbol_repo import SymbolInfo
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
    repo.open_scope(outer)
    repo.add('i', dace.int64, at=outer)
    repo.add_relation(relation(symbolic.RelationKind.LT, 'i', 'N'), at=outer)
    repo.open_scope(edge.data, parent=outer)
    repo.add('k', dace.int64, POSITIVE, at=edge.data)
    repo.open_scope(inner, parent=edge.data)
    repo.add('i', dace.int32, at=inner)
    repo.add_relation(relation(symbolic.RelationKind.LT, 'i', 'k'), at=inner)
    repo.open_scope(entry, parent=inner)
    repo.add('j', dace.int16, at=entry)
    sdfg.validate()
    return sdfg


def scope_owners(sdfg: dace.SDFG):
    outer = next(block for block in sdfg.nodes() if isinstance(block, LoopRegion))
    inner = next(block for block in outer.nodes() if isinstance(block, LoopRegion))
    entry = next(node for node in inner.start_block.nodes() if isinstance(node, nodes.MapEntry))
    return outer, outer.edges()[0].data, inner, entry


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


def test_scopes_round_trip_to_their_owners():
    sdfg = loop_in_loop_sdfg()
    loaded = reloaded(sdfg)
    outer, edge, inner, entry = scope_owners(loaded)
    repo = loaded.symbol_repo
    assert list(repo.scopes) == [outer, edge, inner, entry]
    assert [repo.scopes[owner].parent for owner in (outer, edge, inner, entry)] == [None, outer, edge, inner]
    assert repo.lookup('i', at=outer) == SymbolInfo(dace.int64, frozenset())
    assert repo.lookup('k', at=edge) == SymbolInfo(dace.int64, POSITIVE)
    assert repo.lookup('j', at=entry) == SymbolInfo(dace.int16, frozenset())
    assert json.dumps(loaded.to_json()) == json.dumps(sdfg.to_json())


def test_inner_scope_shadows_outer_after_round_trip():
    loaded = reloaded(loop_in_loop_sdfg())
    _, edge, inner, entry = scope_owners(loaded)
    repo = loaded.symbol_repo
    assert repo.lookup('i', at=entry) == SymbolInfo(dace.int32, frozenset())
    assert repo.lookup('N', at=entry) == SymbolInfo(dace.int64, POSITIVE)
    i_below_n = relation(symbolic.RelationKind.LE, 'i', 'N - 1')
    # The outer ``i < N`` holds on the edge, but not in the inner loop, which reopens ``i``
    assert symbolic.ask(i_below_n, repo.facts(edge)) is symbolic.Truth.TRUE
    assert symbolic.ask(i_below_n, repo.facts(entry)) is symbolic.Truth.UNKNOWN
    assert symbolic.ask(relation(symbolic.RelationKind.LE, 'i', 'k - 1'), repo.facts(entry)) is symbolic.Truth.TRUE
    with pytest.raises(KeyError, match='not declared'):
        repo.lookup('j', at=inner)


def test_deepcopy_keys_scopes_by_the_copied_owners():
    sdfg = loop_in_loop_sdfg()
    copied = copy.deepcopy(sdfg)
    assert list(copied.symbol_repo.scopes) == list(scope_owners(copied))
    assert copied.symbol_repo.scopes[scope_owners(copied)[2]].parent is scope_owners(copied)[1]
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
    outer, _, _, entry = scope_owners(sdfg)
    sdfg.symbol_repo.add('k2', dace.int64, at=entry)
    with pytest.raises(InvalidSDFGError, match='does not bind'):
        sdfg.validate()
    sdfg.symbol_repo.remove('k2', at=entry)
    outer.remove_edge(outer.edges()[0])
    with pytest.raises(InvalidSDFGError, match='no owner in the SDFG'):
        sdfg.validate()
    # Saving leaves out the scopes of removed owners and the scopes nested in them
    loaded = reloaded(sdfg)
    assert list(loaded.symbol_repo.scopes) == [loaded.start_block]


if __name__ == '__main__':
    test_params_and_facts_round_trip()
    test_scopes_round_trip_to_their_owners()
    test_inner_scope_shadows_outer_after_round_trip()
    test_deepcopy_keys_scopes_by_the_copied_owners()
    test_nested_sdfg_round_trips_its_own_repo()
    test_validation_rejects_scopes_that_do_not_match_their_owners()
