# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""One symbol name spelled with two SymPy assumption sets is two DISTINCT objects.

Index arithmetic over them does not cancel and dependence predicates silently answer wrong, so
:func:`dace.sdfg.validation.check_symbol_assumption_collisions` rejects it during validation.
"""
import pytest
import sympy

import dace
from dace import subsets
from dace.config import Config
from dace.sdfg.validation import (InvalidSDFGError, check_symbol_assumption_collisions, scope_parameter_shadowings,
                                  symbol_assumption_collisions, symbol_assumption_spellings)

FLAG = ('experimental', 'check_symbol_assumption_collisions')


def build(read_sym, write_sym) -> dace.SDFG:
    """``B[i] = A[i]`` over a map, with the read and write subsets indexed by the given spellings.

    Whatever spelling is handed in, the stored one is bare: ``Range`` bares every bound it parses.
    """
    sdfg = dace.SDFG('assumption_collision')
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_array('B', [20], dace.float64)
    state = sdfg.add_state('main', is_start_block=True)
    me, mx = state.add_map('m', {'i': '0:20'})
    rd = state.add_access('A')
    wr = state.add_access('B')
    tasklet = state.add_tasklet('copy', {'a'}, {'b'}, 'b = a')
    state.add_memlet_path(rd,
                          me,
                          tasklet,
                          dst_conn='a',
                          memlet=dace.Memlet(data='A', subset=subsets.Range([(read_sym, read_sym, 1)])))
    state.add_memlet_path(tasklet,
                          mx,
                          wr,
                          src_conn='b',
                          memlet=dace.Memlet(data='B', subset=subsets.Range([(write_sym, write_sym, 1)])))
    return sdfg


def stored_symbols(subset: subsets.Range):
    """Every symbol object a subset actually stores; ``Subset.free_symbols`` yields NAMES."""
    stored = []
    for bound in (*(b for r in subset.ranges for b in r), *subset.tile_sizes):
        parts = (bound.expr, bound.approx) if isinstance(bound, dace.symbolic.SymExpr) else (bound, )
        for part in parts:
            stored.extend(part.free_symbols)
    return stored


def resized(assumed) -> dace.SDFG:
    """``B[i] = A[i]`` over ``0:N``, then ``B`` resized with the ASSUMED ``N``, as a pass resizes it.

    Subsets, map ranges and descriptors added through the public API are all spelled bare, so a
    collision reaches a stored object only where the coercing seam does not sit: ``desc.shape``,
    assigned directly by every transformation that reshapes a container (``transformation/helpers``,
    ``map_fission``, ``streaming_memory``, ...), and after ``add_datadesc`` already moved that
    container's assumptions into the registry. That is the pblas defect exactly -- assumed grid
    symbols in descriptor shapes against bare ones in the maps and memlets that index them -- and
    catching it is the whole job of the checker.
    """
    bare = dace.symbolic.symbol(assumed.name, assumed.dtype)
    sdfg = dace.SDFG('assumption_collision')
    sdfg.add_array('A', [bare], dace.float64)
    sdfg.add_array('B', [bare], dace.float64)
    state = sdfg.add_state('main', is_start_block=True)
    me, mx = state.add_map('m', {'i': f'0:{bare}'})
    rd = state.add_access('A')
    wr = state.add_access('B')
    tasklet = state.add_tasklet('copy', {'a'}, {'b'}, 'b = a')
    state.add_memlet_path(rd, me, tasklet, dst_conn='a', memlet=dace.Memlet(data='A', subset='i'))
    state.add_memlet_path(tasklet, mx, wr, src_conn='b', memlet=dace.Memlet(data='B', subset='i'))
    assert symbol_assumption_collisions(sdfg) == {}, 'precondition: construction spells every bound bare'

    sdfg.arrays['B'].shape = (assumed, )
    return sdfg


def test_explicit_none_assumption_splits_the_symbol():
    """The defect the frontend fix prevents: ``None`` is an assumption, not the absence of one."""
    omitted = dace.symbolic.symbol('i', dace.int32)
    explicit_none = dace.symbolic.symbol('i', dace.int32, nonnegative=None)

    assert omitted != explicit_none, 'precondition: SymPy folds the explicit None into identity'
    assert (omitted - explicit_none) != 0, 'two spellings of one name do not cancel'
    assert omitted not in (explicit_none + 1).free_symbols

    # What every reparse of a loop bound mints is the omitted spelling, so that is the canonical one.
    assert dace.symbolic.pystr_to_symbolic('i') == omitted


def test_frontend_mints_the_canonical_spelling():
    """A parsed program is free of collisions, including a loop with established bounds."""

    @dace.program
    def constant_bounds(A: dace.float64[128]):
        for i in range(1, 127):
            A[i] = A[i - 1] + A[i + 1]

    @dace.program
    def symbolic_bounds(A: dace.float64[128], N: dace.int32):
        for i in range(N):
            A[i] = A[i] + 1.0

    for program in (constant_bounds, symbolic_bounds):
        sdfg = program.to_sdfg(simplify=False, validate=True)
        assert symbol_assumption_collisions(sdfg) == {}, program.name
        sdfg.validate()


def test_declared_assumptions_go_to_the_registry():
    """A public-API assumption symbol is recorded, and only the BARE symbol is stored."""
    M = dace.symbol('M', positive=True)

    @dace.program
    def scal(A: dace.float64[M]):
        return A * 2.0

    sdfg = scal.to_sdfg(simplify=False, validate=True)

    assert symbol_assumption_collisions(sdfg) == {}
    assert sdfg.symbol_assumptions['M']['positive'] is True
    # Stored spellings are bare: nothing in the graph knows M is positive.
    for spelling in symbol_assumption_spellings(sdfg)['M']:
        assert ('positive', True) not in spelling, spelling
    # The facts are still reachable, transiently, for reasoning.
    assert sdfg.assume_symbols(dace.symbolic.pystr_to_symbolic('M')).is_positive is True

    j = sdfg.to_json()
    assert dict(dace.SDFG.from_json(j).symbol_assumptions) == dict(sdfg.symbol_assumptions)
    del j['attributes']['symbol_assumptions']
    assert dict(dace.SDFG.from_json(j).symbol_assumptions) == {}  # legacy file


def test_map_range_stores_the_bare_spelling():
    """A pass minting an assumed symbol into a map range must not split the name.

    ``add_datadesc`` absorbs the assumptions off a fresh transient, so a range that kept them
    would be the second spelling -- which is how the distribution passes broke the samples.
    """
    P = dace.symbolic.symbol('P', dace.int32, positive=True)
    assert P != dace.symbolic.symbol('P', dace.int32), 'precondition: the two spellings differ'

    sdfg = dace.SDFG('map_range_intake')
    sdfg.add_array('A', [20], dace.float64)
    state = sdfg.add_state('main', is_start_block=True)
    entry, _ = state.add_map('m', {'i': '0:20'})
    entry.map.range = subsets.Range([(0, 20 // P - 1, 1)])
    sdfg.add_transient('t', [20 // P], dace.float64)

    stored = symbol_assumption_spellings(sdfg)['P']
    assert len(stored) == 1, stored
    assert all(('positive', True) not in key for key in stored), stored
    assert symbol_assumption_collisions(sdfg) == {}
    check_symbol_assumption_collisions(sdfg)  # must not raise


def test_update_symbol_assumptions_is_strict():
    """The registry is the only mutation path, and it only refines."""
    sdfg = dace.SDFG('registry')
    sdfg.add_symbol('n', dace.int32)

    with pytest.raises(KeyError):
        sdfg.update_symbol_assumptions('undeclared', positive=True)

    sdfg.update_symbol_assumptions('n', positive=True)
    sdfg.update_symbol_assumptions('n', positive=True)  # idempotent
    sdfg.update_symbol_assumptions('n', integer=True)  # refinement
    with pytest.raises(ValueError):
        sdfg.update_symbol_assumptions('n', positive=False)
    assert sdfg.symbol_assumptions['n'] == {'positive': True, 'integer': True}

    # add_symbol stays strict and assumption-free.
    with pytest.raises(FileExistsError):
        sdfg.add_symbol('n', dace.int32)


def test_one_spelling_passes():
    """A single instance used by both subsets is not a collision."""
    plain = dace.symbolic.symbol('i', dace.int32)
    sdfg = build(plain, plain)

    spellings = symbol_assumption_spellings(sdfg)
    assert len(spellings['i']) == 1, spellings['i']
    assert symbol_assumption_collisions(sdfg) == {}
    check_symbol_assumption_collisions(sdfg)  # must not raise
    sdfg.validate()  # must not raise


def test_memlet_subset_stores_the_bare_spelling():
    """A memlet subset built from ``dace.symbol('X', positive=True)`` stores the BARE ``X``.

    The fact is not lost, it moves: ``add_datadesc`` files it in ``SDFG.symbol_assumptions`` and
    the graph keeps the name alone. Start, end, step, tile, ``Indices`` elements and
    ``other_subset`` are all written through the same coercing seam.
    """
    X = dace.symbol('X', dace.int32, positive=True)
    bare = dace.symbolic.symbol('X', dace.int32)
    assert X != bare, 'precondition: SymPy folds assumptions into symbol identity'

    sdfg = dace.SDFG('memlet_intake')
    sdfg.add_array('A', [X], dace.float64)
    sdfg.add_array('B', [X], dace.float64)
    state = sdfg.add_state('main', is_start_block=True)
    copy = dace.Memlet(data='A', subset=subsets.Range([(0, X - 1, 1)]), other_subset=subsets.Range([(0, X - 1, 1)]))
    state.add_nedge(state.add_access('A'), state.add_access('B'), copy)

    # The assumption reached the registry, and the graph spells the name one way: bare.
    assert sdfg.symbol_assumptions['X']['positive'] is True
    assert list(symbol_assumption_spellings(sdfg)['X']) == [tuple(sorted(bare.assumptions0.items()))]
    for sub in (copy.subset, copy.other_subset):
        assert stored_symbols(sub), sub
        assert all(sym == bare for sym in stored_symbols(sub)), sub

    # Tiles, indices and the assigning path are coerced the same way as the constructor.
    tiled = subsets.Range([(0, X - 1, 1, X)])
    tiled[0] = (X, X, 1)
    for sub in (tiled, subsets.Range.from_indices([X])):
        assert all(sym == bare for sym in stored_symbols(sub)), sub

    assert symbol_assumption_collisions(sdfg) == {}
    check_symbol_assumption_collisions(sdfg)  # must not raise
    sdfg.validate()  # must not raise


def test_registry_restores_simplification_strength():
    """What baring costs SymPy, ``assume_symbols`` hands back, spelled exactly as before.

    A bare symbol answers ``is_positive`` with ``None``, so ``Max``, ``sqrt(N**2)`` and inequality
    folding all take their conservative branch. Reasoning sites must therefore ask the registry
    first, and what it gives back has to be identical to the declared symbol -- otherwise storing
    bare would be a silent loss of simplification strength.
    """
    N = dace.symbol('N', dace.int32, positive=True)

    sdfg = dace.SDFG('strength')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    state = sdfg.add_state('main', is_start_block=True)
    copy = dace.Memlet(data='A', subset=subsets.Range([(0, N - 1, 1)]), other_subset=subsets.Range([(0, N - 1, 1)]))
    state.add_nedge(state.add_access('A'), state.add_access('B'), copy)

    stored = sdfg.arrays['A'].shape[0]
    assert stored == dace.symbolic.symbol('N', dace.int32), 'precondition: the shape is stored bare'
    # Baring drops only what was DECLARED; the dtype facts are part of the bare spelling.
    assert stored.is_integer is True
    assert stored.is_positive is None

    reasoned = sdfg.assume_symbols(stored)
    assert reasoned.assumptions0 == N.assumptions0
    assert reasoned.is_positive is True
    assert reasoned is sdfg.assume_symbols(stored), 'the assumed spelling is minted once and cached'

    # Every fold that needs positivity answers as it would have before the symbol was stored.
    for ask in (lambda x: sympy.sqrt(x**2), lambda x: sympy.Max(x, 0), lambda x: sympy.simplify(x > 0)):
        assert ask(stored) != ask(N), 'precondition: the bare spelling really does lose the fold'
        assert ask(sdfg.assume_symbols(stored)) == ask(N)

    # The same holds for an expression read back out of the graph, not only for a lone symbol.
    extent = copy.subset.num_elements()
    assert sympy.Max(extent, 0) != extent, 'precondition: bare storage cannot fold the extent'
    assert sympy.Max(sdfg.assume_symbols(extent), 0) == sdfg.assume_symbols(extent)
    assert sdfg.assume_symbols(sympy.sqrt(extent**2)) == reasoned

    # Reasoning never writes back: the graph is still bare, and still collision-free.
    assert sdfg.arrays['A'].shape[0] is stored
    assert symbol_assumption_collisions(sdfg) == {}
    sdfg.validate()  # must not raise


def test_two_spellings_raise():
    """A bare ``N`` in the map and the descriptors against a ``nonnegative=True`` ``N`` in a shape."""
    nonneg = dace.symbolic.symbol('N', dace.int32, nonnegative=True)
    assert nonneg != dace.symbolic.symbol('N', dace.int32), 'precondition: the two spellings differ'
    sdfg = resized(nonneg)

    collisions = symbol_assumption_collisions(sdfg)
    assert list(collisions) == ['N'], collisions
    assert len(collisions['N']) == 2, collisions['N']

    # Each spelling is attributed to what holds it: the rewritten shape, and the map and descriptor
    # that were left alone.
    assumed = [key for key in collisions['N'] if ('nonnegative', True) in key]
    assert len(assumed) == 1, collisions['N']
    assert collisions['N'][assumed[0]] == ['array "B" shape/strides/offset'], collisions['N']
    plain = next(key for key in collisions['N'] if key != assumed[0])
    assert 'state "main" map "m" range' in collisions['N'][plain], collisions['N']
    assert 'array "A" shape/strides/offset' in collisions['N'][plain], collisions['N']

    with pytest.raises(InvalidSDFGError) as exc:
        check_symbol_assumption_collisions(sdfg)
    message = str(exc.value)
    assert 'is spelled 2 ways' in message, message
    assert 'nonnegative=True' in message, message
    # Shared facts are filtered out of the report -- only what SPLITS the symbol is shown.
    assert 'integer=True' not in message, message


def test_validate_honors_flag():
    """``validate`` runs the check only when the experimental flag is on."""
    sdfg = resized(dace.symbolic.symbol('N', dace.int32, nonnegative=True))

    with dace.config.set_temporary(*FLAG, value=False):
        sdfg.validate()  # off by default: the collision does not fail validation

    with dace.config.set_temporary(*FLAG, value=True):
        with pytest.raises(InvalidSDFGError) as exc:
            sdfg.validate()
    assert 'is spelled 2 ways' in str(exc.value)


def test_flag_defaults_off():
    """The check costs nothing unless asked for."""
    assert Config.get_default(*FLAG) is False


def test_name_filter():
    """A name-scoped check ignores collisions on other names."""
    sdfg = resized(dace.symbolic.symbol('N', dace.int32, nonnegative=True))

    assert symbol_assumption_collisions(sdfg, 'j') == {}
    check_symbol_assumption_collisions(sdfg, 'j')  # must not raise
    with pytest.raises(InvalidSDFGError):
        check_symbol_assumption_collisions(sdfg, 'N')


def test_disjoint_scopes_may_reuse_a_parameter_name():
    """A scope parameter is scope-local: ``i`` in two disjoint maps is the norm, not a collision."""
    sdfg = dace.SDFG('two_scopes')
    sdfg.add_array('A', [20], dace.float64)
    previous = None
    for label in ('first', 'second'):
        state = sdfg.add_state(label, is_start_block=previous is None) if previous is None else \
            sdfg.add_state_after(previous, label)
        me, mx = state.add_map(f'm_{label}', {'i': '0:20'})
        tasklet = state.add_tasklet('one', {}, {'a'}, 'a = 1.0')
        state.add_nedge(me, tasklet, dace.Memlet())
        state.add_memlet_path(tasklet,
                              mx,
                              state.add_access('A'),
                              src_conn='a',
                              memlet=dace.Memlet(data='A', subset='i'))
        previous = state

    spellings = symbol_assumption_spellings(sdfg)['i']
    assert len(spellings) == 1, spellings
    assert len(next(iter(spellings.values()))) > 1, spellings  # both scopes, one spelling

    assert scope_parameter_shadowings(sdfg) == {}
    assert symbol_assumption_collisions(sdfg) == {}
    check_symbol_assumption_collisions(sdfg)  # must not raise
    sdfg.validate()  # must not raise


def test_scope_parameter_spelled_two_ways_is_reported():
    """Inside a scope too, a parameter name has exactly ONE stored spelling.

    Passes that reshape a transient from the map extent write ``desc.shape`` straight (that is
    ``map_fission``); computing that extent with an assumed parameter leaves a second spelling of
    the very symbol index arithmetic has to cancel.
    """
    plain = dace.symbolic.symbol('i', dace.int32)
    assumed = dace.symbolic.symbol('i', dace.int32, positive=True)

    sdfg = build(plain, plain)
    sdfg.add_transient('t', [20], dace.float64)
    assert symbol_assumption_collisions(sdfg) == {}, 'precondition: one spelling per name'

    sdfg.arrays['t'].shape = (assumed + 1, )
    collisions = symbol_assumption_collisions(sdfg)
    assert list(collisions) == ['i'], collisions
    assert len(collisions['i']) == 2, collisions['i']
    with pytest.raises(InvalidSDFGError) as exc:
        check_symbol_assumption_collisions(sdfg)
    assert 'is spelled 2 ways' in str(exc.value)

    # The parameter's facts stay in its scope: nothing hoisted them into the registry.
    assert 'i' not in sdfg.symbol_assumptions


def test_scope_parameter_shadowing_the_registry_is_reported():
    """A parameter named like a registry symbol would be assumed the registry's facts, wrongly."""
    N = dace.symbol('N', dace.int32, positive=True)

    sdfg = dace.SDFG('shadow')
    sdfg.add_array('A', [N], dace.float64)
    assert sdfg.symbol_assumptions['N']['positive'] is True, 'precondition: the registry knows N'

    state = sdfg.add_state('main', is_start_block=True)
    me, mx = state.add_map('m', {'N': '0:20'})  # a pass naming its parameter after the symbol
    tasklet = state.add_tasklet('one', {}, {'a'}, 'a = 1.0')
    state.add_nedge(me, tasklet, dace.Memlet())
    state.add_memlet_path(tasklet, mx, state.add_access('A'), src_conn='a', memlet=dace.Memlet(data='A', subset='N'))

    # Not a spelling collision -- every stored `N` is bare -- but still wrong to reason about.
    assert symbol_assumption_collisions(sdfg) == {}
    assert list(scope_parameter_shadowings(sdfg)) == ['N']
    assert scope_parameter_shadowings(sdfg) == {'N': ['state "main" map "m" parameter']}

    with pytest.raises(InvalidSDFGError) as exc:
        check_symbol_assumption_collisions(sdfg)
    assert 'shadows' in str(exc.value), exc.value
    with pytest.raises(InvalidSDFGError):
        sdfg.validate()

    # Name-scoped, like the spelling rule: another name is clean.
    assert scope_parameter_shadowings(sdfg, 'i') == {}
    check_symbol_assumption_collisions(sdfg, 'i')  # must not raise


def test_scope_parameter_assumptions_do_not_reach_the_registry():
    """A pass minting an assumed parameter into a map range: bared, and NOT filed in the registry.

    What is known about a parameter is its range, and that belongs to the scope. Hoisting the fact
    to the SDFG would assume it of every other use of the name, in every other scope.
    """
    assumed = dace.symbolic.symbol('i', dace.int32, positive=True)

    sdfg = dace.SDFG('scope_local')
    sdfg.add_array('A', [20], dace.float64)
    state = sdfg.add_state('main', is_start_block=True)
    entry, _ = state.add_map('m', {'i': '0:20'})
    entry.map.range = subsets.Range([(assumed, assumed + 10, 1)])

    assert 'i' not in sdfg.symbol_assumptions
    assert 'i' not in sdfg.symbols
    stored = symbol_assumption_spellings(sdfg)['i']
    assert len(stored) == 1, stored
    assert all(('positive', True) not in key for key in stored), stored
    assert scope_parameter_shadowings(sdfg) == {}
    check_symbol_assumption_collisions(sdfg)  # must not raise


if __name__ == '__main__':
    test_explicit_none_assumption_splits_the_symbol()
    test_frontend_mints_the_canonical_spelling()
    test_declared_assumptions_go_to_the_registry()
    test_map_range_stores_the_bare_spelling()
    test_update_symbol_assumptions_is_strict()
    test_one_spelling_passes()
    test_memlet_subset_stores_the_bare_spelling()
    test_registry_restores_simplification_strength()
    test_two_spellings_raise()
    test_validate_rejects_a_collision()
    test_name_filter()
    test_disjoint_scopes_may_reuse_a_parameter_name()
    test_scope_parameter_spelled_two_ways_is_reported()
    test_scope_parameter_shadowing_the_registry_is_reported()
    test_scope_parameter_assumptions_do_not_reach_the_registry()
