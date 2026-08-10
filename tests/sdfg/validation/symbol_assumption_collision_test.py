# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""One symbol name spelled with two SymPy assumption sets is two DISTINCT objects.

Index arithmetic over them does not cancel and dependence predicates silently answer wrong, so
:func:`dace.sdfg.validation.check_symbol_assumption_collisions` rejects it during validation.
"""
import pytest

import dace
from dace import subsets
from dace.sdfg.validation import (InvalidSDFGError, check_symbol_assumption_collisions, symbol_assumption_collisions,
                                  symbol_assumption_spellings)


def build(read_sym, write_sym) -> dace.SDFG:
    """``B[i] = A[i]`` over a map, with the read and write subsets indexed by the given spellings."""
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


def test_two_spellings_raise():
    """A plain ``i`` against a ``nonnegative=True`` ``i`` is reported, with both locations."""
    plain = dace.symbolic.symbol('i', dace.int32)
    nonneg = dace.symbolic.symbol('i', dace.int32, nonnegative=True)
    assert plain != nonneg, 'precondition: SymPy folds assumptions into symbol identity'
    sdfg = build(plain, nonneg)

    collisions = symbol_assumption_collisions(sdfg)
    assert list(collisions) == ['i'], collisions
    assert len(collisions['i']) == 2, collisions['i']

    # Both spellings are attributed to the memlet they actually sit in.
    located = {loc for where in collisions['i'].values() for loc in where}
    assert any('memlet "A"' in loc for loc in located), located
    assert any('memlet "B"' in loc for loc in located), located

    # Exactly one of the two variants carries the nonnegativity facts.
    with_nonneg = [key for key in collisions['i'] if ('nonnegative', True) in key]
    assert len(with_nonneg) == 1, collisions['i']

    with pytest.raises(InvalidSDFGError) as exc:
        check_symbol_assumption_collisions(sdfg)
    message = str(exc.value)
    assert 'is spelled 2 ways' in message, message
    assert 'nonnegative=True' in message, message
    # Shared facts are filtered out of the report -- only what SPLITS the symbol is shown.
    assert 'integer=True' not in message, message


def test_validate_rejects_a_collision():
    """The check is unconditional: no flag has to be set for validation to reject the SDFG."""
    plain = dace.symbolic.symbol('i', dace.int32)
    nonneg = dace.symbolic.symbol('i', dace.int32, nonnegative=True)
    sdfg = build(plain, nonneg)

    with pytest.raises(InvalidSDFGError) as exc:
        sdfg.validate()
    assert 'is spelled 2 ways' in str(exc.value)


def test_name_filter():
    """A name-scoped check ignores collisions on other names."""
    plain = dace.symbolic.symbol('i', dace.int32)
    nonneg = dace.symbolic.symbol('i', dace.int32, nonnegative=True)
    sdfg = build(plain, nonneg)

    assert symbol_assumption_collisions(sdfg, 'j') == {}
    check_symbol_assumption_collisions(sdfg, 'j')  # must not raise
    with pytest.raises(InvalidSDFGError):
        check_symbol_assumption_collisions(sdfg, 'i')


if __name__ == '__main__':
    test_explicit_none_assumption_splits_the_symbol()
    test_frontend_mints_the_canonical_spelling()
    test_declared_assumptions_go_to_the_registry()
    test_update_symbol_assumptions_is_strict()
    test_one_spelling_passes()
    test_two_spellings_raise()
    test_validate_rejects_a_collision()
    test_name_filter()
