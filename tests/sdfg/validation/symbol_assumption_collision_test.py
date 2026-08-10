# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""One symbol name spelled with two SymPy assumption sets is two DISTINCT objects.

Index arithmetic over them does not cancel and dependence predicates silently answer wrong, so
:func:`dace.sdfg.validation.check_symbol_assumption_collisions` reports it as a structural defect.
"""
import pytest

import dace
from dace import subsets
from dace.config import Config
from dace.sdfg.validation import (InvalidSDFGError, check_symbol_assumption_collisions, symbol_assumption_collisions,
                                  symbol_assumption_spellings)

FLAG = 'experimental.check_symbol_assumption_collisions'


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


def test_one_spelling_passes():
    """A single instance used by both subsets is not a collision."""
    plain = dace.symbolic.symbol('i', dace.int32)
    sdfg = build(plain, plain)

    spellings = symbol_assumption_spellings(sdfg)
    assert len(spellings['i']) == 1, spellings['i']
    assert symbol_assumption_collisions(sdfg) == {}
    check_symbol_assumption_collisions(sdfg)  # must not raise


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


def test_name_filter():
    """A name-scoped check ignores collisions on other names."""
    plain = dace.symbolic.symbol('i', dace.int32)
    nonneg = dace.symbolic.symbol('i', dace.int32, nonnegative=True)
    sdfg = build(plain, nonneg)

    assert symbol_assumption_collisions(sdfg, 'j') == {}
    check_symbol_assumption_collisions(sdfg, 'j')  # must not raise
    with pytest.raises(InvalidSDFGError):
        check_symbol_assumption_collisions(sdfg, 'i')


def test_validate_honors_flag():
    """``validate`` only runs the check when the experimental flag is on."""
    plain = dace.symbolic.symbol('i', dace.int32)
    nonneg = dace.symbolic.symbol('i', dace.int32, nonnegative=True)
    sdfg = build(plain, nonneg)

    with dace.config.set_temporary('experimental', 'check_symbol_assumption_collisions', value=False):
        sdfg.validate()  # off by default: the collision does not fail validation

    with dace.config.set_temporary('experimental', 'check_symbol_assumption_collisions', value=True):
        with pytest.raises(InvalidSDFGError) as exc:
            sdfg.validate()
    assert 'is spelled 2 ways' in str(exc.value)


def test_add_symbol_tripwire():
    """Registering a symbol trips on a name the SDFG already spells two ways."""
    plain = dace.symbolic.symbol('i', dace.int32)
    nonneg = dace.symbolic.symbol('i', dace.int32, nonnegative=True)
    sdfg = build(plain, nonneg)

    with dace.config.set_temporary('experimental', 'check_symbol_assumption_collisions', value=False):
        sdfg.add_symbol('i', dace.int32)  # off by default: registration succeeds
    del sdfg.symbols['i']

    with dace.config.set_temporary('experimental', 'check_symbol_assumption_collisions', value=True):
        with pytest.raises(InvalidSDFGError) as exc:
            sdfg.add_symbol('i', dace.int32)
        assert 'i' not in sdfg.symbols, 'the tripwire must fire BEFORE the name is registered'
        # A name the SDFG spells only one way still registers fine.
        sdfg.add_symbol('k', dace.int32)
    assert 'k' in sdfg.symbols
    assert 'is spelled 2 ways' in str(exc.value)


def test_flag_defaults_off():
    """The check costs nothing unless asked for."""
    assert Config.get_bool(*FLAG.split('.')) is False


if __name__ == '__main__':
    test_one_spelling_passes()
    test_two_spellings_raise()
    test_name_filter()
    test_validate_honors_flag()
    test_add_symbol_tripwire()
    test_flag_defaults_off()
