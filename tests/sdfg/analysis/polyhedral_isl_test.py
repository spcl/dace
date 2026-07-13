# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for :mod:`dace.sdfg.analysis.polyhedral_isl`, the reusable exact-integer
ISL polyhedral layer. No SDFG is built -- that is the point of the module: its inputs
are :mod:`dace.symbolic` expressions plus variable-name strings, so the polyhedral core
is exercised directly."""
import pytest

pytest.importorskip('islpy')  # HAVE_ISL gate: skip the whole module without islpy.

import islpy as isl

from dace import symbolic
from dace.sdfg.analysis import polyhedral_isl as poly

p = symbolic.pystr_to_symbolic

# --- to_isl / render_affine ---------------------------------------------------------


def test_to_isl_affine_renders_and_isl_parses():
    """An affine ``2*i - j + 3`` renders under a name map and the rendered string
    parses inside a real ``isl.Set`` (via :func:`make_set`)."""
    e = p('2*i - j + 3')
    rendered = poly.render_affine(e, poly.build_name_map(['i', 'j'], []))
    assert isinstance(rendered, str) and rendered
    # make_set renders every constraint through render_affine/to_isl and hands the
    # text to ISL's parser; a successful build proves the string is valid ISL.
    s, _ = poly.make_set(['i', 'j'], [], [e])
    assert isinstance(s, isl.Set)


def test_to_isl_int_floor_maps_to_floor_and_parses():
    """``int_floor(N, 8)`` renders to a native ISL ``floor(.../8)`` and parses."""
    rendered = poly.render_affine(p('int_floor(N, 8)'), poly.build_name_map([], ['N']))
    assert 'floor(' in rendered
    s, _ = poly.make_set(['i'], ['N'], [p('i - int_floor(N, 8)')])
    assert isinstance(s, isl.Set)


def test_to_isl_int_ceil_maps_to_ceil_and_parses():
    """``int_ceil(N, 8)`` renders to a native ISL ``ceil(.../8)`` and parses."""
    rendered = poly.render_affine(p('int_ceil(N, 8)'), poly.build_name_map([], ['N']))
    assert 'ceil(' in rendered
    s, _ = poly.make_set(['i'], ['N'], [p('int_ceil(N, 8) - i')])
    assert isinstance(s, isl.Set)


def test_to_isl_mod_maps_to_mod_and_parses():
    """``N % 8`` renders to a native ISL ``mod``."""
    rendered = poly.render_affine(p('N % 8'), poly.build_name_map([], ['N']))
    assert 'mod' in rendered
    s, _ = poly.make_set(['i'], ['N'], [p('(N % 8) - i')])
    assert isinstance(s, isl.Set)


def test_to_isl_nonlinear_raises():
    """A nonlinear ``i*j`` term is refused with ``ValueError``."""
    with pytest.raises(ValueError):
        poly.to_isl(p('i*j'))


def test_to_isl_non_integer_coefficient_raises():
    """A non-integer coefficient (``i/2``) is refused with ``ValueError``."""
    with pytest.raises(ValueError):
        poly.to_isl(p('i/2'))


def test_to_isl_symbolic_divisor_raises():
    """A symbolic div/mod divisor (``int_floor(N, M)``) is refused with ``ValueError``."""
    with pytest.raises(ValueError):
        poly.to_isl(p('int_floor(N, M)'))


# --- is_domain_empty ----------------------------------------------------------------


def test_is_domain_empty_unsatisfiable_is_empty():
    """``i >= 0 and -i - 1 >= 0`` (i.e. ``i <= -1``) is unsatisfiable -> empty."""
    assert poly.is_domain_empty(['i'], [], [p('i'), p('-i - 1')]) is True


def test_is_domain_empty_satisfiable_is_not_empty():
    """``0 <= i <= 10`` is satisfiable -> not empty."""
    assert poly.is_domain_empty(['i'], [], [p('i'), p('10 - i')]) is False


def test_is_domain_empty_s114_constraint_list_returns_bool_without_raising():
    """Regression lock for the s114 fix: a quasi-affine domain carrying
    ``int_floor`` and a strided constraint returns a plain ``bool`` and does not
    raise on the render."""
    dims = ['_loop_it_0', '_loop_it_1']
    params = ['LEN_2D']
    cons = [
        p('_loop_it_0'),
        p('-_loop_it_0 + int_floor(LEN_2D, 8) - 1'),
        p('_loop_it_1'),
        p('8*_loop_it_0 - _loop_it_1 - 1'),
        p('-_loop_it_0 + _loop_it_1'),
    ]
    result = poly.is_domain_empty(dims, params, cons)
    assert isinstance(result, bool)


# --- classify_dim -------------------------------------------------------------------


def test_classify_dim_positive_unit_coefficient_is_lower_bound():
    """coeff ``+1`` (``i - 3 >= 0`` => ``i >= 3``) yields a lower-bound term."""
    lo, hi, ok = poly.classify_dim(p('i - 3'), p('i'))
    assert ok is True
    assert lo and not hi


def test_classify_dim_negative_unit_coefficient_is_upper_bound():
    """coeff ``-1`` (``5 - i >= 0`` => ``i <= 5``) yields an upper-bound term."""
    lo, hi, ok = poly.classify_dim(p('5 - i'), p('i'))
    assert ok is True
    assert hi and not lo


def test_classify_dim_non_unit_coefficient_is_unsupported():
    """``|coeff| > 1`` (an integer-division bound) is unsupported -> ``ok=False``."""
    lo, hi, ok = poly.classify_dim(p('2*i - 3'), p('i'))
    assert ok is False
    assert not lo and not hi


# --- constraint_to_sympy ------------------------------------------------------------


def test_constraint_to_sympy_reconstructs_symbolic_expressions():
    """Building a set via :func:`make_set`, pulling a basic set's constraints and
    reconstructing them yields symbolic expressions referencing the original names."""
    s, nmap = poly.make_set(['i'], ['N'], [p('i'), p('N - i')])
    inv = {safe: orig for orig, safe in nmap.items()}
    safe_dims = [nmap['i']]
    safe_params = [nmap['N']]
    isym, nsym = p('i'), p('N')
    reconstructed = []
    for b_set in poly.collect_basic_sets(s):
        for c in b_set.get_constraints():
            e = poly.constraint_to_sympy(c, safe_dims, safe_params, inv)
            reconstructed.append(e)
            # Free symbols must map back to the originals we supplied.
            assert e.free_symbols <= {isym, nsym}
    assert reconstructed
    # The two supplied bounds (i >= 0, N - i >= 0) reappear: some term mentions i and
    # some term mentions N.
    all_syms = set().union(*(e.free_symbols for e in reconstructed))
    assert isym in all_syms and nsym in all_syms
