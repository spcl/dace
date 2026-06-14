# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Regression tests for the ``dace.symbolic.simplify`` ``lru_cache`` type-conflation
bug.

``simplify`` is wrapped in :func:`functools.lru_cache`, which keys entries by
``hash``/``==``. Python conflates booleans with integers
(``hash(True) == hash(1)`` and ``True == sympy.Integer(1)``), so an *untyped*
cache stores ``simplify(True)`` and ``simplify(sympy.Integer(1))`` under the
same entry. Once ``simplify(True)`` (returning ``sympy.true``) is cached, a
later ``simplify(sympy.Integer(1))`` returns that cached ``BooleanTrue`` instead
of the integer ``1`` -- and vice versa. The same holds for ``False``/``0``.

This poisoning is process-global and order dependent: an unrelated caller that
fed a plain ``bool`` to ``simplify`` earlier (e.g. comparing a concrete shape,
``simplify(s == 1)``) would later crash memlet-volume propagation with
``TypeError: Property volume must be a literal or symbolic expression`` when the
propagated volume happened to be ``sympy.Integer(1)``.

The fix is to construct the cache with ``typed=True`` so arguments of different
types (``bool`` vs ``sympy.Integer``) are cached under distinct entries.

These tests are written to be **branch-agnostic**: they assert SymPy-level
semantics only (``simplify`` returns a SymPy object), so they pass on the fixed
code and fail/raise on the unfixed code, independent of any caller-side
workaround. Every test clears the cache first to stay order independent.
"""

import sympy
from sympy.logic.boolalg import Boolean

import dace


def _fresh_cache() -> None:
    """Clear the ``simplify`` cache so each test is order independent."""
    dace.symbolic.simplify.cache_clear()


def test_simplify_bool_does_not_poison_integer_one():
    """``simplify(True)`` first must not turn ``simplify(Integer(1))`` boolean."""
    _fresh_cache()
    # Caches sympy.true; on an untyped cache this poisons the key ``1``.
    dace.symbolic.simplify(True)

    result = dace.symbolic.simplify(sympy.Integer(1))
    assert isinstance(result, sympy.Integer), \
        f"expected sympy.Integer, got {type(result).__name__}: {result!r}"
    assert not isinstance(result, Boolean)
    assert result == 1


def test_simplify_integer_one_does_not_poison_bool():
    """The reverse order: a cached ``Integer(1)`` must not leak into ``simplify(True)``."""
    _fresh_cache()
    dace.symbolic.simplify(sympy.Integer(1))

    result = dace.symbolic.simplify(True)
    assert isinstance(result, Boolean), \
        f"expected a sympy boolean, got {type(result).__name__}: {result!r}"
    assert bool(result) is True


def test_simplify_bool_does_not_poison_integer_zero():
    """Same conflation exists for ``False``/``0`` -- integer direction stays typed."""
    _fresh_cache()
    dace.symbolic.simplify(False)

    result = dace.symbolic.simplify(sympy.Integer(0))
    assert isinstance(result, sympy.Integer), \
        f"expected sympy.Integer, got {type(result).__name__}: {result!r}"
    assert not isinstance(result, Boolean)
    assert result == 0


def test_simplify_integer_zero_does_not_poison_bool():
    """``False``/``0`` -- boolean direction stays boolean."""
    _fresh_cache()
    dace.symbolic.simplify(sympy.Integer(0))

    result = dace.symbolic.simplify(False)
    assert isinstance(result, Boolean), \
        f"expected a sympy boolean, got {type(result).__name__}: {result!r}"
    assert bool(result) is False


def test_simplify_symbolic_expressions_still_cache():
    """Normal symbolic inputs are unaffected and are still served from the cache."""
    _fresh_cache()
    n = sympy.Symbol('N')
    assert dace.symbolic.simplify(n + 1 - n) == 1
    assert dace.symbolic.simplify((n**2 - 1) / (n - 1)) == n + 1
    # Same expression again: served from the cache with the same result.
    assert dace.symbolic.simplify(n + 1 - n) == 1
    assert dace.symbolic.simplify.cache_info().hits >= 1


def test_volume_propagation_after_bool_simplify():
    """End-to-end: a ``bool`` fed to ``simplify`` must not break memlet-volume
    propagation on an unrelated SDFG (the original order-dependent crash)."""
    from dace.sdfg.propagation import propagate_memlets_sdfg

    _fresh_cache()
    # The poisoning call pattern: a concrete shape comparison yields a Python
    # ``bool``, which simplifies to ``sympy.true``.
    poison = dace.symbolic.simplify(sympy.Integer(1) == 1)
    assert bool(poison) is True

    sdfg = dace.SDFG('simplify_cache_volume_repro')
    sdfg.add_array('A', [4], dace.float64)
    sdfg.add_array('B', [4], dace.float64)
    state = sdfg.add_state()
    # A single-iteration map: the propagated volume is ``sympy.Integer(1)`` --
    # exactly the cache key that a cached ``simplify(True)`` poisons.
    state.add_mapped_tasklet('copy',
                             dict(i='0:1'),
                             dict(inp=dace.Memlet('A[i]')),
                             'out = inp',
                             dict(out=dace.Memlet('B[i]')),
                             external_edges=True)
    # Without the fix this raised: TypeError: Property volume must be a literal
    # or symbolic expression (the propagated volume was a BooleanTrue).
    propagate_memlets_sdfg(sdfg)
    sdfg.validate()


if __name__ == '__main__':
    test_simplify_bool_does_not_poison_integer_one()
    test_simplify_integer_one_does_not_poison_bool()
    test_simplify_bool_does_not_poison_integer_zero()
    test_simplify_integer_zero_does_not_poison_bool()
    test_simplify_symbolic_expressions_still_cache()
    test_volume_propagation_after_bool_simplify()
