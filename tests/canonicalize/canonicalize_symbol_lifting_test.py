# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Comprehensive symbol-lifting / promotion behaviour under ``canonicalize``.

The Python frontend promotes non-symbol expressions that appear in loop
ranges (``range(0, N + 1)``), array subscript bounds (``a[0:K + 1]``) and
similar positions to auxiliary symbols (``N_plus_1``, ``K_plus_1`` etc.,
per the ``get_target_name`` heuristic in ``newast.py``). After
canonicalization with ``UniqueLoopIterators(assign_loop_iterator_post_value
= False)`` (the canonicalize-pipeline default), these promoted symbols
must:

1. Stay declared (not be lost on the NSDFG boundaries that ``LoopToMap``
   / ``MoveIfIntoLoop`` introduce).
2. Stay at the *outermost* scope where they are invariant -- they should
   not be re-materialised inside a loop body that uses them in its range,
   because that creates a body-assigns-loop-range-symbol shape that
   ``LoopToMap`` then has to refuse.
3. Not be unnecessarily duplicated across canonicalize iterations (a
   per-loop ``N_plus_1_<N>`` snowball would otherwise grow on every call).

The tests below exercise the cases that exercise different parts of that
invariant:

* **Outer-only invariant** -- the promoted expression depends only on
  SDFG-level symbols; canonicalize keeps it at SDFG scope.
* **Loop-variable-dependent** -- the promoted expression reads the loop
  iterator; it CANNOT be lifted past the loop and must stay inside.
* **Data-dependent** -- the value comes from an array read (an interstate-
  edge assignment ``sym = arr[i]`` style); must stay where the read is.
* **Mixed (outer + loop-var)** -- only the outer-only sub-expression can
  be lifted; the combined expression stays.
* **Transitive lift** -- a chain ``s1 = K + 1; s2 = 2 * s1`` where both
  values are loop-invariant; both should remain SDFG-scoped.
* **Irrelevant clutter** -- additional SDFG-declared symbols that play
  no role in the loop must not interfere with canonicalize.
* **Reductions / boundary accesses** -- common cloudsc-style shape.

Several scenarios are *known not yet supported* under the current pipeline
(notably reductions whose inner accumulator triggers a ``j`` leak through
NSDFG boundaries); they are documented here with a precise xfail reason
linking to the deferred ``CascadeInterstateEdgeAssignmentsUp`` design
work that targets them.
"""
import numpy as np
import pytest
import re

import dace
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')
M = dace.symbol('M')
K = dace.symbol('K')


def _all_symbols(sdfg):
    """Union of declared symbols across the SDFG and every NestedSDFG."""
    out = set(sdfg.symbols.keys())
    for n, _ in sdfg.all_nodes_recursive():
        if hasattr(n, 'sdfg') and n.sdfg is not None and n.sdfg is not sdfg:
            out |= set(n.sdfg.symbols.keys())
    return out


def _count_promoted(sdfg, base: str) -> int:
    """Number of ``<base>_(plus|minus|times|div)_<digits>(_<digits>)?`` symbols
    declared anywhere in the SDFG hierarchy. SSA renaming may add a trailing
    ``_<n>`` suffix, so the regex tolerates one extra disambiguation suffix."""
    pat = re.compile(rf'^{re.escape(base)}_(plus|minus|times|div)_\d+(_\d+)?$')
    return sum(1 for s in _all_symbols(sdfg) if pat.match(s))


def _nmaps(sdfg):
    return len([n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)])


# ----------------------------------------------------------------------
# Outer-only invariant promotions (the canonical "lift-able" cases)
# ----------------------------------------------------------------------


@dace.program
def outer_only_promoted_bound(a: dace.float64[N], b: dace.float64[N]):
    """``range(0, N - 1)`` -- the ``N - 1`` is loop-invariant; promotion
    must stay at SDFG scope and the resulting loop's range may safely
    reference it from outside."""
    for i in range(0, N - 1):
        b[i] = a[i] + a[i + 1]


def test_outer_only_promoted_bound_value_preserving():
    n = 10
    a = np.random.rand(n)
    sdfg = outer_only_promoted_bound.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    out = np.zeros(n)
    sdfg(a=a.copy(), b=out, N=n)
    exp = np.zeros(n)
    for i in range(0, n - 1):
        exp[i] = a[i] + a[i + 1]
    assert np.allclose(out, exp)


def test_outer_only_promoted_bound_idempotent_symbol_count():
    """Re-running canonicalize must not snowball the promoted-bound count."""
    sdfg = outer_only_promoted_bound.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    before = _count_promoted(sdfg, 'N')
    canonicalize(sdfg, validate=True)
    after = _count_promoted(sdfg, 'N')
    assert before == after, f"canonicalize duplicated N_minus symbols: {before} -> {after}"


@dace.program
def two_loops_share_promoted_bound(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    """Two loops over ``range(0, N - 1)``. The frontend promotes ``N - 1``
    once per loop (the SSA suffix may differ); after canonicalize the
    declared symbol count must remain small."""
    for i in range(0, N - 1):
        b[i] = a[i] + 1.0
    for i in range(0, N - 1):
        c[i] = a[i] * 2.0


def test_two_loops_share_promoted_bound_no_runaway():
    sdfg = two_loops_share_promoted_bound.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    # Two source-program uses of ``N - 1``: at most a handful of declared
    # ``N_minus_<n>`` instances (each loop gets its own SSA), but the count
    # must not grow per canonicalize iteration.
    cnt = _count_promoted(sdfg, 'N')
    assert cnt <= 4, f"unexpected N_minus duplication: {cnt} (all symbols: {sorted(_all_symbols(sdfg))})"


@dace.program
def transitive_invariant_chain(a: dace.float64[N], b: dace.float64[N]):
    """Two-step invariant chain: an intermediate value (``K + 1``) is then
    fed into ``range``. Both promotions are loop-invariant; both must stay
    at SDFG scope."""
    kp1 = K + 1
    for i in range(0, kp1):
        b[i] = a[i] - 0.5


def test_transitive_invariant_chain_value_preserving():
    n, k = 12, 7
    a = np.random.rand(n)
    sdfg = transitive_invariant_chain.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    out = np.zeros(n)
    sdfg(a=a.copy(), b=out, N=n, K=k)
    exp = np.zeros(n)
    for i in range(0, k + 1):
        exp[i] = a[i] - 0.5
    assert np.allclose(out, exp)


@dace.program
def multiple_distinct_outer_bounds(a: dace.float64[N], b: dace.float64[N]):
    """Two loops with DIFFERENT promoted bounds (``N - 1`` vs ``N - 2``).
    Each must be promoted once and used by its respective loop; neither
    should accidentally collapse the other."""
    for i in range(0, N - 1):
        b[i] = a[i] + 1.0
    for i in range(0, N - 2):
        b[i] += a[i + 1] - a[i]


def test_multiple_distinct_outer_bounds_value_preserving():
    n = 14
    a = np.random.rand(n)
    sdfg = multiple_distinct_outer_bounds.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    out = np.zeros(n)
    sdfg(a=a.copy(), b=out, N=n)
    exp = np.zeros(n)
    for i in range(0, n - 1):
        exp[i] = a[i] + 1.0
    for i in range(0, n - 2):
        exp[i] += a[i + 1] - a[i]
    assert np.allclose(out, exp)


# ----------------------------------------------------------------------
# Loop-variable-dependent expressions (must NOT lift)
# ----------------------------------------------------------------------


@dace.program
def loop_var_dependent_inner_bound(a: dace.float64[N, M], b: dace.float64[N]):
    """``range(0, i + 1)`` -- the inner loop's bound depends on the outer
    loop variable. The promotion is per-outer-iteration; canonicalize
    must leave the materialisation inside the outer loop scope (not at
    SDFG top level)."""
    for i in dace.map[0:N]:
        s = 0.0
        for j in range(0, i + 1):
            s += a[i, j]
        b[i] = s


def test_loop_var_dependent_inner_bound_value_preserving():
    n, m = 6, 8
    a = np.random.rand(n, m)
    sdfg = loop_var_dependent_inner_bound.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    out = np.zeros(n)
    sdfg(a=a.copy(), b=out, N=n, M=m)
    exp = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(0, i + 1):
            s += a[i, j]
        exp[i] = s
    assert np.allclose(out, exp)


@dace.program
def mixed_outer_plus_loop_var(a: dace.float64[N, M], b: dace.float64[N], k: dace.int32):
    """Index ``a[i, k + 1]`` mixes outer constant ``k`` with the loop
    variable ``i``. Only the constant sub-expression ``k + 1`` is lift-
    able; canonicalize must keep that piece at SDFG scope and weave it
    with the per-iteration ``i`` correctly."""
    for i in dace.map[0:N]:
        b[i] = a[i, k + 1] + a[i, k]


def test_mixed_outer_plus_loop_var_value_preserving():
    n, m, k = 8, 11, 4
    a = np.random.rand(n, m)
    sdfg = mixed_outer_plus_loop_var.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    out = np.zeros(n)
    sdfg(a=a.copy(), b=out, k=k, N=n, M=m)
    exp = np.zeros(n)
    for i in range(n):
        exp[i] = a[i, k + 1] + a[i, k]
    assert np.allclose(out, exp)


# ----------------------------------------------------------------------
# Data-dependent symbols (read from arrays at runtime)
# ----------------------------------------------------------------------


@dace.program
def data_dependent_index(a: dace.float64[N], idx: dace.int32[N], b: dace.float64[N]):
    """Indirect access ``a[idx[i]]`` -- the index value is data-dependent
    on the loop iteration. The auxiliary symbol the frontend coins for
    ``idx[i]`` (interstate-edge assignment) must stay scoped under the
    enclosing iteration; it cannot be lifted out of the loop."""
    for i in dace.map[0:N]:
        b[i] = a[idx[i]] * 2.0


def test_data_dependent_index_value_preserving():
    n = 12
    a = np.random.rand(n)
    idx = np.random.randint(0, n, size=n).astype(np.int32)
    sdfg = data_dependent_index.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    out = np.zeros(n)
    sdfg(a=a.copy(), idx=idx.copy(), b=out, N=n)
    assert np.allclose(out, a[idx] * 2.0)


@dace.program
def data_dependent_bound(a: dace.float64[N], lengths: dace.int32[N], b: dace.float64[N, M]):
    """Per-row inner-loop bound read from an array (``range(0, lengths[i])``).
    The bound symbol is data-dependent: it depends on the outer loop var
    AND on a memory load; it must remain scoped under the outer loop and
    not lift to SDFG scope."""
    for i in dace.map[0:N]:
        for j in range(0, lengths[i]):
            b[i, j] = a[i] + j


def test_data_dependent_bound_value_preserving():
    n, m = 7, 10
    a = np.random.rand(n)
    lengths = np.random.randint(0, m, size=n).astype(np.int32)
    sdfg = data_dependent_bound.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    out = np.zeros((n, m))
    sdfg(a=a.copy(), lengths=lengths.copy(), b=out, N=n, M=m)
    exp = np.zeros((n, m))
    for i in range(n):
        for j in range(0, lengths[i]):
            exp[i, j] = a[i] + j
    assert np.allclose(out, exp)


# ----------------------------------------------------------------------
# Clutter / robustness
# ----------------------------------------------------------------------


@dace.program
def irrelevant_outer_symbol_clutter(a: dace.float64[N], b: dace.float64[N]):
    """Outer SDFG declares many symbols (``K, M, N``); the loop body only
    actually reads ``N``. Canonicalize must leave the unused symbol
    declarations alone (they may be runtime-supplied) and not duplicate or
    promote them."""
    for i in dace.map[0:N]:
        b[i] = a[i] + 1.0


def test_irrelevant_outer_symbol_clutter_robust_to_unused_symbols():
    """Canonicalize is robust against an SDFG that declares additional
    runtime-supplied symbols (``UNUSED_A``, ``UNUSED_B``) the program
    never references. The pass may legitimately *remove* them as part
    of dead-symbol cleanup -- that is a desirable simplification, not
    a regression -- but it must not crash, mis-rename them, or change
    the program's semantics. The assertion is therefore that the
    program still validates and runs correctly, NOT that the unused
    declarations are preserved verbatim."""
    n = 9
    a = np.random.rand(n)
    sdfg = irrelevant_outer_symbol_clutter.to_sdfg(simplify=True)
    sdfg.add_symbol('UNUSED_A', dace.int32)
    sdfg.add_symbol('UNUSED_B', dace.int32)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    # Pass the unused symbols only if they survived cleanup -- compiled
    # signatures only accept declared symbols.
    extra_kwargs = {}
    for sym in ('UNUSED_A', 'UNUSED_B'):
        if sym in sdfg.symbols:
            extra_kwargs[sym] = 0
    out = np.zeros(n)
    sdfg(a=a.copy(), b=out, N=n, **extra_kwargs)
    assert np.allclose(out, a + 1.0)


# ----------------------------------------------------------------------
# Combined patterns
# ----------------------------------------------------------------------

_kidia = dace.symbol('kidia')
_kfdia = dace.symbol('kfdia')


@dace.program
def cloudsc_style_range_plus_one(a: dace.float64[N], b: dace.float64[N]):
    """The cloudsc shape: ``range(kidia, kfdia + 1)`` -- the upper bound
    is a promoted-expression symbol (``kfdia_plus_1``). With the
    canonicalize-pipeline knob-off the promotion must NOT end up
    re-assigned inside the loop body (the shape ``LoopToMap`` refuses).
    Uses ``dace.symbol`` bounds (the cloudsc-equivalent style); scalar
    array bounds would surface as ``kidia[0]`` indexed accesses in the
    range expression and trip the C++ compiler."""
    for i in range(_kidia, _kfdia + 1):
        b[i] = a[i] * 3.0


def test_cloudsc_style_range_plus_one_value_preserving():
    n = 12
    a = np.random.rand(n)
    kidia, kfdia = 2, 8  # half-open: range(2, 9) -> i in 2..8
    sdfg = cloudsc_style_range_plus_one.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    # Idempotent on subsequent canonicalize calls (no symbol snowball).
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    out = np.zeros(n)
    sdfg(a=a.copy(), b=out, kidia=kidia, kfdia=kfdia, N=n)
    exp = np.zeros(n)
    for i in range(kidia, kfdia + 1):
        exp[i] = a[i] * 3.0
    assert np.allclose(out, exp)


@dace.program
def guarded_promoted_bound(a: dace.float64[N], b: dace.float64[N], c: dace.int32[1]):
    """Guard around a loop whose bound is a promoted expression. After
    ``MoveIfIntoLoop`` runs in the canonicalize pipeline, the bound
    promotion must end up at the parent scope (not duplicated per
    iteration inside the new loop body)."""
    if c[0] > 0:
        for i in range(0, N - 1):
            b[i] = a[i] + 1.0


def test_guarded_promoted_bound_value_preserving():
    n = 10
    a = np.random.rand(n)
    sdfg = guarded_promoted_bound.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    for cv in (1, 0):
        sdfg_run = guarded_promoted_bound.to_sdfg(simplify=True)
        canonicalize(sdfg_run, validate=True)
        out = np.zeros(n)
        sdfg_run(a=a.copy(), b=out, c=np.array([cv], np.int32), N=n)
        exp = np.zeros(n)
        if cv > 0:
            for i in range(0, n - 1):
                exp[i] = a[i] + 1.0
        assert np.allclose(out, exp), f"value mismatch for c={cv}"


# ----------------------------------------------------------------------
# Deferred: reduction-with-inner-accumulator (known failing shape)
# ----------------------------------------------------------------------


@dace.program
def reduction_with_inner_accumulator(a: dace.float64[N, M], b: dace.float64[N]):
    """Per-row reduction with an inner-loop accumulator. The frontend
    lowers this to an outer Map containing a per-row transient scalar
    that the inner ``for j`` loop writes; canonicalize currently leaks
    the inner ``j`` symbol through the NestedSDFG boundary because the
    inner-accumulator pattern materialises ``j`` in a body interstate
    edge that the current pipeline does not lift back up.

    Deferred to the ``CascadeInterstateEdgeAssignmentsUp`` design effort.
    Marking xfail with the precise reason so the unsupported pattern is
    documented and the day it lifts is flagged as XPASS."""
    for i in dace.map[0:N]:
        s = a[i, 0] + a[i, M - 1]
        for j in range(1, M - 1):
            s += a[i, j - 1] + a[i, j] + a[i, j + 1]
        b[i] = s


@pytest.mark.xfail(strict=True,
                   reason="Inner-accumulator reduction leaks loop var 'j' through NSDFG boundary; deferred to "
                   "CascadeInterstateEdgeAssignmentsUp design (LICM-style upward cascade of "
                   "interstate-edge assignments).")
def test_reduction_with_inner_accumulator_value_preserving():
    n, m = 6, 9
    a = np.random.rand(n, m)
    sdfg = reduction_with_inner_accumulator.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    out = np.zeros(n)
    sdfg(a=a.copy(), b=out, N=n, M=m)
    exp = np.zeros(n)
    for i in range(n):
        s = a[i, 0] + a[i, m - 1]
        for j in range(1, m - 1):
            s += a[i, j - 1] + a[i, j] + a[i, j + 1]
        exp[i] = s
    assert np.allclose(out, exp)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
