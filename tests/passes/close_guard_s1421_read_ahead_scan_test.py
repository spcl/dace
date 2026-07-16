# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Guard-closing regression: the ``s1421`` read-ahead shape must lift to an UNCONDITIONAL Map.

TSVC ``s1421_d_single``::

    half = LEN_1D // 2
    for i in range(half):
        b[i] = b[half + i] + a[i]

writes ``b[0 : half)`` while reading ``b[half : 2 * half)`` -- two DISJOINT regions, so the
loop is DOALL for every ``LEN_1D >= 0``. ``LoopToScan``'s matcher nevertheless saw a carrier
read AND written on the same axis and derived a "scan stride" ``k_w - k_r = -int_floor(LEN_1D, 2)``.
``int_floor`` carries no SymPy sign rule, so ``is_nonpositive`` returned ``None`` (sign UNKNOWN),
the stride was admitted as symbolic, and the loop was specialized into
``if (-int_floor(LEN_1D, 2)) >= 1: <Scan> else: <pinned sequential loop>``. That predicate is
always FALSE for a nonnegative ``LEN_1D``, so the scan branch was dead and the sequential
fallback ran every single time -- correct, but never parallel.

:func:`_provably_nonpositive` rewrites ``int_floor``/``int_ceil`` to SymPy's sign-propagating
``floor``/``ceiling`` over nonnegative-rebuilt symbols, so the stride is now PROVEN ``<= 0``,
the bogus scan match is refused, and ``LoopToMap`` lifts the loop to a bare Map.
"""
import copy

import numpy as np
import sympy

import dace
from dace import symbolic
from dace.sdfg import nodes as nd
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.loop_to_scan import _admissible_scan_stride, _provably_nonpositive

LEN_1D = dace.symbol('LEN_1D')

#: The corpus' CPU canonicalize knobs (``tests/corpus/measure_parallelization.py`` ``cpu_params``).
_CPU = dict(target='cpu',
            peel_limit=4,
            break_anti_dependence=True,
            interchange_carry_with_map=True,
            scatter_to_guarded_maps=True)


def _counts(sdfg):
    """``(loops, maps, conditionals)`` over the whole SDFG."""
    loops = sum(1 for c in sdfg.all_control_flow_regions() if isinstance(c, LoopRegion))
    maps = sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nd.MapEntry))
    conds = sum(1 for c in sdfg.all_control_flow_regions() if isinstance(c, ConditionalBlock))
    return loops, maps, conds


def test_int_floor_stride_is_proven_nonpositive():
    """The ``s1421`` stride is PROVEN ``<= 0``, so it is not an admissible scan stride."""
    n = symbolic.symbol('LEN_1D', dtype=dace.int64, nonnegative=True)
    stride = -symbolic.int_floor(n, 2)

    # Bare SymPy cannot sign ``int_floor``; the floor rewrite is what makes the proof go through.
    assert stride.is_nonpositive is None
    assert _provably_nonpositive(stride)
    assert _admissible_scan_stride(stride) is None


def test_unknown_sign_stride_still_admitted():
    """A genuine symbolic stride (``a[i] = a[i - K] + x[i]``) keeps its ``K >= 1`` guard.

    ``K`` may be ``0`` at runtime, so its sign is NOT provable and the guard MUST stay.
    """
    k = symbolic.symbol('K', dtype=dace.int64, nonnegative=True)
    assert not _provably_nonpositive(k)
    assert _admissible_scan_stride(k) is k

    # A positive-but-not-provably-so floor stride is likewise still admitted (only PROVABLE
    # nonpositivity refuses).
    n = symbolic.symbol('LEN_1D', dtype=dace.int64, nonnegative=True)
    assert not _provably_nonpositive(symbolic.int_floor(n, 2))


def test_s1421_read_ahead_closes_to_unconditional_map():
    """End-to-end: no guard, no pinned fallback, one bare Map -- and bit-exact."""

    @dace.program
    def s1421_read_ahead_half(b: dace.float64[LEN_1D], a: dace.float64[LEN_1D]):
        half = LEN_1D // 2
        for i in range(half):
            b[i] = b[half + i] + a[i]

    sdfg = s1421_read_ahead_half.to_sdfg(simplify=True)
    canon = copy.deepcopy(sdfg)
    canonicalize(canon, validate=True, validate_all=False, **_CPU)

    loops, maps, conds = _counts(canon)
    # The guard is CLOSED: no ``if cond: Map else: seq`` conditional, hence no pinned
    # sequential fallback (the ``g`` / ``guarded_fallback_loops`` metric goes 1 -> 0), and the
    # loop survives as an unconditional Map rather than a residual sequential LoopRegion.
    assert conds == 0, 'the always-false scan guard must not be emitted'
    assert loops == 0, 'the read-ahead loop must not stay sequential'
    assert maps >= 1, 'the read-ahead loop must be an unconditional Map'

    n = 64
    rng = np.random.default_rng(1421)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)

    expected_b = b.copy()
    half = n // 2
    for i in range(half):
        expected_b[i] = expected_b[half + i] + a[i]

    fin = finalize_for_target(copy.deepcopy(canon), 'cpu')
    fin.name = f'{fin.name}_close_guard_s1421'
    compiled = fin.compile()
    for _ in range(10):  # in-process repeat: defeats the cross-process canon flake
        got_b, got_a = b.copy(), a.copy()
        compiled(b=got_b, a=got_a, LEN_1D=n)
        assert np.allclose(got_b, expected_b, rtol=1e-9, atol=1e-9, equal_nan=True)
        assert np.allclose(got_a, a, rtol=1e-9, atol=1e-9, equal_nan=True)


if __name__ == '__main__':
    test_int_floor_stride_is_proven_nonpositive()
    test_unknown_sign_stride_still_admitted()
    test_s1421_read_ahead_closes_to_unconditional_map()
    print('OK')
