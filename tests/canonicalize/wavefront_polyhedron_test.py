# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for the ISL layer behind :class:`WavefrontSkew`.

``wavefront_polyhedron`` answers the two questions the skew pass cannot decide
syntactically: is a dependence domain integer-EMPTY (legality), and what are the
loop bounds after projecting the domain through a unimodular skew. Both were
covered only end-to-end through the pass, so a wrong answer could only ever
surface as a mis-skewed kernel. These pin the layer directly on hand-built
constraint sets, where the expected answer is obvious by inspection.

Constraints are sympy expressions read as ``expr >= 0``.
"""
import pytest

from dace import symbolic
from dace.transformation.passes.canonicalize import wavefront_polyhedron as poly

pytestmark = pytest.mark.skipif(not poly.HAVE_ISL, reason="islpy not installed")

#: ``0 <= u < N`` and ``0 <= v < M``.
_DIMS = ("u", "v")
_PARAMS = ("N", "M")


def _sym(expr: str):
    return symbolic.pystr_to_symbolic(expr)


def _rectangle():
    """The constraint list for ``0 <= u < N``, ``0 <= v < M``."""
    return [_sym("u"), _sym("N - 1 - u"), _sym("v"), _sym("M - 1 - v")]


def test_rectangle_is_not_empty() -> None:
    """A parametric rectangle has integer points (N, M are unconstrained, so ISL
    must not conclude emptiness from the parameters alone)."""
    assert poly.is_domain_empty(_DIMS, _PARAMS, _rectangle()) is False


def test_contradictory_constraints_are_empty() -> None:
    """``u >= 1`` together with ``u <= 0`` has no integer solution."""
    assert poly.is_domain_empty(_DIMS, _PARAMS, [_sym("u - 1"), _sym("-u")]) is True


def test_integer_gap_is_empty() -> None:
    """``2u >= 1`` and ``2u <= 1`` is satisfiable over the RATIONALS (u = 1/2) but
    empty over the integers -- the reason the legality test must be an exact
    integer query and not a relaxation."""
    assert poly.is_domain_empty(_DIMS, _PARAMS, [_sym("2*u - 1"), _sym("1 - 2*u")]) is True


def test_skew_bounds_refuses_non_unimodular_tau() -> None:
    """``tau = (2, 3)`` has no single-coordinate integer complement, so the skew is
    refused rather than silently approximated."""
    assert poly.skew_bounds(_DIMS, _PARAMS, _rectangle(), (2, 3), "t", "p") is None


@pytest.mark.parametrize("tau", [(1, 1), (1, -1), (1, 2)])
def test_skew_bounds_shallow_family(tau) -> None:
    """``|a| == 1``: the parallel axis is ``v``; every bound list is populated."""
    bounds = poly.skew_bounds(_DIMS, _PARAMS, _rectangle(), tau, "t", "p")
    assert bounds is not None, f"tau={tau} should be expressible"
    assert bounds.t_lo_terms and bounds.t_hi_terms
    assert bounds.p_lo_terms and bounds.p_hi_terms


@pytest.mark.parametrize("tau", [(2, 1), (2, -1)])
def test_skew_bounds_steep_family(tau) -> None:
    """``|b| == 1`` (Gauss-Seidel steep diagonal): the parallel axis is ``u`` and the
    ``p`` bounds may carry the int-division that scaling by ``|a| > 1`` introduces."""
    bounds = poly.skew_bounds(_DIMS, _PARAMS, _rectangle(), tau, "t", "p")
    assert bounds is not None, f"tau={tau} should be expressible"
    assert bounds.t_lo_terms and bounds.t_hi_terms
    assert bounds.p_lo_terms and bounds.p_hi_terms


def test_skew_bounds_t_range_covers_the_diagonal() -> None:
    """For the unit skew ``t = u + v`` over the rectangle, ``t`` runs from 0 to
    ``(N - 1) + (M - 1)`` -- the anti-diagonal count. Pinned symbolically so a
    projection that dropped a term (reading per-constraint instead of via
    dim_min/dim_max on the projected set) is caught."""
    bounds = poly.skew_bounds(_DIMS, _PARAMS, _rectangle(), (1, 1), "t", "p")
    assert bounds is not None
    lo = {str(symbolic.simplify(term)) for term in bounds.t_lo_terms}
    hi = {str(symbolic.simplify(term)) for term in bounds.t_hi_terms}
    assert "0" in lo, f"expected a 0 lower term, got {lo}"
    assert any("N" in h and "M" in h for h in hi), f"expected an N+M upper term, got {hi}"


def test_skew_bounds_seidel_diagonal_over_rebased_box() -> None:
    """The ISL projection behind polybench ``seidel_2d``'s wavefront.

    Once ``NormalizeLoopAndMapOrigin`` rebases both axes, seidel_2d's ``(i, j)`` nest is the square
    ``0 <= u <= N-3``, ``0 <= v <= N-3``, and the legal schedule is the STEEP Gauss-Seidel diagonal
    ``tau = (2, 1)`` -- the one whose stored deps ``{(0,-1), (-1,0), (-1,-1), (-1,1)}`` need
    ``a > b > 0``. Projecting that skew must give ``t`` running from ``0`` to ``3*(N-3) = 3N - 9``,
    which is exactly the diagonal bound the end-to-end test
    ``canonicalize_wavefront_skew_test.test_seidel_2d_ij_wavefront_skews_under_reconstruct_plus_origin_knobs``
    observes on the rewritten SDFG. Pinned here directly on the constraint set so a projection
    regression is attributed to this layer rather than to the pass.
    """
    box = [_sym("u"), _sym("N - 3 - u"), _sym("v"), _sym("N - 3 - v")]
    bounds = poly.skew_bounds(_DIMS, ("N", ), box, (2, 1), "t", "p")
    assert bounds is not None, "the steep (2, 1) diagonal must be expressible"
    lo = {str(symbolic.simplify(term)) for term in bounds.t_lo_terms}
    assert "0" in lo, f"expected a 0 lower term, got {lo}"
    hi = {str(symbolic.simplify(term)) for term in bounds.t_hi_terms}
    assert any(symbolic.simplify(term - _sym("3*N - 9")) == 0 for term in bounds.t_hi_terms), \
        f"expected an upper term equal to 3*N - 9, got {hi}"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
