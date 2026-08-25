# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``LoopToScan`` on the first-order LINEAR recurrence ``x[i] = c[i]*x[i-1] + d[i]``."""
from pathlib import Path

import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes.scan import COEF_CONNECTOR_NAME, Scan, ScanOp
from dace.transformation.passes.lift_preprocess import LiftPreprocess
from dace.transformation.passes.loop_to_scan import LoopToScan

N = dace.symbol('N', dtype=dace.int64)


def lift(sdfg: dace.SDFG) -> int:
    """Run the matcher's required preprocessing, then the pass. Returns the lift count."""
    LiftPreprocess().apply_pass(sdfg, {})
    return LoopToScan().apply_pass(sdfg, {}) or 0


def affine_nodes(sdfg: dace.SDFG):
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Scan) and n.op is ScanOp.AFFINE]


@dace.program
def linear_recurrence(x: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
    for i in range(1, N):
        x[i] = c[i] * x[i - 1] + d[i]


@dace.program
def constant_coefficient(x: dace.float64[N], d: dace.float64[N]):
    for i in range(1, N):
        x[i] = 0.75 * x[i - 1] + d[i]


@dace.program
def compound_coefficient(x: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
    for i in range(1, N):
        x[i] = (c[i] - d[i]) * x[i - 1] + c[i] * d[i]


@dace.program
def quadratic_in_carry(x: dace.float64[N], d: dace.float64[N]):
    for i in range(1, N):
        x[i] = x[i - 1] * x[i - 1] + d[i]


@dace.program
def second_order(x: dace.float64[N], d: dace.float64[N]):
    for i in range(2, N):
        x[i] = 0.5 * x[i - 1] + 0.25 * x[i - 2] + d[i]


def contracting(n: int):
    rng = np.random.default_rng(7)
    return ((0.3 + 0.4 * rng.random(n)).astype(np.float64), rng.standard_normal(n).astype(np.float64),
            rng.standard_normal(n).astype(np.float64))


def test_linear_recurrence_lifts_to_an_affine_scan():
    """The shape the scalar matchers refuse: one AFFINE Scan node, and the loop is gone."""
    sdfg = linear_recurrence.to_sdfg(simplify=True)
    assert lift(sdfg) == 1
    nodes = affine_nodes(sdfg)
    assert len(nodes) == 1
    assert COEF_CONNECTOR_NAME in nodes[0].in_connectors

    # The build loop stays -- LoopToMap parallelizes it afterwards, as it does for the scalar
    # ops. What must be gone is the carry: no body state may touch the carrier any more, which
    # is the whole reason the loop is now data-parallel.
    loops = [
        b for b in sdfg.all_control_flow_blocks()
        if isinstance(b, dace.sdfg.state.LoopRegion) and b.loop_variable == 'i'
    ]
    assert len(loops) == 1
    body_reads = {
        n.data
        for blk in loops[0].all_control_flow_blocks() if isinstance(blk, dace.SDFGState) for n in blk.data_nodes()
    }
    assert 'x' not in body_reads
    assert any(name.startswith('_scan_coef_') for name in body_reads)
    assert any(name.startswith('_scan_in_') for name in body_reads)


@pytest.mark.parametrize('n', [1, 2, 4, 129, 20011])
def test_lifted_linear_recurrence_matches_the_sequential_loop(n):
    c, d, _ = contracting(n)
    x0 = np.zeros(n, dtype=np.float64)
    x0[0] = 0.5

    reference = x0.copy()
    for i in range(1, n):
        reference[i] = c[i] * reference[i - 1] + d[i]

    # ``n == 1`` leaves the loop with an empty iteration space, so the lifted form runs a
    # zero-length scan writing an empty range -- the shape that has to be a no-op, not a crash.
    sdfg = linear_recurrence.to_sdfg(simplify=True)
    assert lift(sdfg) == 1
    got = x0.copy()
    sdfg(x=got, c=c, d=d, N=n)
    assert np.allclose(got, reference, rtol=0, atol=1e-11)


def test_the_lift_is_what_lets_loop_to_map_parallelize_it():
    """The point of the whole exercise: unlifted, ``LoopToMap`` refuses; lifted, it applies."""
    from dace.sdfg import nodes as dnodes
    from dace.transformation.interstate import LoopToMap

    unlifted = linear_recurrence.to_sdfg(simplify=True)
    assert unlifted.apply_transformations_repeated([LoopToMap]) == 0
    assert not [n for n, _ in unlifted.all_nodes_recursive() if isinstance(n, dnodes.MapEntry)]

    lifted = linear_recurrence.to_sdfg(simplify=True)
    assert lift(lifted) == 1
    assert lifted.apply_transformations_repeated([LoopToMap]) == 1
    assert [n for n, _ in lifted.all_nodes_recursive() if isinstance(n, dnodes.MapEntry)]
    assert not [b for b in lifted.all_control_flow_blocks() if isinstance(b, dace.sdfg.state.LoopRegion)]


def test_constant_coefficient_lifts_and_matches():
    """The constant-coefficient case (an IIR / exponential-moving-average body)."""
    n = 5003
    _, d, _ = contracting(n)
    x0 = np.zeros(n, dtype=np.float64)
    x0[0] = 1.5
    reference = x0.copy()
    for i in range(1, n):
        reference[i] = 0.75 * reference[i - 1] + d[i]

    sdfg = constant_coefficient.to_sdfg(simplify=True)
    assert lift(sdfg) == 1
    got = x0.copy()
    sdfg(x=got, d=d, N=n)
    assert np.allclose(got, reference, rtol=0, atol=1e-11)


def test_compound_coefficient_is_split_symbolically():
    """``(c-d)*x[i-1] + c*d`` -- the coefficient is an expression, not a single connector read."""
    n = 3001
    c, d, _ = contracting(n)
    c = 0.2 + 0.3 * c
    d = 0.1 * d
    x0 = np.zeros(n, dtype=np.float64)
    x0[0] = 0.25
    reference = x0.copy()
    for i in range(1, n):
        reference[i] = (c[i] - d[i]) * reference[i - 1] + c[i] * d[i]

    sdfg = compound_coefficient.to_sdfg(simplify=True)
    assert lift(sdfg) == 1
    got = x0.copy()
    sdfg(x=got, c=c, d=d, N=n)
    assert np.allclose(got, reference, rtol=0, atol=1e-11)


def test_nonlinear_carry_is_refused():
    """``x[i-1]*x[i-1]`` is associative under composition but has no bounded-width carry."""
    sdfg = quadratic_in_carry.to_sdfg(simplify=True)
    lift(sdfg)
    assert not affine_nodes(sdfg)


def test_second_order_recurrence_is_refused():
    """A two-deep carry needs a 2x2 matrix, not the scalar affine map. Not matched."""
    sdfg = second_order.to_sdfg(simplify=True)
    lift(sdfg)
    assert not affine_nodes(sdfg)


def test_refused_recurrences_still_compute_the_right_answer():
    """Refusing must leave a working loop, not a broken graph."""
    n = 101
    _, d, _ = contracting(n)
    d = 0.01 * d
    x0 = np.zeros(n, dtype=np.float64)
    x0[0] = 0.1
    reference = x0.copy()
    for i in range(1, n):
        reference[i] = reference[i - 1] * reference[i - 1] + d[i]

    sdfg = quadratic_in_carry.to_sdfg(simplify=True)
    lift(sdfg)
    got = x0.copy()
    sdfg(x=got, d=d, N=n)
    assert np.allclose(got, reference, rtol=0, atol=1e-12)


def test_plain_sum_scan_still_takes_the_scalar_path():
    """The affine matcher runs last; a plain prefix sum must keep its SUM lowering."""

    @dace.program
    def prefix_sum(x: dace.float64[N], d: dace.float64[N]):
        for i in range(1, N):
            x[i] = x[i - 1] + d[i]

    sdfg = prefix_sum.to_sdfg(simplify=True)
    assert lift(sdfg) == 1
    ops = [n.op for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Scan)]
    assert ops == [ScanOp.SUM]


# --------------------------------------------------------------------------------------------
# End-to-end through the production recipes. A lift that only holds under a bare ``LoopToScan``
# is not worth much: what matters is that the shape survives canonicalize, survives the
# vectorizer running over the build map it leaves behind, and still computes the recurrence.
# --------------------------------------------------------------------------------------------


@dace.program
def two_recurrences(x: dace.float64[N], y: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
    for i in range(1, N):
        x[i] = c[i] * x[i - 1] + d[i]
    for i in range(1, N):
        y[i] = d[i] * y[i - 1] + c[i]


@dace.program
def divided_coefficient(x: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
    for i in range(1, N):
        x[i] = (c[i] / 2.0) * x[i - 1] + d[i] / c[i]


@dace.program
def affine_then_consumer(x: dace.float64[N], c: dace.float64[N], d: dace.float64[N], z: dace.float64[N]):
    for i in range(1, N):
        x[i] = c[i] * x[i - 1] + d[i]
    for i in dace.map[0:N]:
        z[i] = x[i] * 2.0


def apply_recipe(sdfg: dace.SDFG, config: str) -> dace.SDFG:
    """Run one production pipeline configuration over ``sdfg`` in place."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from corpus.measure_parallelization import apply_config, cpu_params
    apply_config(sdfg, config, cpu_params())
    return sdfg


def recurrence_reference(seed: float, coef: np.ndarray, delta: np.ndarray) -> np.ndarray:
    out = np.zeros(coef.shape[0], dtype=np.float64)
    out[0] = seed
    for i in range(1, out.shape[0]):
        out[i] = coef[i] * out[i - 1] + delta[i]
    return out


@pytest.mark.parametrize('config', ['canon', 'canon+vec'])
def test_recurrence_survives_the_production_recipes(config):
    """The plain shape, end to end. ``canon+vec`` is the one that runs the vectorizer."""
    n = 4001
    c, d, _ = contracting(n)
    sdfg = apply_recipe(linear_recurrence.to_sdfg(simplify=True), config)
    assert affine_nodes(sdfg)
    got = np.zeros(n)
    got[0] = 0.5
    sdfg(x=got, c=c, d=d, N=n)
    assert np.allclose(got, recurrence_reference(0.5, c, d), rtol=0, atol=1e-11)


@pytest.mark.parametrize('config', ['canon', 'canon+vec'])
def test_two_independent_recurrences_both_lift(config):
    """Two carriers in one program, each its own affine scan."""
    n = 513
    c, d, _ = contracting(n)
    sdfg = apply_recipe(two_recurrences.to_sdfg(simplify=True), config)
    assert len(affine_nodes(sdfg)) == 2
    x = np.zeros(n)
    x[0] = 0.5
    y = np.zeros(n)
    y[0] = 0.25
    sdfg(x=x, y=y, c=c, d=d, N=n)
    assert np.allclose(x, recurrence_reference(0.5, c, d), rtol=0, atol=1e-11)
    assert np.allclose(y, recurrence_reference(0.25, d, c), rtol=0, atol=1e-11)


@pytest.mark.parametrize('config', ['canon', 'canon+vec'])
def test_coefficient_and_offset_may_be_divisions(config):
    """``(c[i]/2)*x[i-1] + d[i]/c[i]`` -- both halves are expressions the build tasklets rebuild."""
    n = 513
    c, d, _ = contracting(n)
    sdfg = apply_recipe(divided_coefficient.to_sdfg(simplify=True), config)
    assert affine_nodes(sdfg)
    got = np.zeros(n)
    got[0] = 0.4
    sdfg(x=got, c=c, d=d, N=n)
    assert np.allclose(got, recurrence_reference(0.4, c / 2.0, d / c), rtol=0, atol=1e-11)


@pytest.mark.parametrize('config', ['canon', 'canon+vec'])
def test_a_later_consumer_of_the_carrier_reads_the_scanned_values(config):
    """A downstream map reading ``x`` must see the scan's output, not the pre-loop array.

    The rewrite moves the carrier's write out of the loop and into a post-loop state, so a
    consumer that used to be ordered after the loop has to end up ordered after THAT state
    instead. If the ordering were lost this reads the untouched input and the check fails.
    """
    n = 513
    c, d, _ = contracting(n)
    sdfg = apply_recipe(affine_then_consumer.to_sdfg(simplify=True), config)
    assert affine_nodes(sdfg)
    x = np.zeros(n)
    x[0] = 0.6
    z = np.zeros(n)
    sdfg(x=x, c=c, d=d, z=z, N=n)
    assert np.allclose(x, recurrence_reference(0.6, c, d), rtol=0, atol=1e-11)
    assert np.allclose(z, x * 2.0, rtol=0, atol=1e-13)


def test_vectorizer_tiles_the_build_map_it_is_left_with():
    """``canon+vec`` must actually process the build map, not merely decline without crashing.

    The coefficient here is a computed expression, so the build survives as a real map rather
    than collapsing into a direct wire to an existing array. The vectorizer splits a tiled map
    into a main and a remainder map, so the map count going UP across ``canon -> canon+vec`` is
    the observable that separates "tiled" from "left alone".
    """
    from dace.sdfg import nodes as dnodes

    def map_count(config):
        sdfg = apply_recipe(compound_coefficient.to_sdfg(simplify=True), config)
        assert affine_nodes(sdfg)
        return sum(1 for n_, _ in sdfg.all_nodes_recursive() if isinstance(n_, dnodes.MapEntry))

    plain, vectorized = map_count('canon'), map_count('canon+vec')
    assert plain >= 1, 'the computed coefficient should leave a build map behind'
    assert vectorized > plain, f'vectorizer left the build map alone ({plain} -> {vectorized})'


def test_a_trivial_build_collapses_instead_of_leaving_a_map():
    """When ``c`` and ``d`` are already arrays there is nothing to build.

    The coefficient and offset buffers are then plain copies of existing arrays, and
    canonicalize wires the Scan straight to them -- so the right outcome is NO loop and NO map
    at all, which is a stronger result than a parallelized build and worth pinning as such.
    """
    from dace.sdfg import nodes as dnodes

    sdfg = apply_recipe(linear_recurrence.to_sdfg(simplify=True), 'canon')
    assert affine_nodes(sdfg)
    assert not [b for b in sdfg.all_control_flow_blocks() if isinstance(b, dace.sdfg.state.LoopRegion)]
    assert not [n_ for n_, _ in sdfg.all_nodes_recursive() if isinstance(n_, dnodes.MapEntry)]


if __name__ == '__main__':
    test_linear_recurrence_lifts_to_an_affine_scan()
    test_lifted_linear_recurrence_matches_the_sequential_loop(129)
    print('ok')
