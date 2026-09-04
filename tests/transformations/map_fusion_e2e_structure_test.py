# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" End-to-end and structural tests for vertical and horizontal map fusion.

    The kernels mirror the ICON velocity-advection neighbour-gather shape
    (``out[i, k] = c1 * w[cidx[i, 0], k] - c2 * w[vidx[i, 1], k]``): the
    first array dimension is gathered through an int32 neighbour-index
    table while the level dimension stays structured. Each test builds the
    SDFG through the python frontend, captures a numpy oracle (and a
    deep-copied pre-fusion run), applies :class:`MapFusionVertical` or
    :class:`MapFusionHorizontal` repeatedly, asserts the expected map count
    via :func:`_n_maps`, validates the SDFG, and checks the post-fusion
    result against the oracle with :func:`numpy.allclose`.
"""
import copy

import numpy as np

import dace
from dace.sdfg import nodes
from dace.transformation.dataflow.map_fusion_vertical import MapFusionVertical
from dace.transformation.dataflow.map_fusion_horizontal import MapFusionHorizontal

N = dace.symbol('N')
L = dace.symbol('L')


def _n_maps(sdfg):
    """ Count :class:`~dace.sdfg.nodes.MapEntry` nodes through the whole SDFG.

    :param sdfg: The SDFG to inspect.
    :returns: The number of map entries, recursing into nested SDFGs.
    """
    return len([n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)])


def _make_inputs(n, levels, seed):
    """ Build a reproducible set of velocity-advection-shaped inputs.

    :param n: The gathered (edge) dimension size.
    :param levels: The structured (level) dimension size.
    :param seed: The RNG seed.
    :returns: A tuple ``(w, v, cidx, vidx)`` with in-range int32 indices.
    """
    rng = np.random.default_rng(seed)
    w = rng.random((n, levels))
    v = rng.random((n, levels))
    cidx = rng.integers(0, n, size=(n, 2), dtype=np.int32)
    vidx = rng.integers(0, n, size=(n, 2), dtype=np.int32)
    return w, v, cidx, vidx


@dace.program
def vertical_producer_consumer(w: dace.float64[N, L], v: dace.float64[N, L], cidx: dace.int32[N, 2],
                               vidx: dace.int32[N, 2], b: dace.float64[N, L]):
    t = np.empty_like(w)
    for i, k in dace.map[0:N, 0:L]:
        t[i, k] = w[cidx[i, 0], k] * 2.0
    for i, k in dace.map[0:N, 0:L]:
        b[i, k] = t[i, k] + v[vidx[i, 1], k]


@dace.program
def vertical_chain_of_three(w: dace.float64[N, L], v: dace.float64[N, L], cidx: dace.int32[N, 2],
                            vidx: dace.int32[N, 2], b: dace.float64[N, L]):
    t1 = np.empty_like(w)
    t2 = np.empty_like(w)
    for i, k in dace.map[0:N, 0:L]:
        t1[i, k] = w[cidx[i, 0], k] * 2.0
    for i, k in dace.map[0:N, 0:L]:
        t2[i, k] = t1[i, k] + v[vidx[i, 1], k]
    for i, k in dace.map[0:N, 0:L]:
        b[i, k] = t2[i, k] - 1.0


@dace.program
def horizontal_two_gathers(w: dace.float64[N, L], v: dace.float64[N, L], cidx: dace.int32[N, 2], vidx: dace.int32[N, 2],
                           b: dace.float64[N, L], d: dace.float64[N, L]):
    for i, k in dace.map[0:N, 0:L]:
        b[i, k] = w[cidx[i, 0], k] + 1.0
    for i, k in dace.map[0:N, 0:L]:
        d[i, k] = v[vidx[i, 1], k] * 3.0


@dace.program
def vertical_dependent_gather(w: dace.float64[N, L], cidx: dace.int32[N, 2], vidx: dace.int32[N, 2],
                              b: dace.float64[N, L]):
    # The consumer reads the transient through a *different* gathered index
    # (vidx[i, 1]) than the structured slot the producer wrote (i), so the
    # producer-consumer link is a genuine indirect cross-iteration
    # dependency that fusion must not silently break. The transient is
    # fully defined (one structured write per row) so the numpy oracle and
    # the SDFG agree bit-for-bit regardless of fusion.
    t = np.empty_like(w)
    for i, k in dace.map[0:N, 0:L]:
        t[i, k] = w[cidx[i, 0], k] * 2.0
    for i, k in dace.map[0:N, 0:L]:
        b[i, k] = t[vidx[i, 1], k] + 1.0


def test_vertical_producer_consumer():
    n, levels = 32, 16
    w, v, cidx, vidx = _make_inputs(n, levels, seed=1)

    oracle = np.empty((n, levels))
    for i in range(n):
        for k in range(levels):
            oracle[i, k] = w[cidx[i, 0], k] * 2.0 + v[vidx[i, 1], k]

    sdfg = vertical_producer_consumer.to_sdfg(simplify=True)
    n_before = _n_maps(sdfg)
    assert n_before == 2

    ref_sdfg = copy.deepcopy(sdfg)
    b_ref = np.zeros((n, levels))
    ref_sdfg(w=w, v=v, cidx=cidx, vidx=vidx, b=b_ref, N=n, L=levels)
    assert np.allclose(b_ref, oracle)

    applied = sdfg.apply_transformations_repeated(MapFusionVertical)
    assert applied >= 1
    sdfg.validate()
    assert _n_maps(sdfg) == 1

    b_out = np.zeros((n, levels))
    sdfg(w=w, v=v, cidx=cidx, vidx=vidx, b=b_out, N=n, L=levels)
    assert np.allclose(b_out, oracle)


def test_vertical_chain_of_three():
    n, levels = 24, 12
    w, v, cidx, vidx = _make_inputs(n, levels, seed=2)

    oracle = np.empty((n, levels))
    for i in range(n):
        for k in range(levels):
            oracle[i, k] = (w[cidx[i, 0], k] * 2.0 + v[vidx[i, 1], k]) - 1.0

    sdfg = vertical_chain_of_three.to_sdfg(simplify=True)
    assert _n_maps(sdfg) == 3

    ref_sdfg = copy.deepcopy(sdfg)
    b_ref = np.zeros((n, levels))
    ref_sdfg(w=w, v=v, cidx=cidx, vidx=vidx, b=b_ref, N=n, L=levels)
    assert np.allclose(b_ref, oracle)

    applied = sdfg.apply_transformations_repeated(MapFusionVertical)
    assert applied >= 1
    sdfg.validate()
    assert _n_maps(sdfg) == 1

    b_out = np.zeros((n, levels))
    sdfg(w=w, v=v, cidx=cidx, vidx=vidx, b=b_out, N=n, L=levels)
    assert np.allclose(b_out, oracle)


def test_horizontal_two_gathers():
    n, levels = 32, 16
    w, v, cidx, vidx = _make_inputs(n, levels, seed=3)

    oracle_b = np.empty((n, levels))
    oracle_d = np.empty((n, levels))
    for i in range(n):
        for k in range(levels):
            oracle_b[i, k] = w[cidx[i, 0], k] + 1.0
            oracle_d[i, k] = v[vidx[i, 1], k] * 3.0

    sdfg = horizontal_two_gathers.to_sdfg(simplify=True)
    assert _n_maps(sdfg) == 2

    ref_sdfg = copy.deepcopy(sdfg)
    b_ref = np.zeros((n, levels))
    d_ref = np.zeros((n, levels))
    ref_sdfg(w=w, v=v, cidx=cidx, vidx=vidx, b=b_ref, d=d_ref, N=n, L=levels)
    assert np.allclose(b_ref, oracle_b)
    assert np.allclose(d_ref, oracle_d)

    applied = sdfg.apply_transformations_repeated(MapFusionHorizontal)
    assert applied >= 1
    sdfg.validate()
    assert _n_maps(sdfg) == 1

    b_out = np.zeros((n, levels))
    d_out = np.zeros((n, levels))
    sdfg(w=w, v=v, cidx=cidx, vidx=vidx, b=b_out, d=d_out, N=n, L=levels)
    assert np.allclose(b_out, oracle_b)
    assert np.allclose(d_out, oracle_d)


def test_vertical_dependent_gather_preserves_numerics():
    # Genuine cross-iteration dependency through the gathered transient:
    # whether or not fusion applies, the numerics must match the oracle.
    n, levels = 28, 10
    w, _, cidx, vidx = _make_inputs(n, levels, seed=4)

    oracle = np.empty((n, levels))
    t = np.empty((n, levels))
    for i in range(n):
        for k in range(levels):
            t[i, k] = w[cidx[i, 0], k] * 2.0
    for i in range(n):
        for k in range(levels):
            oracle[i, k] = t[vidx[i, 1], k] + 1.0

    sdfg = vertical_dependent_gather.to_sdfg(simplify=True)

    ref_sdfg = copy.deepcopy(sdfg)
    b_ref = np.zeros((n, levels))
    ref_sdfg(w=w, cidx=cidx, vidx=vidx, b=b_ref, N=n, L=levels)
    assert np.allclose(b_ref, oracle)

    applied = sdfg.apply_transformations_repeated(MapFusionVertical)
    sdfg.validate()
    # Fusion may legitimately decline here because the consumer reads the
    # transient through a different gathered index than the producer wrote
    # (an across-iteration dependency that fusion must not silently break).
    # Either way the post-pass SDFG stays numerically faithful.
    assert _n_maps(sdfg) >= 1

    b_out = np.zeros((n, levels))
    sdfg(w=w, cidx=cidx, vidx=vidx, b=b_out, N=n, L=levels)
    assert np.allclose(b_out, oracle)


if __name__ == '__main__':
    test_vertical_producer_consumer()
    test_vertical_chain_of_three()
    test_horizontal_two_gathers()
    test_vertical_dependent_gather_preserves_numerics()
