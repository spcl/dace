# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Canonicalize on the ECRAD two-stream transcendental + clamp shape.

Distilled from ``radiation_two_stream.F90:210-243`` (Meador & Weaver
two-stream reflectance/transmittance). Each spectral g-point is
independent (parallel), with:

* a ``SQRT(MAX(..., 1e-12))`` clamp guarding the eigenvalue,
* an ``EXP(-k*od)`` pair,
* a data-dependent branch on the optical depth ``od`` (full formula vs a
  small-``od`` linear approximation),
* and trailing ``MAX(0, MIN(x, 1))`` saturation clamps on the outputs.

Canonical-form contract: the per-g-point body is fully parallel and must
become an elementwise Map; the transcendental ops and the saturation
clamps must survive (canonicalize must not drop or reorder them in a way
that changes results). Value-preserving against a numpy oracle that
mirrors the same ``exp`` / ``sqrt`` / ``min`` / ``max`` chain.
"""
import math

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize import canonicalize

NG = dace.symbol('NG')


@dace.program
def two_stream_reftrans(od: dace.float64[NG], g1: dace.float64[NG], g2: dace.float64[NG], ref: dace.float64[NG],
                        trans: dace.float64[NG]):
    """Per-g-point two-stream reflectance/transmittance with a small-``od``
    branch and saturation clamps. Fully parallel over ``jg``."""
    for jg in dace.map[0:NG]:
        if od[jg] > 1.0e-3:
            k = math.sqrt(max((g1[jg] - g2[jg]) * (g1[jg] + g2[jg]), 1.0e-12))
            e = math.exp(-k * od[jg])
            e2 = e * e
            rf = 1.0 / (k + g1[jg] + (k - g1[jg]) * e2)
            ref[jg] = g2[jg] * (1.0 - e2) * rf
            trans[jg] = 2.0 * k * e * rf
        else:
            ref[jg] = g2[jg] * od[jg]
            trans[jg] = 1.0 - g1[jg] * od[jg]
        # Saturation clamps (must survive canonicalize).
        ref[jg] = max(0.0, min(ref[jg], 1.0))
        trans[jg] = max(0.0, min(trans[jg], 1.0 - ref[jg]))


def _two_stream_oracle(od, g1, g2):
    n = od.shape[0]
    ref = np.zeros(n)
    trans = np.zeros(n)
    for jg in range(n):
        if od[jg] > 1.0e-3:
            k = math.sqrt(max((g1[jg] - g2[jg]) * (g1[jg] + g2[jg]), 1.0e-12))
            e = math.exp(-k * od[jg])
            e2 = e * e
            rf = 1.0 / (k + g1[jg] + (k - g1[jg]) * e2)
            ref[jg] = g2[jg] * (1.0 - e2) * rf
            trans[jg] = 2.0 * k * e * rf
        else:
            ref[jg] = g2[jg] * od[jg]
            trans[jg] = 1.0 - g1[jg] * od[jg]
        ref[jg] = max(0.0, min(ref[jg], 1.0))
        trans[jg] = max(0.0, min(trans[jg], 1.0 - ref[jg]))
    return ref, trans


def test_two_stream_reftrans_value_preserving():
    n = 16
    rng = np.random.default_rng(50)
    # Mix of small-od (linear branch) and large-od (full formula) lanes.
    od = np.concatenate([rng.uniform(0.0, 5e-4, n // 2), rng.uniform(0.5, 3.0, n - n // 2)])
    g1 = rng.uniform(1.0, 2.0, n)
    g2 = rng.uniform(0.1, 0.9, n)
    exp_ref, exp_trans = _two_stream_oracle(od, g1, g2)
    sdfg = two_stream_reftrans.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    ref = np.zeros(n)
    trans = np.zeros(n)
    sdfg(od=od, g1=g1, g2=g2, ref=ref, trans=trans, NG=n)
    assert np.allclose(ref, exp_ref), 'two-stream reflectance mis-canonicalized'
    assert np.allclose(trans, exp_trans), 'two-stream transmittance mis-canonicalized'


def test_two_stream_reftrans_is_elementwise_map():
    """The per-g-point body is independent; canonicalize must keep it an
    elementwise Map (the transcendental + clamp chain does not serialize
    it)."""
    sdfg = two_stream_reftrans.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    n_maps = sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))
    assert n_maps >= 1, 'the parallel g-point body must keep a Map'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
