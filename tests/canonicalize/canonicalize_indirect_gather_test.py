# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Canonicalize on indirect-gather + twin-reduction shapes (ICON dycore).

Distilled from ``mo_solve_nonhydro.f90`` / ``mo_velocity_advection.f90``:

* **Indirect 3-neighbor gather** -- ``z_ekinh[jc] = sum_k w[jc,k] *
  field[idx[jc,k]]`` (cell-to-edge interpolation via an index table
  ``ieidx``). The outer ``jc`` axis is parallel; each lane reads three
  table-selected elements. Canonicalize must keep the ``jc`` Map and the
  per-element indirect read; value-preserving.

* **Twin reduction sharing one stencil** -- ``mo_solve_nonhydro.f90:2162``
  (``z_flxdiv_mass`` and ``z_flxdiv_theta`` both = a 3-edge gather sum over
  the SAME index table). Two accumulators, identical neighbour stencil --
  a fusion candidate (the two should co-locate under one ``jc`` Map).

Both contracts are primarily numerical (value-preserving vs the original
SDFG); structural assertions pin the parallel ``jc`` Map and, for the twin
case, that the two writes share a Map scope.
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')  # number of cells (parallel)


def _nmaps(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))


# ----------------------------------------------------------------------
# Indirect 3-neighbor gather (cell <- 3 edges via an index table)
# ----------------------------------------------------------------------


@dace.program
def indirect_gather_3nbr(field: dace.float64[N], idx: dace.int32[N, 3], w: dace.float64[N, 3],
                         out: dace.float64[N]):
    """``out[jc] = sum_k w[jc, k] * field[idx[jc, k]]`` -- 3-neighbor
    gather via an index table. Parallel over ``jc``."""
    for jc in dace.map[0:N]:
        out[jc] = (w[jc, 0] * field[idx[jc, 0]] + w[jc, 1] * field[idx[jc, 1]] + w[jc, 2] * field[idx[jc, 2]])


def _gather_oracle(field, idx, w):
    n = field.shape[0]
    out = np.zeros(n)
    for jc in range(n):
        out[jc] = sum(w[jc, k] * field[idx[jc, k]] for k in range(3))
    return out


def _random_index_table(n, rng):
    return rng.integers(0, n, size=(n, 3)).astype(np.int32)


def test_indirect_gather_3nbr_value_preserving():
    n = 12
    rng = np.random.default_rng(40)
    field = rng.standard_normal(n)
    idx = _random_index_table(n, rng)
    w = rng.standard_normal((n, 3))
    exp = _gather_oracle(field, idx, w)
    sdfg = indirect_gather_3nbr.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    out = np.zeros(n)
    sdfg(field=field, idx=idx, w=w, out=out, N=n)
    assert np.allclose(out, exp), 'indirect gather mis-canonicalized'


def test_indirect_gather_3nbr_keeps_parallel_map():
    """The per-cell gather is independent; the ``jc`` axis must be a Map
    (the indirect ``field[idx[jc,k]]`` read does not serialize it)."""
    sdfg = indirect_gather_3nbr.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    assert _nmaps(sdfg) >= 1, 'the parallel jc gather must keep a Map'


# ----------------------------------------------------------------------
# Twin reduction sharing one neighbour stencil
# ----------------------------------------------------------------------


@dace.program
def twin_reduction_shared_stencil(mass_fl: dace.float64[N], theta_fl: dace.float64[N], idx: dace.int32[N, 3],
                                  gfac: dace.float64[N, 3], div_mass: dace.float64[N], div_theta: dace.float64[N]):
    """Two accumulators over the SAME 3-edge index-table stencil
    (``mo_solve_nonhydro.f90`` flux-divergence). Both are parallel over
    ``jc`` and share ``idx`` / ``gfac`` -- a fusion candidate."""
    for jc in dace.map[0:N]:
        div_mass[jc] = (mass_fl[idx[jc, 0]] * gfac[jc, 0] + mass_fl[idx[jc, 1]] * gfac[jc, 1] +
                        mass_fl[idx[jc, 2]] * gfac[jc, 2])
        div_theta[jc] = (theta_fl[idx[jc, 0]] * gfac[jc, 0] + theta_fl[idx[jc, 1]] * gfac[jc, 1] +
                         theta_fl[idx[jc, 2]] * gfac[jc, 2])


def _twin_oracle(mass_fl, theta_fl, idx, gfac):
    n = mass_fl.shape[0]
    dm = np.zeros(n)
    dt = np.zeros(n)
    for jc in range(n):
        dm[jc] = sum(mass_fl[idx[jc, k]] * gfac[jc, k] for k in range(3))
        dt[jc] = sum(theta_fl[idx[jc, k]] * gfac[jc, k] for k in range(3))
    return dm, dt


def test_twin_reduction_value_preserving():
    n = 10
    rng = np.random.default_rng(41)
    mass_fl = rng.standard_normal(n)
    theta_fl = rng.standard_normal(n)
    idx = _random_index_table(n, rng)
    gfac = rng.standard_normal((n, 3))
    exp_m, exp_t = _twin_oracle(mass_fl, theta_fl, idx, gfac)
    sdfg = twin_reduction_shared_stencil.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    dm = np.zeros(n)
    dt = np.zeros(n)
    sdfg(mass_fl=mass_fl, theta_fl=theta_fl, idx=idx, gfac=gfac, div_mass=dm, div_theta=dt, N=n)
    assert np.allclose(dm, exp_m), 'div_mass mis-canonicalized'
    assert np.allclose(dt, exp_t), 'div_theta mis-canonicalized'


def test_twin_reduction_fuses_to_one_map():
    """The two accumulators share the ``jc`` iteration space and the same
    index-table stencil; they must co-locate / fuse into a single ``jc``
    Map (not two separate map nests)."""
    sdfg = twin_reduction_shared_stencil.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    assert _nmaps(sdfg) == 1, f'twin reduction should fuse to a single jc map, got {_nmaps(sdfg)}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
