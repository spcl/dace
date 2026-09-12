# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""AoSoA layout (Zip ∘ Block) on particle-style kernels.

Particle fields start fully unzipped (SoA): separate arrays ``x, y, vx, vy`` of shape ``[N]``. The
AoSoA layout blocks the particle axis by a vector width ``V`` and interleaves the fields at ``V``
granularity, giving ONE array ``P`` of shape ``[N/V, F, V]`` -- particle ``i``, field ``f`` at
``P[i//V, f, i%V]``. The kernels are run bit-exact against a numpy SoA oracle, with the input packed
into the AoSoA array and the output unpacked back to per-field vectors.
"""
import numpy
import dace

from dace.transformation.layout.zip_arrays import aosoa_layout

N = dace.symbol("N")


@dace.program
def leapfrog(x: dace.float64[N], y: dace.float64[N], vx: dace.float64[N], vy: dace.float64[N]):
    """One symplectic-Euler position update per particle (in-place on x, y)."""
    for i in dace.map[0:N] @ dace.ScheduleType.Sequential:
        x[i] = x[i] + 0.5 * vx[i]
        y[i] = y[i] + 0.5 * vy[i]


@dace.program
def gravity_kick(px: dace.float64[N], py: dace.float64[N], pz: dace.float64[N]):
    """Read three coordinate fields, write one (pz) -- a mixed read/write AoSoA access."""
    for i in dace.map[0:N] @ dace.ScheduleType.Sequential:
        pz[i] = pz[i] - 0.1 * (px[i] * px[i] + py[i] * py[i])


def _pack_aosoa(fields, vector_width):
    """Pack a list of ``[N]`` numpy vectors into an AoSoA array ``[N/V, F, V]``."""
    n = fields[0].shape[0]
    v = vector_width
    chunks = n // v
    P = numpy.empty((chunks, len(fields), v), dtype=fields[0].dtype)
    for f, arr in enumerate(fields):
        P[:, f, :] = arr.reshape(chunks, v)
    return P


def _unpack_field(P, f, n, vector_width):
    """Extract field ``f`` from an AoSoA array back to a ``[N]`` vector."""
    return P[:, f, :].reshape(n)


def test_aosoa_leapfrog_bitexact():
    sdfg = leapfrog.to_sdfg()
    sdfg.name = "leapfrog_aosoa"
    V, _N, F = 8, 32, 4
    aosoa_layout(sdfg, "P", ["x", "y", "vx", "vy"], V)
    sdfg.validate()

    assert all(f not in sdfg.arrays for f in ("x", "y", "vx", "vy"))
    ps = sdfg.arrays["P"].shape  # outer dim stays symbolic int_ceil(N, V)
    assert len(ps) == 3 and int(ps[1]) == F and int(ps[2]) == V
    assert int(dace.symbolic.evaluate(ps[0], {N: _N})) == _N // V

    x = numpy.random.rand(_N)
    y = numpy.random.rand(_N)
    vx = numpy.random.rand(_N)
    vy = numpy.random.rand(_N)
    x_ref = x + 0.5 * vx
    y_ref = y + 0.5 * vy

    P = _pack_aosoa([x, y, vx, vy], V)  # field order matches aosoa_layout arg order
    sdfg(P=P, N=_N)

    assert numpy.allclose(_unpack_field(P, 0, _N, V), x_ref)
    assert numpy.allclose(_unpack_field(P, 1, _N, V), y_ref)
    # velocities untouched
    assert numpy.allclose(_unpack_field(P, 2, _N, V), vx)
    assert numpy.allclose(_unpack_field(P, 3, _N, V), vy)


def test_aosoa_gravity_kick_bitexact():
    sdfg = gravity_kick.to_sdfg()
    sdfg.name = "gravity_aosoa"
    V, _N, F = 4, 24, 3
    aosoa_layout(sdfg, "P", ["px", "py", "pz"], V)
    sdfg.validate()

    ps = sdfg.arrays["P"].shape
    assert len(ps) == 3 and int(ps[1]) == F and int(ps[2]) == V
    assert int(dace.symbolic.evaluate(ps[0], {N: _N})) == _N // V

    px = numpy.random.rand(_N)
    py = numpy.random.rand(_N)
    pz = numpy.random.rand(_N)
    pz_ref = pz - 0.1 * (px * px + py * py)

    P = _pack_aosoa([px, py, pz], V)
    sdfg(P=P, N=_N)

    assert numpy.allclose(_unpack_field(P, 2, _N, V), pz_ref)
    assert numpy.allclose(_unpack_field(P, 0, _N, V), px)
    assert numpy.allclose(_unpack_field(P, 1, _N, V), py)


if __name__ == "__main__":
    test_aosoa_leapfrog_bitexact()
    test_aosoa_gravity_kick_bitexact()
    print("AoSoA particle tests PASS")
