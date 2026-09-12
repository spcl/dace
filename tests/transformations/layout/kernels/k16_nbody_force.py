# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""k16 N-body pairwise force -- the layout+tiling reuse witness (AoS/SoA over an O(N^2) loop).

The gravitational all-pairs force: each body ``i`` accumulates over every other body ``j``::

    acc[i]  = sum_j  m[j] * (pos[j] - pos[i]) / (|pos[j]-pos[i]|^2 + soft)^(3/2)
    vel'[i] = vel[i] + DT * acc[i]
    pos'[i] = pos[i] + DT * vel'[i]

Unlike k09 (a streaming per-particle update), the inner ``j`` loop reads the WHOLE ``pos`` array for
every ``i``, so the storage orientation of ``pos`` is the layout decision for this all-pairs reduction:

    AoS   (N, 3) : body-major, the three coords of a body are contiguous   (identity)
    SoA   (3, N) : coord-major, one contiguous vector per axis             (transparent Permute)

Every candidate reads ``pos[k, d]`` at the same logical index, so the layout is transparent and each
reproduces the SoA oracle; the sweep only picks the physical order. ``soft`` avoids the ``i==j``
singularity. Source: npbench ``nbody`` (all-pairs ``getAcc`` kernel); Cabana AoS/SoA (Slattery et al.,
JOSS'22); SC26 layout paper (Permute over an O(N^2) reduction).
"""
import numpy
import dace

from dace.transformation.layout.brute_force import permutation_candidates

N = dace.symbol("N")

DT = 1e-2
SOFT = 1e-3  # softening: (r^2 + soft)^(3/2) so the i==j self-term stays finite


@dace.program
def nbody_force(pos: dace.float64[N, 3], vel: dace.float64[N, 3], mass: dace.float64[N], out_vel: dace.float64[N, 3],
                out_pos: dace.float64[N, 3]):
    """One force step. The (i, j) pairwise accumulation is a WCR sum-reduction over j (both are map
    dims so ``pos``'s Permute stays transparent, k10-style); then a per-body drift/kick update."""
    acc = numpy.zeros((N, 3), dace.float64)
    for i, j in dace.map[0:N, 0:N] @ dace.ScheduleType.Sequential:
        dx = pos[j, 0] - pos[i, 0]
        dy = pos[j, 1] - pos[i, 1]
        dz = pos[j, 2] - pos[i, 2]
        inv = 1.0 / (dx * dx + dy * dy + dz * dz + SOFT)**1.5  # 1 / r^3
        acc[i, 0] += mass[j] * dx * inv
        acc[i, 1] += mass[j] * dy * inv
        acc[i, 2] += mass[j] * dz * inv
    for i in dace.map[0:N] @ dace.ScheduleType.Sequential:
        out_vel[i, 0] = vel[i, 0] + DT * acc[i, 0]
        out_vel[i, 1] = vel[i, 1] + DT * acc[i, 1]
        out_vel[i, 2] = vel[i, 2] + DT * acc[i, 2]
        out_pos[i, 0] = pos[i, 0] + DT * out_vel[i, 0]
        out_pos[i, 1] = pos[i, 1] + DT * out_vel[i, 1]
        out_pos[i, 2] = pos[i, 2] + DT * out_vel[i, 2]


def oracle(pos, vel, mass):
    diff = pos[None, :, :] - pos[:, None, :]  # diff[i, j] = pos[j] - pos[i]
    inv = (numpy.square(diff).sum(-1) + SOFT)**-1.5  # (N, N) = 1 / r^3
    acc = (mass[None, :, None] * diff * inv[:, :, None]).sum(1)  # sum over j
    out_vel = vel + DT * acc
    out_pos = pos + DT * out_vel
    return {"out_vel": out_vel, "out_pos": out_pos}


def make_inputs(n, seed=0):
    rng = numpy.random.default_rng(seed)
    return {"pos": rng.random((n, 3)), "vel": rng.random((n, 3)), "mass": rng.random(n) + 0.5}


def candidates():
    """AoS (N,3) identity vs SoA (3,N): one transparent Permute of ``pos``'s two axes."""
    return dict(permutation_candidates("pos", 2))  # aos: permute_pos_01, soa: permute_pos_10


def run_closure(inputs, n):
    """The Permute is transparent, so inputs bind as-is; fresh zeroed outputs each call."""

    def run(sdfg):
        out_vel = numpy.zeros((n, 3))
        out_pos = numpy.zeros((n, 3))
        sdfg(pos=inputs["pos"].copy(),
             vel=inputs["vel"].copy(),
             mass=inputs["mass"].copy(),
             out_vel=out_vel,
             out_pos=out_pos,
             N=n)
        return {"out_vel": out_vel, "out_pos": out_pos}

    return run
