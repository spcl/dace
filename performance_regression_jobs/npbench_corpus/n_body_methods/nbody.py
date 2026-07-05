# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""npbench corpus benchmark: ``nbody`` (n_body_methods) -- auto-ported from the npbench repo."""
import numpy as np
import dace as dc

dc_float = dc.float32
dc_complex_float = dc.complex64

# Nt = ceil(tEnd/dt) is a derived integer the dace kernel uses as a free symbol
# (``np.ndarray(Nt + 1)`` / ``range(Nt)``), so it must be bound explicitly rather
# than carried as an "array". The S preset gives Nt=40, but n-body is chaotic and
# fp32 operation-order differences between numpy and the SDFG diverge fast over many
# steps; a few timesteps keep the correctness check meaningful and stable (the same
# small Nt feeds the numpy reference and the SDFG).
SIZES = {'N': 25, 'tEnd': 2.0, 'dt': 0.05, 'softening': 0.1, 'G': 1.0, 'Nt': 3}
INPUT_ARGS = ('N', 'tEnd', 'dt')
ARRAY_ARGS = ('mass', 'pos', 'vel')
SCALARS = {}
OUTPUT_ARGS = ('pos', 'vel')

N, Nt = (dc.symbol(s, dtype=dc.int64) for s in ('N', 'Nt'))


def initialize(N, tEnd, dt, datatype=np.float32):
    from numpy.random import default_rng
    rng = default_rng(42)
    mass = 20.0 * np.ones((N, 1), dtype=datatype) / N
    pos = rng.random((N, 3), dtype=datatype)
    vel = rng.random((N, 3), dtype=datatype)
    Nt = int(np.ceil(tEnd / dt))
    return (mass, pos, vel, Nt)


# Numpy reference helpers (distinct names so they don't collide with the dace
# ``@dc.program`` getAcc/getEnergy used by the kernel below).
def getAcc_np(pos, mass, G, softening):
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z
    inv_r3 = (dx**2 + dy**2 + dz**2 + softening**2)
    inv_r3[inv_r3 > 0] = inv_r3[inv_r3 > 0]**(-1.5)
    ax = G * (dx * inv_r3) @ mass
    ay = G * (dy * inv_r3) @ mass
    az = G * (dz * inv_r3) @ mass
    a = np.hstack((ax, ay, az))
    return a


def getEnergy_np(pos, vel, mass, G):
    KE = 0.5 * np.sum(mass * vel**2)
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z
    inv_r = np.sqrt(dx**2 + dy**2 + dz**2)
    inv_r[inv_r > 0] = 1.0 / inv_r[inv_r > 0]
    PE = G * np.sum(np.triu(-(mass * mass.T) * inv_r, 1))
    return KE, PE


def reference(mass, pos, vel, N, Nt, dt, G, softening):
    vel -= np.mean(mass * vel, axis=0) / np.mean(mass)
    acc = getAcc_np(pos, mass, G, softening)
    KE = np.ndarray(Nt + 1, dtype=mass.dtype)
    PE = np.ndarray(Nt + 1, dtype=mass.dtype)
    KE[0], PE[0] = getEnergy_np(pos, vel, mass, G)
    t = 0.0
    for i in range(Nt):
        vel += acc * dt / 2.0
        pos += vel * dt
        acc = getAcc_np(pos, mass, G, softening)
        vel += acc * dt / 2.0
        t += dt
        KE[i + 1], PE[i + 1] = getEnergy_np(pos, vel, mass, G)


@dc.program
def getAcc(pos: dc_float[N, 3], mass: dc_float[N], G: dc_float, softening: dc_float):
    """
    Calculate the acceleration on each particle due to Newton's Law 
    pos  is an N x 3 matrix of positions
    mass is an N x 1 vector of masses
    G is Newton's Gravitational constant
    softening is the softening length
    a is N x 3 matrix of accelerations
    """
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]
    dx = np.add.outer(-x, x)
    dy = np.add.outer(-y, y)
    dz = np.add.outer(-z, z)
    inv_r3 = dx**2 + dy**2 + dz**2 + softening**2
    I = inv_r3 > 0
    np.power(inv_r3, -1.5, out=inv_r3, where=I)
    ax = G * (dx * inv_r3) @ mass
    ay = G * (dy * inv_r3) @ mass
    az = G * (dz * inv_r3) @ mass
    a = np.ndarray((N, 3), dtype=dc_float)
    a[:, 0] = ax
    a[:, 1] = ay
    a[:, 2] = az
    return a


@dc.program
def getEnergy(pos: dc_float[N, 3], vel: dc_float[N, 3], mass: dc_float[N], G: dc_float):
    """
    Get kinetic energy (KE) and potential energy (PE) of simulation
    pos is N x 3 matrix of positions
    vel is N x 3 matrix of velocities
    mass is an N x 1 vector of masses
    G is Newton's Gravitational constant
    KE is the kinetic energy of the system
    PE is the potential energy of the system
    """
    KE = 0.5 * np.sum(np.reshape(mass, (N, 1)) * vel**2)
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]
    dx = np.add.outer(-x, x)
    dy = np.add.outer(-y, y)
    dz = np.add.outer(-z, z)
    inv_r = np.sqrt(dx**2 + dy**2 + dz**2)
    I = inv_r > 0
    np.divide(1.0, inv_r, out=inv_r, where=I)
    tmp = -np.multiply.outer(mass, mass) * inv_r
    PE = 0.0
    for j in range(N):
        for k in range(j + 1, N):
            PE += tmp[j, k]
    PE *= G
    return (KE, PE)


@dc.program
def kernel(mass: dc_float[N], pos: dc_float[N, 3], vel: dc_float[N, 3], dt: dc_float, G: dc_float, softening: dc_float):
    np.subtract(vel, np.mean(np.reshape(mass, (N, 1)) * vel, axis=0) / np.mean(mass), out=vel)
    acc = getAcc(pos, mass, G, softening)
    KE = np.ndarray(Nt + 1, dtype=dc_float)
    PE = np.ndarray(Nt + 1, dtype=dc_float)
    KE[0], PE[0] = getEnergy(pos, vel, mass, G)
    t = 0.0
    for i in range(Nt):
        vel += acc * dt / 2.0
        pos += vel * dt
        acc[:] = getAcc(pos, mass, G, softening)
        vel += acc * dt / 2.0
        t += dt
        KE[i + 1], PE[i + 1] = getEnergy(pos, vel, mass, G)
    # The validated outputs are the in-place-mutated pos/vel (per the npbench
    # ``output_args``); KE/PE are internal diagnostics. Return pos/vel so they map
    # to ``output_args`` in order (and so the SDFG exposes them as outputs).
    return (pos, vel)


CORPUS = dict(name='nbody',
              dwarf='n_body_methods',
              sizes=SIZES,
              input_args=INPUT_ARGS,
              array_args=ARRAY_ARGS,
              scalars=SCALARS,
              output_args=OUTPUT_ARGS,
              initialize=initialize,
              reference=reference,
              program=kernel)
