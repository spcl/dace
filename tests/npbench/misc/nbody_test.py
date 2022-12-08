# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench
import dace.dtypes
import numpy as np
import dace as dc
import pytest
import argparse
from dace.fpga_testing import fpga_test, xilinx_test
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from dace.transformation.dataflow import StreamingMemory, StreamingComposition
from dace.transformation.auto.auto_optimize import auto_optimize, fpga_auto_opt
from dace.config import set_temporary

N, Nt = (dc.symbol(s, dtype=dc.int64) for s in ('N', 'Nt'))


@dc.program
def getAcc(pos: dc.float64[N, 3], mass: dc.float64[N], G: dc.float64, softening: dc.float64):
    """
    Calculate the acceleration on each particle due to Newton's Law 
    pos  is an N x 3 matrix of positions
    mass is an N x 1 vector of masses
    G is Newton's Gravitational constant
    softening is the softening length
    a is N x 3 matrix of accelerations
    """
    # positions r = [x,y,z] for all particles
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]

    # matrix that stores all pairwise particle separations: r_j - r_i
    dx = np.add.outer(-x, x)
    dy = np.add.outer(-y, y)
    dz = np.add.outer(-z, z)

    # matrix that stores 1/r^3 for all particle pairwise particle separations
    inv_r3 = (dx**2 + dy**2 + dz**2 + softening**2)
    # inv_r3[inv_r3>0] = inv_r3[inv_r3>0]**(-1.5)
    I = inv_r3 > 0
    np.power(inv_r3, -1.5, out=inv_r3, where=I)

    ax = G * (dx * inv_r3) @ mass
    ay = G * (dy * inv_r3) @ mass
    az = G * (dz * inv_r3) @ mass

    # pack together the acceleration components
    a = np.ndarray((N, 3), dtype=np.float64)
    a[:, 0] = ax
    a[:, 1] = ay
    a[:, 2] = az

    return a


@dc.program
def getEnergy(pos: dc.float64[N, 3], vel: dc.float64[N, 3], mass: dc.float64[N], G: dc.float64):
    """
    Get kinetic energy (KE) and potential energy (PE) of simulation
    pos is N x 3 matrix of positions
    vel is N x 3 matrix of velocities
    mass is an N x 1 vector of masses
    G is Newton's Gravitational constant
    KE is the kinetic energy of the system
    PE is the potential energy of the system
    """
    # Kinetic Energy:
    # KE = 0.5 * np.sum(np.sum( mass * vel**2 ))
    # KE = 0.5 * np.sum( mass * vel**2 )
    KE = 0.5 * np.sum(np.reshape(mass, (N, 1)) * vel**2)

    # Potential Energy:

    # positions r = [x,y,z] for all particles
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]

    # matrix that stores all pairwise particle separations: r_j - r_i
    dx = np.add.outer(-x, x)
    dy = np.add.outer(-y, y)
    dz = np.add.outer(-z, z)

    # matrix that stores 1/r for all particle pairwise particle separations
    inv_r = np.sqrt(dx**2 + dy**2 + dz**2)
    # inv_r[inv_r>0] = 1.0/inv_r[inv_r>0]
    I = inv_r > 0
    np.divide(1.0, inv_r, out=inv_r, where=I)

    # sum over upper triangle, to count each interaction only once
    tmp = -np.multiply.outer(mass, mass) * inv_r
    PE = 0.0
    for j in range(N):
        for k in range(j + 1, N):
            PE += tmp[j, k]
    PE *= G

    return KE, PE


@dc.program
def nbody(mass: dc.float64[N], pos: dc.float64[N, 3], vel: dc.float64[N, 3], dt: dc.float64, G: dc.float64,
          softening: dc.float64):

    # Convert to Center-of-Mass frame
    np.subtract(vel, np.mean(np.reshape(mass, (N, 1)) * vel, axis=0) / np.mean(mass), out=vel)

    # calculate initial gravitational accelerations
    acc = getAcc(pos, mass, G, softening)

    # calculate initial energy of system
    KE = np.ndarray(Nt + 1, dtype=np.float64)
    PE = np.ndarray(Nt + 1, dtype=np.float64)
    KE[0], PE[0] = getEnergy(pos, vel, mass, G)

    t = 0.0

    # Simulation Main Loop
    for i in range(Nt):
        # (1/2) kick
        vel += acc * dt / 2.0

        # drift
        pos += vel * dt

        # update accelerations
        acc[:] = getAcc(pos, mass, G, softening)

        # (1/2) kick
        vel += acc * dt / 2.0

        # update time
        t += dt

        # get energy of system
        KE[i + 1], PE[i + 1] = getEnergy(pos, vel, mass, G)

    return KE, PE


def initialize(N, tEnd, dt):
    from numpy.random import default_rng
    rng = default_rng(42)
    mass = 20.0 * np.ones((N, 1)) / N  # total mass of particles is 20
    pos = rng.random((N, 3))  # randomly selected positions and velocities
    vel = rng.random((N, 3))
    Nt = int(np.ceil(tEnd / dt))
    return mass, pos, vel, Nt


### Ground Truth


def getAcc_np(pos, mass, G, softening):
    """
    Calculate the acceleration on each particle due to Newton's Law 
    pos  is an N x 3 matrix of positions
    mass is an N x 1 vector of masses
    G is Newton's Gravitational constant
    softening is the softening length
    a is N x 3 matrix of accelerations
    """
    # positions r = [x,y,z] for all particles
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]

    # matrix that stores all pairwise particle separations: r_j - r_i
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    # matrix that stores 1/r^3 for all particle pairwise particle separations
    inv_r3 = (dx**2 + dy**2 + dz**2 + softening**2)
    inv_r3[inv_r3 > 0] = inv_r3[inv_r3 > 0]**(-1.5)

    ax = G * (dx * inv_r3) @ mass
    ay = G * (dy * inv_r3) @ mass
    az = G * (dz * inv_r3) @ mass

    # pack together the acceleration components
    a = np.hstack((ax, ay, az))

    return a


def getEnergy_np(pos, vel, mass, G):
    """
    Get kinetic energy (KE) and potential energy (PE) of simulation
    pos is N x 3 matrix of positions
    vel is N x 3 matrix of velocities
    mass is an N x 1 vector of masses
    G is Newton's Gravitational constant
    KE is the kinetic energy of the system
    PE is the potential energy of the system
    """
    # Kinetic Energy:
    # KE = 0.5 * np.sum(np.sum( mass * vel**2 ))
    KE = 0.5 * np.sum(mass * vel**2)

    # Potential Energy:

    # positions r = [x,y,z] for all particles
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]

    # matrix that stores all pairwise particle separations: r_j - r_i
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    # matrix that stores 1/r for all particle pairwise particle separations
    inv_r = np.sqrt(dx**2 + dy**2 + dz**2)
    inv_r[inv_r > 0] = 1.0 / inv_r[inv_r > 0]

    # sum over upper triangle, to count each interaction only once
    # PE = G * np.sum(np.sum(np.triu(-(mass*mass.T)*inv_r,1)))
    PE = G * np.sum(np.triu(-(mass * mass.T) * inv_r, 1))

    return KE, PE


def nbody_np(mass, pos, vel, N, Nt, dt, G, softening):

    # Convert to Center-of-Mass frame
    vel -= np.mean(mass * vel, axis=0) / np.mean(mass)

    # calculate initial gravitational accelerations
    acc = getAcc_np(pos, mass, G, softening)

    # calculate initial energy of system
    KE = np.ndarray(Nt + 1, dtype=np.float64)
    PE = np.ndarray(Nt + 1, dtype=np.float64)
    KE[0], PE[0] = getEnergy_np(pos, vel, mass, G)

    t = 0.0

    # Simulation Main Loop
    for i in range(Nt):
        # (1/2) kick
        vel += acc * dt / 2.0

        # drift
        pos += vel * dt

        # update accelerations
        acc = getAcc_np(pos, mass, G, softening)

        # (1/2) kick
        vel += acc * dt / 2.0

        # update time
        t += dt

        # get energy of system
        KE[i + 1], PE[i + 1] = getEnergy_np(pos, vel, mass, G)

    return KE, PE


def run_nbody(device_type: dace.dtypes.DeviceType):
    '''
    Runs nbody for the given device
    :return: the SDFG
    '''

    # Initialize data (npbench small size)
    N, tEnd, dt, softening, G = 25, 2.0, 0.05, 0.1, 1.0
    mass, pos, vel, Nt = initialize(N, tEnd, dt)
    mass_ref = np.copy(mass)
    pos_ref = np.copy(pos)
    vel_ref = np.copy(vel)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = nbody.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        KE, PE = sdfg(mass, pos, vel, dt, G, softening, N=N, Nt=Nt)
    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = nbody.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        from dace.libraries.standard import Reduce
        Reduce.default_implementation = "FPGAPartialReduction"
        from dace.libraries.blas import Gemv
        Gemv.default_implementation = "FPGA_Accumulate"
        sdfg.expand_library_nodes()

        sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
        sdfg.specialize(dict(N=N, Nt=Nt))
        KE, PE = sdfg(mass, pos, vel, dt, G, softening)

    # Compute ground truth and validate
    KE_ref, PE_ref = nbody_np(mass_ref, pos_ref, vel_ref, N, Nt, dt, G, softening)
    assert np.allclose(KE, KE_ref)
    assert np.allclose(PE, PE_ref)
    return sdfg


def test_cpu():
    run_nbody(dace.dtypes.DeviceType.CPU)


@pytest.mark.skip(reason="Compiler error")
@pytest.mark.gpu
def test_gpu():
    run_nbody(dace.dtypes.DeviceType.GPU)


@pytest.mark.skip(reason="Xilinx validation error, Intel argument overflow")
@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_nbody(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_nbody(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_nbody(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_nbody(dace.dtypes.DeviceType.FPGA)
