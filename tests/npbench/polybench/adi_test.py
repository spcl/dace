# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench
import dace.dtypes
import numpy as np
import dace
import pytest
import argparse
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.autodiff import add_backward_pass

# Dataset sizes
# TSTEPS, N
sizes = {"mini": (20, 20), "small": (40, 60), "medium": (100, 200), "large": (500, 1000), "extra-large": (1000, 2000)}

N = dace.symbol('N', dtype=dace.int64)


def relerror(val, ref):
    if np.linalg.norm(ref) == 0:
        return np.linalg.norm(val - ref)
    return np.linalg.norm(val - ref) / np.linalg.norm(ref)


def numpy_kernel(TSTEPS, N, u):

    v = np.empty(u.shape, dtype=u.dtype)
    p = np.empty(u.shape, dtype=u.dtype)
    q = np.empty(u.shape, dtype=u.dtype)

    DX = 1.0 / N
    DY = 1.0 / N
    DT = 1.0 / TSTEPS
    B1 = 2.0
    B2 = 1.0
    mul1 = B1 * DT / (DX * DX)
    mul2 = B2 * DT / (DY * DY)

    a = -mul1 / 2.0
    b = 1.0 + mul2
    c = a
    d = -mul2 / 2.0
    e = 1.0 + mul2
    f = d

    for t in range(0, TSTEPS):
        v[0, 1:N - 1] = 1.0
        p[1:N - 1, 0] = 0.0
        q[1:N - 1, 0] = v[0, 1:N - 1]
        for j in range(1, N - 1):
            p[1:N - 1, j] = -c / (a * p[1:N - 1, j - 1] + b)
            q[1:N - 1, j] = (-d * u[j, 0:N - 2] + (1.0 + 2.0 * d) * u[j, 1:N - 1] - f * u[j, 2:N] -
                             a * q[1:N - 1, j - 1]) / (a * p[1:N - 1, j - 1] + b)
        v[N - 1, 1:N - 1] = 1.0
        for j in range(N - 2, 0, -1):
            v[j, 1:N - 1] = p[1:N - 1, j] * v[j + 1, 1:N - 1] + q[1:N - 1, j]

        u[1:N - 1, 0] = 1.0
        p[1:N - 1, 0] = 0.0
        q[1:N - 1, 0] = u[1:N - 1, 0]
        for j in range(1, N - 1):
            p[1:N - 1, j] = -f / (d * p[1:N - 1, j - 1] + e)
            q[1:N - 1, j] = (-a * v[0:N - 2, j] + (1.0 + 2.0 * a) * v[1:N - 1, j] - c * v[2:N, j] -
                             d * q[1:N - 1, j - 1]) / (d * p[1:N - 1, j - 1] + e)
        u[1:N - 1, N - 1] = 1.0
        for j in range(N - 2, 0, -1):
            u[1:N - 1, j] = p[1:N - 1, j] * u[1:N - 1, j + 1] + q[1:N - 1, j]


@dace.program
def adi_kernel(TSTEPS: dace.int64, u: dace.float64[N, N]):

    v = np.empty(u.shape, dtype=u.dtype)
    p = np.empty(u.shape, dtype=u.dtype)
    q = np.empty(u.shape, dtype=u.dtype)

    DX = 1.0 / np.float64(N)
    DY = 1.0 / np.float64(N)
    DT = 1.0 / np.float64(TSTEPS)
    B1 = 2.0
    B2 = 1.0
    mul1 = B1 * DT / (DX * DX)
    mul2 = B2 * DT / (DY * DY)

    a = -mul1 / 2.0
    b = 1.0 + mul2
    c = a
    d = -mul2 / 2.0
    e = 1.0 + mul2
    f = d

    for t in range(0, TSTEPS):
        v[0, 1:N - 1] = 1.0
        p[1:N - 1, 0] = 0.0
        q[1:N - 1, 0] = v[0, 1:N - 1]
        for j in range(1, N - 1):
            p[1:N - 1, j] = -c / (a * p[1:N - 1, j - 1] + b)
            q[1:N - 1, j] = (-d * u[j, 0:N - 2] + (1.0 + 2.0 * d) * u[j, 1:N - 1] - f * u[j, 2:N] -
                             a * q[1:N - 1, j - 1]) / (a * p[1:N - 1, j - 1] + b)
        v[N - 1, 1:N - 1] = 1.0
        for j in range(N - 2, 0, -1):
            v[j, 1:N - 1] = p[1:N - 1, j] * v[j + 1, 1:N - 1] + q[1:N - 1, j]

        u[1:N - 1, 0] = 1.0
        p[1:N - 1, 0] = 0.0
        q[1:N - 1, 0] = u[1:N - 1, 0]
        for j in range(1, N - 1):
            p[1:N - 1, j] = -f / (d * p[1:N - 1, j - 1] + e)
            q[1:N - 1, j] = (-a * v[0:N - 2, j] + (1.0 + 2.0 * a) * v[1:N - 1, j] - c * v[2:N, j] -
                             d * q[1:N - 1, j - 1]) / (d * p[1:N - 1, j - 1] + e)
        u[1:N - 1, N - 1] = 1.0
        for j in range(N - 2, 0, -1):
            u[1:N - 1, j] = p[1:N - 1, j] * u[1:N - 1, j + 1] + q[1:N - 1, j]


def initialize(N, datatype=np.float64):
    u = np.fromfunction(lambda i, j: (i + N - j) / N, (N, N), dtype=datatype)
    return u


def adi_jax_kernel(jnp, lax, TSTEPS, u):
    N = u.shape[0]
    v = jnp.zeros_like(u)
    p = jnp.zeros_like(u)
    q = jnp.zeros_like(u)

    DX = 1.0 / N
    DY = 1.0 / N
    DT = 1.0 / TSTEPS
    B1 = 2.0
    B2 = 1.0
    mul1 = B1 * DT / (DX * DX)
    mul2 = B2 * DT / (DY * DY)
    a = -mul1 / 2.0
    b = 1.0 + mul2
    c = a
    d = -mul2 / 2.0
    e = 1.0 + mul2
    f = d

    def first_j_scan(carry, j):
        p, q, u = carry

        p = p.at[1:N - 1, j].set(-c / (a * p[1:N - 1, j - 1] + b))
        q = q.at[1:N - 1,
                 j].set((-d * u[j, 0:N - 2] + (1.0 + 2.0 * d) * u[j, 1:N - 1] - f * u[j, 2:N] - a * q[1:N - 1, j - 1]) /
                        (a * p[1:N - 1, j - 1] + b))
        return (p, q, u), None

    def first_backward_j_scan(carry, j):
        v, p, q = carry
        idx = N - 2 - j  # reverse order index: when j=0, idx = N-2; when j=N-2, idx = 0.
        v = v.at[idx, 1:N - 1].set(p[1:N - 1, idx] * v[idx + 1, 1:N - 1] + q[1:N - 1, idx])
        return (v, p, q), None

    def second_j_scan(carry, j):
        p, q, v = carry
        p = p.at[1:N - 1, j].set(-f / (d * p[1:N - 1, j - 1] + e))
        q = q.at[1:N - 1,
                 j].set((-a * v[0:N - 2, j] + (1.0 + 2.0 * a) * v[1:N - 1, j] - c * v[2:N, j] - d * q[1:N - 1, j - 1]) /
                        (d * p[1:N - 1, j - 1] + e))
        return (p, q, v), None

    def second_backward_j_scan(carry, j):
        u, p, q = carry
        idx = N - 2 - j
        u = u.at[1:N - 1, idx].set(p[1:N - 1, idx] * u[1:N - 1, idx + 1] + q[1:N - 1, idx])
        return (u, p, q), None

    def time_step_body(carry, t):
        u, v, p, q = carry

        v = v.at[0, 1:N - 1].set(1.0)
        p = p.at[1:N - 1, 0].set(0.0)
        q = q.at[1:N - 1, 0].set(v[0, 1:N - 1])
        (p, q, u), _ = lax.scan(first_j_scan, (p, q, u), jnp.arange(1, N - 1))

        v = v.at[N - 1, 1:N - 1].set(1.0)

        (v, p, q), _ = lax.scan(first_backward_j_scan, (v, p, q), jnp.arange(0, N - 2))

        u = u.at[1:N - 1, 0].set(1.0)
        p = p.at[1:N - 1, 0].set(0.0)
        q = q.at[1:N - 1, 0].set(u[1:N - 1, 0])
        (p, q, v), _ = lax.scan(second_j_scan, (p, q, v), jnp.arange(1, N - 1))
        u = u.at[1:N - 1, N - 1].set(1.0)
        (u, p, q), _ = lax.scan(second_backward_j_scan, (u, p, q), jnp.arange(0, N - 2))

        return (u, v, p, q), None

    (u, v, p, q), _ = lax.scan(time_step_body, (u, v, p, q), jnp.arange(0, TSTEPS))
    return jnp.sum(u)


def run_adi(device_type: dace.dtypes.DeviceType):
    '''
    Runs ADI for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench small)
    TSTEPS, N = sizes["small"]
    u = initialize(N)
    dace_u = u.copy()

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = adi_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        sdfg(TSTEPS=TSTEPS, u=dace_u, N=N)
    else:
        raise ValueError(f"Unsupported device type: {device_type}")

    # Compute ground truth and Validate result
    numpy_kernel(TSTEPS, N, u)
    assert (np.allclose(dace_u, u) or relerror(dace_u, u) < 1e-10)
    return sdfg


def run_adi_autodiff():
    import jax
    import jax.numpy as jnp
    import jax.lax as lax

    # Initialize data (polybench mini size for smaller problem)
    _, N = sizes["mini"]

    # Use smaller number of timesteps to avoid exploding gradients
    TSTEPS = 10

    u = initialize(N)

    # Initialize gradient computation data
    gradient_u = np.zeros_like(u)
    gradient___return = np.ones((1, ), dtype=np.float64)

    # Define sum reduction for the output
    @dace.program
    def autodiff_kernel(TSTEPS: dace.int64, u: dace.float64[N, N]):
        adi_kernel(TSTEPS, u)
        return np.sum(u)

    # Add the backward pass to the SDFG
    sdfg = autodiff_kernel.to_sdfg()
    add_backward_pass(sdfg=sdfg, inputs=["u"], outputs=["__return"])
    sdfg(TSTEPS, u, N=N, gradient_u=gradient_u, gradient___return=gradient___return)

    # Enable float64 support
    jax.config.update("jax_enable_x64", True)

    # Numerically validate vs JAX
    jax_kernel = lambda TSTEPS, u: adi_jax_kernel(jnp, lax, TSTEPS, u)
    jax_grad = jax.jit(jax.grad(jax_kernel, argnums=1), static_argnums=0)
    u_jax = np.copy(initialize(N))
    jax_grad_u = jax_grad(TSTEPS, u_jax)

    np.testing.assert_allclose(gradient_u, jax_grad_u, rtol=1e-6)


def test_cpu():
    run_adi(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_adi(dace.dtypes.DeviceType.GPU)


@pytest.mark.autodiff
def test_autodiff():
    pytest.importorskip("jax", reason="jax not installed. Please install with: pip install dace[ml-testing]")
    run_adi_autodiff()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_adi(dace.dtypes.DeviceType.CPU)
        run_adi_autodiff()
    elif target == "gpu":
        run_adi(dace.dtypes.DeviceType.GPU)
