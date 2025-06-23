import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass

N = dc.symbol('N', dtype=dc.int64)


@dc.program
def kernel(TSTEPS: dc.int64, u: dc.float64[N, N], S: dc.float64[1]):

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
        for j_1 in range(1, N - 1):
            p[1:N - 1, j_1] = -c / (a * p[1:N - 1, j_1 - 1] + b)
            q[1:N - 1, j_1] = (-d * u[j_1, 0:N - 2] + (1.0 + 2.0 * d) * u[j_1, 1:N - 1] - f * u[j_1, 2:N] -
                               a * q[1:N - 1, j_1 - 1]) / (a * p[1:N - 1, j_1 - 1] + b)
        v[N - 1, 1:N - 1] = 1.0
        for j_2 in range(N - 2, 0, -1):
            v[j_2, 1:N - 1] = p[1:N - 1, j_2] * v[j_2 + 1, 1:N - 1] + q[1:N - 1, j_2]

        u[1:N - 1, 0] = 1.0
        p[1:N - 1, 0] = 0.0
        q[1:N - 1, 0] = u[1:N - 1, 0]
        for j_3 in range(1, N - 1):
            p[1:N - 1, j_3] = -f / (d * p[1:N - 1, j_3 - 1] + e)
            q[1:N - 1, j_3] = (-a * v[0:N - 2, j_3] + (1.0 + 2.0 * a) * v[1:N - 1, j_3] - c * v[2:N, j_3] -
                               d * q[1:N - 1, j_3 - 1]) / (d * p[1:N - 1, j_3 - 1] + e)
        u[1:N - 1, N - 1] = 1.0
        for j_4 in range(N - 2, 0, -1):
            u[1:N - 1, j_4] = p[1:N - 1, j_4] * u[1:N - 1, j_4 + 1] + q[1:N - 1, j_4]

    S[0] = np.sum(u)


sdfg = kernel.to_sdfg()

sdfg.save("log_sdfgs/adi_forward.sdfg")

add_backward_pass(sdfg=sdfg, inputs=["u"], outputs=["S"], autooptimize=True)
# sdfg.simplify()
# sdfg.save("log_sdfgs/adi_backward.sdfg")
N_ = 15
TST = 3
# Create the u and S arrays and call the sdfg
# u array of ones float64
# u = np.fromfunction(lambda i, j: (i + N_ - j) / N_, (N_, N_), dtype=np.float64)
u = np.ones((N_, N_), dtype=np.float64) * 100
S = np.zeros([1], dtype=np.float64)

# adn gradient arrays
u_grad = np.zeros((N_, N_), dtype=np.float64)
S_grad = np.ones([1], dtype=np.float64)
sdfg(TST, u, S, gradient_u=u_grad, gradient_S=S_grad, N=N_)

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def adi_j(TSTEPS, u, S):
    N = u.shape[0]
    v = jnp.empty(u.shape, dtype=u.dtype)
    p = jnp.empty(u.shape, dtype=u.dtype)
    q = jnp.empty(u.shape, dtype=u.dtype)

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
        v = v.at[0, 1:N - 1].set(1.0)
        p = p.at[1:N - 1, 0].set(0.0)
        q = q.at[1:N - 1, 0].set(v[0, 1:N - 1])
        for j in range(1, N - 1):
            p = p.at[1:N - 1, j].set(-c / (a * p[1:N - 1, j - 1] + b))
            q = q.at[1:N - 1, j].set(
                (-d * u[j, 0:N - 2] +
                 (1.0 + 2.0 * d) * u[j, 1:N - 1] - f * u[j, 2:N] - a * q[1:N - 1, j - 1]) / (a * p[1:N - 1, j - 1] + b))
        v = v.at[N - 1, 1:N - 1].set(1.0)
        for j in range(N - 2, 0, -1):
            v = v.at[j, 1:N - 1].set(p[1:N - 1, j] * v[j + 1, 1:N - 1] + q[1:N - 1, j])

        u = u.at[1:N - 1, 0].set(1.0)
        p = p.at[1:N - 1, 0].set(0.0)
        q = q.at[1:N - 1, 0].set(u[1:N - 1, 0])
        for j in range(1, N - 1):
            p = p.at[1:N - 1, j].set(-f / (d * p[1:N - 1, j - 1] + e))
            q = q.at[1:N - 1, j].set(
                (-a * v[0:N - 2, j] +
                 (1.0 + 2.0 * a) * v[1:N - 1, j] - c * v[2:N, j] - d * q[1:N - 1, j - 1]) / (d * p[1:N - 1, j - 1] + e))
        u = u.at[1:N - 1, N - 1].set(1.0)
        for j in range(N - 2, 0, -1):
            u = u.at[1:N - 1, j].set(p[1:N - 1, j] * u[1:N - 1, j + 1] + q[1:N - 1, j])

    return jax.block_until_ready(jnp.sum(u))


# Instaniate JAX arrays and call the grad function
# enable jax_enable_x64

# u = jnp.fromfunction(lambda i, j: (i + N_ - j) / N_, (N_, N_), dtype=jnp.float64)
u = jnp.ones((N_, N_), dtype=jnp.float64) * 100

jax_grad = jax.grad(adi_j, argnums=1)

j_u = jax_grad(TST, u, N_)
print(np.max(np.max(j_u)))
print(np.max(j_u - u_grad))
print(j_u - u_grad)
print(j_u.dtype)
assert np.allclose(j_u, u_grad, atol=1e-6)
