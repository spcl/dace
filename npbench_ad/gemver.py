import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass

N = 32
M = 32


@dc.program
def gemver(alpha: dc.float64, beta: dc.float64, A: dc.float64[N, N], u1: dc.float64[N], v1: dc.float64[N],
           u2: dc.float64[N], v2: dc.float64[N], w: dc.float64[N], x: dc.float64[N], y: dc.float64[N], z: dc.float64[N],
           S: dc.float64[1]):

    A += np.multiply.outer(u1, v1) + np.multiply.outer(u2, v2)
    x += beta * y @ A + z
    w += alpha * A @ x

    S[0] = np.sum(w)


A = np.ones(shape=[N, N])
u1 = np.ones(shape=[N])
u2 = np.ones(shape=[N])
v1 = np.ones(shape=[N])
v2 = np.ones(shape=[N])
x = np.ones(shape=[N])
y = np.ones(shape=[N])
w = np.ones(shape=[N])
z = np.ones(shape=[N])

sdfg = gemver.to_sdfg()

sdfg.save("log_sdfgs/gemver_forward.sdfg")

add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["S"])

sdfg.save("log_sdfgs/gemver_backward.sdfg")

alpha = 0.2
beta = 1.2
A = np.ones(shape=[N, N])
u1 = np.ones(shape=[N])
u2 = np.ones(shape=[N])
v1 = np.ones(shape=[N])
v2 = np.ones(shape=[N])
x = np.ones(shape=[N])
y = np.ones(shape=[N])
w = np.ones(shape=[N])
z = np.ones(shape=[N])
gradient_A = np.zeros(shape=[N, N])
gradient_S = np.ones(shape=[1])
S = np.zeros(shape=[1])

sdfg(alpha, beta, A, u1, v1, u2, v2, w, x, y, z, S, gradient_A=gradient_A, gradient_S=gradient_S)

# JAX
import jax
import jax.numpy as jnp


def k2mm_jax(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    A += np.multiply.outer(u1, v1) + np.multiply.outer(u2, v2)
    x += beta * y @ A + z
    w += alpha * A @ x
    return jnp.sum(w)


jax_grad = jax.grad(k2mm_jax, argnums=[2])
A = jnp.ones(shape=[N, N])
u1 = jnp.ones(shape=[N])
u2 = jnp.ones(shape=[N])
v1 = jnp.ones(shape=[N])
v2 = jnp.ones(shape=[N])
x = jnp.ones(shape=[N])
y = jnp.ones(shape=[N])
w = jnp.ones(shape=[N])
z = jnp.ones(shape=[N])
gradient_A_jax = jax_grad(alpha, beta, A, u1, v1, u2, v2, w, x, y, z)
assert np.allclose(gradient_A_jax, gradient_A)
