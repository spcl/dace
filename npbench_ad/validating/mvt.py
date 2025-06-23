import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass

N = 32
M = 32


@dc.program
def mvt(x1: dc.float64[N], x2: dc.float64[N], y_1: dc.float64[N], y_2: dc.float64[N], A: dc.float64[N, N],
        S: dc.float64[1]):

    x1 += A @ y_1
    x2 += y_2 @ A

    @dc.map(_[0:N])
    def summap(i):
        s >> S(1, lambda x, y: x + y)[0]
        z << x2[i]
        s = z


A = np.ones(shape=[N, N])
u1 = np.ones(shape=[N])
u2 = np.ones(shape=[N])
v1 = np.ones(shape=[N])
v2 = np.ones(shape=[N])
x = np.ones(shape=[N])
y = np.ones(shape=[N])
w = np.ones(shape=[N])
z = np.ones(shape=[N])

sdfg = mvt.to_sdfg()

sdfg.save("log_sdfgs/mvt_forward.sdfg")

add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["S"])

sdfg.save("log_sdfgs/mvt_backward.sdfg")

A = np.ones(shape=[N, N])
x1 = np.ones(shape=[N])
x2 = np.ones(shape=[N])
y_1 = np.ones(shape=[N])
y_2 = np.ones(shape=[N])
S = np.zeros(shape=[1])
gradient_S = np.ones(shape=[1])
gradient_A = np.zeros(shape=[N, N])
sdfg(x1, x2, y_1, y_2, A, S, gradient_S=gradient_S, gradient_A=gradient_A)

# JAX
import jax
import jax.numpy as jnp


def k2mm_jax(x1, x2, y_1, y_2, A):
    x1 += A @ y_1
    x2 += y_2 @ A
    return jnp.sum(x2)


jax_grad = jax.grad(k2mm_jax, argnums=[4])

A = jnp.ones(shape=[N, N])
x1 = jnp.ones(shape=[N])
x2 = jnp.ones(shape=[N])
y_1 = jnp.ones(shape=[N])
y_2 = jnp.ones(shape=[N])

gradient_A_jax = jax_grad(x1, x2, y_1, y_2, A)
assert np.allclose(gradient_A_jax, gradient_A)
