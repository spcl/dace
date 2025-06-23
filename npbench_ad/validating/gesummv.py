import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass

N = 32


@dc.program
def gesummv(alpha: dc.float64, beta: dc.float64, A: dc.float64[N, N], B: dc.float64[N, N], x: dc.float64[N],
            D: dc.float64[N], S: dc.float64[1]):

    D[:] = alpha * A @ x + beta * B @ x
    S[0] = np.sum(D)


sdfg = gesummv.to_sdfg()

sdfg.save("log_sdfgs/gesummv_forward.sdfg")

add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["S"])

sdfg.save("log_sdfgs/gesummv_backward.sdfg")

alpha = 0.2
beta = 1.2
A = np.ones(shape=[N, N])
D = np.ones(shape=[N])
B = np.ones(shape=[N, N])
x = np.ones(shape=[N])
gradient_A = np.zeros(shape=[N, N])
gradient_S = np.ones(shape=[1])
S = np.zeros(shape=[1])

sdfg(alpha, beta, A, B, x, D, S, gradient_A=gradient_A, gradient_S=gradient_S)

# JAX
import jax
import jax.numpy as jnp


def k2mm_jax(alpha, beta, A, B, x, D):
    D = D.at[:].set(alpha * A @ x + beta * B @ x)
    return jnp.sum(D)


jax_grad = jax.grad(k2mm_jax, argnums=[2])

A = jnp.ones(shape=[N, N])
D = jnp.ones(shape=[N])
B = jnp.ones(shape=[N, N])
x = jnp.ones(shape=[N])

gradient_A_jax = jax_grad(alpha, beta, A, B, x, D)
assert np.allclose(gradient_A_jax, gradient_A)
