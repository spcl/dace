import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass

NI = 32
NJ = 32
NK = 32
NL = 32


@dc.program
def k2mm(alpha: dc.float64, beta: dc.float64, A: dc.float64[NI, NK], B: dc.float64[NK, NJ], C: dc.float64[NJ, NL],
         D: dc.float64[NI, NL], S: dc.float64[1]):

    D[:] = alpha * A @ B @ C + beta * D

    @dc.map(_[0:NI, 0:NL])
    def summap(i, j):
        s >> S(1, lambda x, y: x + y)[0]
        z << D[i, j]
        s = z


alpha = 0.2
beta = 1.2

A = np.ones(shape=[NI, NJ])
B = np.ones(shape=[NK, NJ])
C = np.ones(shape=[NJ, NL])
D = np.ones(shape=[NI, NL])
gradient_A = np.zeros(shape=[NI, NL])
gradient_S = np.ones(shape=[1])
S = np.zeros(shape=[1])

sdfg = k2mm.to_sdfg(alpha=alpha, beta=beta, A=A, B=B, C=C, D=D)

sdfg.save("log_sdfgs/k2mm_forward.sdfg")

add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["S"])

sdfg.save("log_sdfgs/k2mm_backward.sdfg")

sdfg(alpha, beta, A, B, C, D, S, gradient_A=gradient_A, gradient_S=gradient_S)

# JAX
import jax
import jax.numpy as jnp


def k2mm_jax(alpha, beta, A, B, C, D):
    D = D.at[:, :].set(alpha * A @ B @ C + beta * D)
    return jnp.sum(D)


jax_grad = jax.grad(k2mm_jax, argnums=[2])

A = jnp.ones(shape=[NI, NJ])
B = jnp.ones(shape=[NK, NJ])
C = jnp.ones(shape=[NJ, NL])
D = jnp.ones(shape=[NI, NL])

gradient_A_jax = jax_grad(alpha, beta, A, B, C, D)
assert np.allclose(gradient_A_jax, gradient_A)
