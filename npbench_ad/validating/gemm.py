import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass

NI = 32
NJ = 36
NK = 38
NL = 42


@dc.program
def gemm(alpha: dc.float64, beta: dc.float64, A: dc.float64[NI, NK], B: dc.float64[NK, NJ], C: dc.float64[NI, NJ],
         D: dc.float64[NI, NJ], S: dc.float64[1]):

    D[:] = alpha * A @ B + beta * C

    S[0] = np.sum(D)


alpha = 0.2
beta = 1.2

A = np.ones(shape=[NI, NK])
B = np.ones(shape=[NK, NJ])
C = np.ones(shape=[NI, NJ])
D = np.ones(shape=[NI, NJ])
gradient_A = np.zeros(shape=[NI, NK])
gradient_S = np.ones(shape=[1])
S = np.zeros(shape=[1])

sdfg = gemm.to_sdfg(alpha=alpha, beta=beta, A=A, B=B, C=C, D=D)

sdfg.save("log_sdfgs/gemm_forward.sdfg")

add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["S"])

sdfg.save("log_sdfgs/gemm_backward.sdfg")

sdfg(alpha, beta, A, B, C, D, S, gradient_A=gradient_A, gradient_S=gradient_S)

# JAX
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def k2mm_jax(alpha, beta, A, B, C, D):
    D = D.at[:].set(alpha * A @ B + beta * C)
    return jnp.sum(D)


jax_grad = jax.grad(k2mm_jax, argnums=[2])

A = jnp.ones(shape=[NI, NK])
B = jnp.ones(shape=[NK, NJ])
C = jnp.ones(shape=[NI, NJ])
D = jnp.ones(shape=[NI, NJ])

gradient_A_jax = jax_grad(alpha, beta, A, B, C, D)
print(np.max(gradient_A_jax - gradient_A))
assert np.allclose(gradient_A_jax, gradient_A)
