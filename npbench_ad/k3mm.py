import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass

NI = 32
NJ = 32
NK = 32
NM = 32
NL = 32


@dc.program
def k3mm(A: dc.float64[NI, NK], B: dc.float64[NK, NJ], C: dc.float64[NJ, NM], D: dc.float64[NM, NL],
         E: dc.float64[NI, NL], S: dc.float64[1]):

    E[:] = A @ B @ C @ D

    @dc.map(_[0:NI, 0:NL])
    def summap(i, j):
        s >> S(1, lambda x, y: x + y)[0]
        z << E[i, j]
        s = z


alpha = 0.2
beta = 1.2

A = np.ones(shape=[NI, NJ])
B = np.ones(shape=[NK, NJ])
C = np.ones(shape=[NJ, NL])
D = np.ones(shape=[NI, NL])
E = np.ones(shape=[NI, NL])
gradient_A = np.zeros(shape=[NI, NL])
gradient_S = np.ones(shape=[1])
S = np.zeros(shape=[1])

k3mm.use_experimental_cfg_blocks = True
sdfg = k3mm.to_sdfg(A=A, B=B, C=C, D=D)

sdfg.save("log_sdfgs/k3mm_forward.sdfg")

add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["S"])

sdfg.save("log_sdfgs/k3mm_backward.sdfg")

sdfg(A, B, C, D, E, S, gradient_A=gradient_A, gradient_S=gradient_S)
# JAX
import jax
import jax.numpy as jnp


def k2mm_jax(A, B, C, D, E):
    E = E.at[:, :].set(A @ B @ C @ D)
    return jnp.sum(E)


jax_grad = jax.grad(k2mm_jax, argnums=[0])

A = jnp.ones(shape=[NI, NJ])
B = jnp.ones(shape=[NK, NJ])
C = jnp.ones(shape=[NJ, NL])
D = jnp.ones(shape=[NI, NL])
E = jnp.ones(shape=[NI, NL])

gradient_A_jax = jax_grad(A, B, C, D, E)
assert np.allclose(gradient_A_jax, gradient_A)
