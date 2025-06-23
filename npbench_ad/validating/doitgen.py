import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass

NP = 32
NQ = 34
NR = 36


@dc.program
def doitgen(A: dc.float64[NR, NQ, NP], C4: dc.float64[NP, NP], S: dc.float64[1]):

    for r in range(NR):
        A[r, :, :] = np.reshape(np.reshape(A[r], (NQ, NP)) @ C4, (NQ, NP))

    S[0] = np.sum(A)


sdfg = doitgen.to_sdfg()

sdfg.save("log_sdfgs/doitgen_forward.sdfg")

add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["S"], autooptimize=True)

sdfg.save("log_sdfgs/doitgen_backward.sdfg")

S = np.zeros(shape=[1])
gradient_S = np.ones(shape=[1])
gradient_input = np.zeros(shape=(NR, NQ, NP))

A = np.fromfunction(lambda i, j, k: ((i * j + k) % NP) / NP, (NR, NQ, NP), dtype=np.float64)
C4 = np.fromfunction(lambda i, j: (i * j % NP) / NP, (NP, NP), dtype=np.float64)
sdfg(A, C4, S, gradient_A=gradient_input, gradient_S=gradient_S)

# JAX
import jax
import jax.numpy as jnp


def k2mm_jax(A, C4):
    for r in range(NR):
        A = A.at[r, :, :].set(jnp.reshape(jnp.reshape(A[r], (NQ, NP)) @ C4, (NQ, NP)))
    return jnp.sum(A)


jax_grad = jax.grad(k2mm_jax, argnums=[0])

A = jnp.copy(A)
C4 = jnp.copy(C4)

gradient_A_jax = jax_grad(A, C4)
assert np.allclose(gradient_A_jax, gradient_input)
