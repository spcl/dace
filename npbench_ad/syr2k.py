import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass

N = 32
M = 32


@dc.program
def syr2k(alpha: dc.float64, beta: dc.float64, C: dc.float64[N, N], A: dc.float64[N, M], B: dc.float64[N, M],
          S: dc.float64[1]):

    for i in range(N):
        C[i, :i + 1] *= beta
        for k in range(M):
            C[i, :i + 1] += (A[:i + 1, k] * alpha * B[i, k] + B[:i + 1, k] * alpha * A[i, k])

    @dc.map(_[0:N, 0:N])
    def summap(i, j):
        s >> S(1, lambda x, y: x + y)[0]
        z << C[i, j]
        s = z


sdfg = syr2k.to_sdfg()

sdfg.save("log_sdfgs/syr2k_forward.sdfg")

add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["S"])

sdfg.save("log_sdfgs/syr2k_backward.sdfg")

alpha = 0.2
beta = 1.2
A = np.ones(shape=[N, M])
B = np.ones(shape=[N, M])
C = np.ones(shape=[N, N])
S = np.zeros(shape=[1])
gradient_S = np.ones(shape=[1])
gradient_A = np.zeros(shape=[N, M])
sdfg(alpha, beta, C, A, B, S, gradient_S=gradient_S, gradient_A=gradient_A)

# JAX
import jax
import jax.numpy as jnp


def k2mm_jax(alpha, beta, C, A, B):
    for i in range(N):
        C = C.at[i, :i + 1].set(C[i, :i + 1] * beta)
        for k in range(M):
            C = C.at[i, :i + 1].set(C[i, :i + 1] + (A[:i + 1, k] * alpha * B[i, k] + B[:i + 1, k] * alpha * A[i, k]))
    return jnp.sum(C)


jax_grad = jax.grad(k2mm_jax, argnums=[3])

C = jnp.ones(shape=[N, N])
A = jnp.ones(shape=[N, M])
B = jnp.ones(shape=[N, M])

gradient_A_jax = jax_grad(alpha, beta, C, A, B)
assert np.allclose(gradient_A_jax, gradient_A)
