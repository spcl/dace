import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass

N = 32
M = 32


@dc.program
def symm(alpha: dc.float64, beta: dc.float64, C: dc.float64[M, N], A: dc.float64[M, M], B: dc.float64[M, N],
         S: dc.float64[1]):

    temp2 = np.empty((N, ), dtype=C.dtype)
    C *= beta
    for i in range(M):
        for j in range(N):
            C[:i, j] += alpha * B[i, j] * A[i, :i]
            temp2[j] = B[:i, j] @ A[i, :i]
        C[i, :] += alpha * B[i, :] * A[i, i] + alpha * temp2

    @dc.map(_[0:N, 0:N])
    def summap(i, j):
        s >> S(1, lambda x, y: x + y)[0]
        z << C[i, j]
        s = z


sdfg = symm.to_sdfg()

sdfg.save("log_sdfgs/symm_forward.sdfg")

add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["S"])

sdfg.save("log_sdfgs/symm_backward.sdfg")

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
    temp2 = jnp.empty((N, ), dtype=C.dtype)
    C = C.at[:, :].set(C[:, :] * beta)
    for i in range(M):
        for j in range(N):
            C = C.at[:i, j].set(C[:i, j] + alpha * B[i, j] * A[i, :i])
            temp2 = temp2.at[j].set(B[:i, j] @ A[i, :i])
        C = C.at[i, :].set(C[i, :] + alpha * B[i, :] * A[i, i] + alpha * temp2)
    return jnp.sum(C)


jax_grad = jax.grad(k2mm_jax, argnums=[3])

C = jnp.ones(shape=[N, N])
A = jnp.ones(shape=[N, M])
B = jnp.ones(shape=[N, M])

gradient_A_jax = jax_grad(alpha, beta, C, A, B)
assert np.allclose(gradient_A_jax, gradient_A)
