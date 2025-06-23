import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass
import jax
import jax.numpy as jnp

N = 32
M = 32
TSTEPS = 5


@dc.program
def jacobi_1d(TSTEPS: dc.int64, A: dc.float32[N], B: dc.float32[N], S: dc.float32[1]):

    for t in range(1, TSTEPS):
        B[1:-1] = 0.33333 * (A[:-2] + A[1:-1] + A[2:])
        A[1:-1] = 0.33333 * (B[:-2] + B[1:-1] + B[2:])

    @dc.map(_[0:N])
    def summap(i):
        s >> S(1, lambda x, y: x + y)[0]
        z << A[i]
        s = z


sdfg = jacobi_1d.to_sdfg(use_cache=False)

sdfg.save("log_sdfgs/jacobi_1d_forward.sdfg")

add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["S"])

sdfg.save("log_sdfgs/jacobi_1d_backward.sdfg")

A = np.fromfunction(lambda i: (i + 2) / N, (N, ), dtype=np.float32)
A_j = jnp.copy(A)
B = np.fromfunction(lambda i: (i + 3) / N, (N, ), dtype=np.float32)
B_j = jnp.copy(B)
S = np.zeros(shape=[1], dtype=np.float32)
gradient_S = np.ones(shape=[1], dtype=np.float32)
gradient_A = np.zeros(shape=[N], dtype=np.float32)

sdfg(TSTEPS, A, B, S, gradient_S=gradient_S, gradient_A=gradient_A)
# print(gradient_A)
# JAX


def k2mm_jax(TSTEPS, A, B):
    for t in range(1, TSTEPS):
        B = B.at[1:-1].set(0.33333 * (A[:-2] + A[1:-1] + A[2:]))
        A = A.at[1:-1].set(0.33333 * (B[:-2] + B[1:-1] + B[2:]))
    return jnp.sum(A)


jax_grad = jax.grad(k2mm_jax, argnums=[1])

gradient_A_jax = jax_grad(TSTEPS, A_j, B_j)
assert np.allclose(gradient_A_jax, gradient_A)
