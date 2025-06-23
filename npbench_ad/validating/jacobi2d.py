import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass

N = 32
M = 32
TSTEPS = 5


@dc.program
def jacobi_2d(TSTEPS: dc.int64, A: dc.float32[N, N], B: dc.float32[N, N], S: dc.float32[1]):

    for t in range(1, TSTEPS):
        B[1:-1, 1:-1] = 0.2 * (A[1:-1, 1:-1] + A[1:-1, :-2] + A[1:-1, 2:] + A[2:, 1:-1] + A[:-2, 1:-1])
        A[1:-1, 1:-1] = 0.2 * (B[1:-1, 1:-1] + B[1:-1, :-2] + B[1:-1, 2:] + B[2:, 1:-1] + B[:-2, 1:-1])

    @dc.map(_[0:N, 0:N])
    def summap(i, j):
        s >> S(1, lambda x, y: x + y)[0]
        z << A[i, j]
        s = z


sdfg = jacobi_2d.to_sdfg(use_cache=False)

sdfg.save("log_sdfgs/jacobi_2d_forward.sdfg")

add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["S"])

sdfg.save("log_sdfgs/jacobi_2d_backward.sdfg")

A = np.ones(shape=[N, N], dtype=np.float32)
B = np.ones(shape=[N, N], dtype=np.float32)
S = np.zeros(shape=[1], dtype=np.float32)
gradient_S = np.ones(shape=[1], dtype=np.float32)
gradient_A = np.zeros(shape=[N, N], dtype=np.float32)

sdfg(TSTEPS, A, B, S, gradient_S=gradient_S, gradient_A=gradient_A)
# print(gradient_A)
# JAX
import jax
import jax.numpy as jnp


def k2mm_jax(TSTEPS, A, B):
    for t in range(1, TSTEPS):
        B = B.at[1:-1, 1:-1].set(0.2 * (A[1:-1, 1:-1] + A[1:-1, :-2] + A[1:-1, 2:] + A[2:, 1:-1] + A[:-2, 1:-1]))
        A = A.at[1:-1, 1:-1].set(0.2 * (B[1:-1, 1:-1] + B[1:-1, :-2] + B[1:-1, 2:] + B[2:, 1:-1] + B[:-2, 1:-1]))
    return jnp.sum(A)


jax_grad = jax.grad(k2mm_jax, argnums=[1])

A = jnp.ones(shape=[N, N])
B = jnp.ones(shape=[N, N])

gradient_A_jax = jax_grad(TSTEPS, A, B)
assert np.allclose(gradient_A_jax, gradient_A)
