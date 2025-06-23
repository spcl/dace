import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass
import jax.lax as lax
N = 32
M = 32
TSTEPS = 3


@dc.program
def seidel_2d(TSTEPS: dc.int64, A: dc.float64[N, N], S: dc.float64[1]):

    for t in range(0, TSTEPS - 1):
        for i in range(1, N - 1):
            A[i, 1:-1] += (A[i - 1, :-2] + A[i - 1, 1:-1] + A[i - 1, 2:] + A[i, 2:] + A[i + 1, :-2] + A[i + 1, 1:-1] +
                           A[i + 1, 2:])
            for j in range(1, N - 1):
                A[i, j] += A[i, j - 1]
                A[i, j] /= 9.0

    @dc.map(_[0:N, 0:N])
    def summap(i, j):
        s >> S(1, lambda x, y: x + y)[0]
        z << A[i, j]
        s = z


sdfg = seidel_2d.to_sdfg()

sdfg.save("log_sdfgs/seidel_2d_forward.sdfg")

add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["S"])

sdfg.save("log_sdfgs/seidel_2d_backward.sdfg")

A = np.ones(shape=[N, N])
S = np.zeros(shape=[1])
gradient_S = np.ones(shape=[1])
gradient_A = np.zeros(shape=[N, N])

sdfg(TSTEPS, A, S, gradient_S=gradient_S, gradient_A=gradient_A)

# JAX
import jax
import jax.numpy as jnp

def jax_kernel_lax(TSTEPS, A, S):
    # Ensure TSTEPS and N are concrete integers.
    N = A.shape[0]
    # Outer loop: iterate for TSTEPS-1 iterations.
    def loop1_body(A, t):
        # Middle loop: iterate over rows i from 1 to N-2.
        def loop2_body(A, i):
            # First, update row i in a vectorized way.
            # Update columns 1:-1:
            update_val = (A[i, 1:-1] +
                          (A[i - 1, :-2] + A[i - 1, 1:-1] + A[i - 1, 2:] +
                           A[i, 2:] +
                           A[i + 1, :-2] + A[i + 1, 1:-1] + A[i + 1, 2:]))
            A = A.at[i, 1:-1].set(update_val)

            # Inner loop: iterate over columns j from 1 to N-2.
            def loop3_body(A, j):
                # Update element A[i, j] based on its left neighbor.
                new_val = (A[i, j] + A[i, j - 1]) / 9.0
                A = A.at[i, j].set(new_val)
                return A, None

            A, _ = lax.scan(loop3_body, A, jnp.arange(1, N - 1))
            return A, None

        A, _ = lax.scan(loop2_body, A, jnp.arange(1, N - 1))
        return A, None

    A, _ = lax.scan(loop1_body, A, jnp.arange(TSTEPS - 1))
    return jnp.sum(A)

def k2mm_jax(TSTEPS, A):
    for t in range(0, TSTEPS - 1):
        for i in range(1, N - 1):
            A = A.at[i, 1:-1].set(A[i, 1:-1] + (A[i - 1, :-2] + A[i - 1, 1:-1] + A[i - 1, 2:] + A[i, 2:] +
                                                A[i + 1, :-2] + A[i + 1, 1:-1] + A[i + 1, 2:]))
            for j in range(1, N - 1):
                A = A.at[i, j].set(A[i, j] + A[i, j - 1])
                A = A.at[i, j].set(A[i, j] / 9.0)
    return jnp.sum(A)


jax_grad = jax.jit(jax.grad(jax_kernel_lax, argnums=[1]), static_argnums=[0])

A = jnp.ones(shape=[N, N])
# A_gpu = jax.device_put(A)

# # gradient_A_jax = jax_grad(TSTEPS, A_gpu)
# assert np.allclose(gradient_A_jax, gradient_A)
