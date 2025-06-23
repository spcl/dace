import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass
import jax.lax as lax
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

add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["S"], autooptimize=True)
# sdfg.save("log_sdfgs/symm_backward.sdfg")
# sdfg.simplify(validate=True)
# sdfg.save("log_sdfgs/symm_backward_a.sdfg")

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

def jax_kernel_lax(alpha, beta, C: jax.Array, A: jax.Array, B: jax.Array):
    # Allocate temporary storage of shape (C.shape[1],)
    temp2 = jnp.empty((C.shape[1],), dtype=C.dtype)
    # Scale C by beta.
    C = C * beta

    # Outer scan: loop over row index i.
    def row_update_body(carry, i):
        C, temp2 = carry

        # Inner scan: loop over column index j.
        def col_update_body(carry_inner, j):
            C, temp2 = carry_inner

            # For row i, compute A_slice and B_slice using a mask.
            A_slice = jnp.where(jnp.arange(A.shape[1]) < i, A[i, :], 0.0)
            B_slice = jnp.where(jnp.arange(B.shape[0]) < i, B[:, j], 0.0)

            # Update column j of C:
            updated_col = C[:, j] + (alpha * B[i, j] * A_slice)
            C = lax.dynamic_update_slice(C, updated_col[:, None], (0, j))
            # Update temp2 at index j as the dot product of B_slice and A_slice.
            temp2 = temp2.at[j].set(B_slice @ A_slice)
            return (C, temp2), jnp.array(0)  # dummy output

        # Run inner scan over j in 0 ... C.shape[1]-1.
        (C, temp2), _ = lax.scan(col_update_body, (C, temp2), jnp.arange(C.shape[1]))
        # After scanning all columns, update row i of C.
        C = C.at[i, :].add(alpha * B[i, :] * A[i, i] + alpha * temp2)
        return (C, temp2), jnp.array(0)  # dummy output for the outer scan

    # Run outer scan over i in 0 ... C.shape[0]-1.
    (C, temp2), _ = lax.scan(row_update_body, (C, temp2), jnp.arange(C.shape[0]))
    return jnp.sum(C)

def k2mm_jax(alpha, beta, C, A, B):
    temp2 = jnp.empty((N, ), dtype=C.dtype)
    C = C.at[:, :].set(C[:, :] * beta)
    for i in range(M):
        for j in range(N):
            C = C.at[:i, j].set(C[:i, j] + alpha * B[i, j] * A[i, :i])
            temp2 = temp2.at[j].set(B[:i, j] @ A[i, :i])
        C = C.at[i, :].set(C[i, :] + alpha * B[i, :] * A[i, i] + alpha * temp2)
    return jnp.sum(C)


jax_grad = jax.grad(jax_kernel_lax, argnums=[3])

C = jnp.ones(shape=[N, N])
A = jnp.ones(shape=[N, M])
B = jnp.ones(shape=[N, M])

gradient_A_jax = jax_grad(alpha, beta, C, A, B)
assert np.allclose(gradient_A_jax, gradient_A)
