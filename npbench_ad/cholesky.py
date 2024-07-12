import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass

N = 32
k = 1


@dc.program
def cholesky(A: dc.float64[N, N], S: dc.float64[1]):

    A[0, 0] = np.sqrt(A[0, 0])
    for i in range(1, N):
        for j in range(i):
            A[i, j] -= np.dot(A[i, :j], A[j, :j])
            A[i, j] /= A[j, j]
        A[i, i] -= np.dot(A[i, :i], A[i, :i])
        A[i, i] = np.sqrt(A[i, i])

    @dc.map(_[0:N, 0:N])
    def summap(i, j):
        s >> S(1, lambda x, y: x + y)[0]
        z << A[i, j]
        s = z


sdfg = cholesky.to_sdfg()

sdfg.save("log_sdfgs/cholesky_forward.sdfg")

add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["S"])

sdfg.save("log_sdfgs/cholesky_backward.sdfg")

A = np.ones(shape=[N, N])
S = np.zeros(shape=[1])
gradient_S = np.ones(shape=[1])
gradient_A = np.zeros(shape=[N, N])
sdfg(A, S, gradient_S=gradient_S, gradient_A=gradient_A)

# JAX
import jax
import jax.numpy as jnp


def k2mm_jax(A):
    A = A.at[0, 0].set(jnp.sqrt(A[0, 0]))
    for i in range(1, N):
        for j in range(i):
            A = A.at[i, j].set(A[i, j] - jnp.dot(A[i, :j], A[j, :j]))
            A = A.at[i, j].set(A[i, j] / A[j, j])
        A = A.at[i, i].set(A[i, i] - jnp.dot(A[i, :i], A[i, :i]))
        A = A.at[i, i].set(jnp.sqrt(A[i, i]))
    return jnp.sum(A)


jax_grad = jax.grad(k2mm_jax, argnums=[0])

A = jnp.ones(shape=[N, N])
gradient_A_jax = jax_grad(A)
print(gradient_A_jax)
print(gradient_A)
assert np.allclose(gradient_A_jax, gradient_A)
