import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass

M = 32
N = 32

L_M, L_N = 32, 32


@dc.program
def trmm(alpha: dc.float64, A: dc.float64[M, M], B: dc.float64[M, N], S: dc.float64[1]):

    for i in range(L_M):
        for j in range(L_N):
            B[i, j] += np.dot(A[i + 1:, i], B[i + 1:, j])
    B *= alpha

    @dc.map(_[0:M, 0:N])
    def summap(i, j):
        s >> S(1, lambda x, y: x + y)[0]
        z << B[i, j]
        s = z


alpha = 0.2
beta = 1.2

A = np.ones(shape=[M, M])
B = np.ones(shape=[M, N])
S = np.zeros(shape=[1])
gradient_S = np.ones(shape=[1])
gradient_B = np.zeros(shape=[M, N])
sdfg = trmm.to_sdfg(alpha=alpha, beta=beta, A=A)

sdfg.save("log_sdfgs/trmm_forward.sdfg")

add_backward_pass(sdfg=sdfg, inputs=["B"], outputs=["S"], autooptimize=True)
# sdfg.save("log_sdfgs/trmm_backward_bs.sdfg")
# sdfg.simplify()
sdfg.save("log_sdfgs/trmm_backward.sdfg")

sdfg(
    alpha,
    A,
    B,
    S,
    gradient_B=gradient_B,
    gradient_S=gradient_S,
)

# JAX
import jax
import jax.numpy as jnp


def k2mm_jax(alpha, A, B):
    for i in range(L_M):
        for j in range(L_N):
            B = B.at[i, j].set(B[i, j] + jnp.dot(A[i + 1:, i], B[i + 1:, j]))
    B *= alpha
    return jnp.sum(B)


jax_grad = jax.grad(k2mm_jax, argnums=[2])

A = jnp.ones(shape=[M, M])
B = jnp.ones(shape=[M, N])

gradient_A_jax = jax_grad(alpha, A, B)
print(gradient_A_jax)
print(gradient_B)
assert np.allclose(gradient_A_jax, gradient_B)
