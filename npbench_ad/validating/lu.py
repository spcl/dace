import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass
import jax
import jax.numpy as jnp
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.dtypes import DeviceType

jax.config.update("jax_enable_x64", True)
N = dc.symbol('N', dtype=dc.int64)


@dc.program
def lu(A: dc.float64[N, N], S: dc.float64[1]):

    for i in range(N):
        for j in range(i):
            A[i, j] -= A[i, :j] @ A[:j, j]
            A[i, j] /= A[j, j]
        for j1 in range(i, N):
            A[i, j1] -= A[i, :i] @ A[:i, j1]

    S[0] = np.sum(A)


sdfg = lu.to_sdfg()

sdfg.save("log_sdfgs/lu_forward.sdfg")

add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["S"], autooptimize=False)
# sdfg.simplify()
# sdfg = auto_optimize(sdfg, device=DeviceType.CPU)
sdfg.save("log_sdfgs/lu_backward_not.sdfg")

# sdfg.compile()

N_m = 5
# # create the sdfg arrays in addition to gradient arrays
A = np.empty((N_m, N_m), dtype=np.float64)
for i in range(N_m):
    for j in range(i + 1):
        A[i, j] = (-j % N_m) / N_m + 1
    for j in range(i + 1, N_m):
        A[i, j] = 0.0
    A[i, i] = 1.0

B = np.empty((N_m, N_m), dtype=np.float64)
B[:] = A @ np.transpose(A)
A[:] = B
S = np.zeros(shape=[1], dtype=np.float64)
S_j = jnp.array(S)
gradient_S = np.ones(shape=[1], dtype=np.float64)
gradient_A = np.zeros(shape=[N_m, N_m], dtype=np.float64)

sdfg(A, S, gradient_S=gradient_S, gradient_A=gradient_A, N=N_m)


def jax_kernel(A, S):
    for i in range(A.shape[0]):
        for j in range(i):
            A = A.at[i, j].set(A[i, j] - A[i, :j] @ A[:j, j])
            A = A.at[i, j].set(A[i, j] / A[j, j])
        for j in range(i, A.shape[0]):
            A = A.at[i, j].set(A[i, j] - A[i, :i] @ A[:i, j])

    S = S.at[0].set(jnp.sum(A))
    return S[0]


jax_grad = jax.grad(jax_kernel, argnums=[0])

A = np.empty((N_m, N_m), dtype=np.float64)
for i in range(N_m):
    for j in range(i + 1):
        A[i, j] = (-j % N_m) / N_m + 1
    for j in range(i + 1, N_m):
        A[i, j] = 0.0
    A[i, i] = 1.0

B = np.empty((N_m, N_m), dtype=np.float64)
B[:] = A @ np.transpose(A)
A[:] = B
A_j = jnp.array(A)
S = np.zeros(shape=[1], dtype=np.float64)
S_j = jnp.array(S)
j_grad = jax_grad(A_j, S_j)
print(j_grad)
print(np.max(np.abs(j_grad - gradient_A)))
# print(gradient_A)
# Increased tolerance because jax doesn't do float64 for some reason
assert np.allclose(j_grad, gradient_A, atol=1e-5)
