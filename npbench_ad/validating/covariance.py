import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass
# JAX
import jax
import jax.numpy as jnp
import copy

N = 4
M = 4


@dc.program
def covariance(float_n: dc.float64, data: dc.float64[N, M], S: dc.float64[1]):

    mean = np.mean(data, axis=0)
    # data -= mean
    np.subtract(data, mean, out=data)
    cov = np.zeros((M, M), dtype=data.dtype)
    for i in range(M):
        cov[i, i:M] = data[:, i] @ data[:, i:M] / (float_n - 1.0)
        cov[i:M, i] = cov[i, i:M]

    @dc.map(_[0:M, 0:M])
    def summap(i, j):
        s >> S(1, lambda x, y: x + y)[0]
        z << cov[i, j]
        s = z


sdfg = covariance.to_sdfg()

sdfg.save("log_sdfgs/covariance_forward.sdfg")

add_backward_pass(sdfg=sdfg, inputs=["data"], outputs=["S"], autooptimize=True)
sdfg.simplify()
sdfg.save("log_sdfgs/covariance_backward.sdfg")

float_n = dc.float64(N)
base_data = np.random.normal(loc=0.0, scale=1.0, size=(N, M))

# Compute current stddev per column (to scale precisely)
col_std = base_data.std(axis=0, ddof=1)

# Rescale each column to have exactly 0.5 std dev
for m in range(M):
    if col_std[m] > 0:
        base_data[:, m] = (base_data[:, m] / col_std[m]) * 0.5

data = base_data
# data = np.fromfunction(lambda i, j: (i * j) / M + i, (N, M), dtype=np.float64)
data_j = copy.deepcopy(data)
S = np.zeros(shape=[1], dtype=np.float64)
gradient_S = np.ones(shape=[1], dtype=np.float64)
gradient_data = np.zeros(shape=(N, M), dtype=np.float64)

sdfg(float_n, data, S, gradient_data=gradient_data, gradient_S=gradient_S)
print(gradient_data)


def k2mm_jax(float_n, data):
    mean = jnp.mean(data, axis=0)
    data -= mean
    cov = jnp.zeros((M, M), dtype=data.dtype)
    for i in range(M):
        cov = cov.at[i, i:M].set(data[:, i] @ data[:, i:M] / (float_n - 1.0))
        cov = cov.at[i:M, i].set(cov[i, i:M])  # Update the row

    return jnp.sum(cov)


jax_grad = jax.grad(k2mm_jax, argnums=[1])

float_n = jnp.copy(float_n)
data_j = jnp.array(data_j)
gradient_A_jax = jax_grad(float_n, data_j)
print(gradient_A_jax - gradient_data)
print(np.max(gradient_A_jax - gradient_data))
assert np.allclose(gradient_A_jax, gradient_data)
