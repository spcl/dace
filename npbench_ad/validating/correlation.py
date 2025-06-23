import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass
# JAX
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
N = 4
M = 4


@dc.program
def correlation(float_n: dc.float64, data: dc.float64[N, M], S: dc.float64[1]):

    mean = np.mean(data, axis=0)

    stddev = np.sqrt(np.mean(np.subtract(data, mean)**2, axis=0))
    stddev[stddev <= 0.1] = 1.0

    np.subtract(data, mean, out=data)

    np.divide(data, np.sqrt(float_n) * stddev, out=data)
    corr = np.eye(M, dtype=data.dtype)
    for i in range(M - 1):
        corr[i, i + 1:M] = data[:, i] @ data[:, i + 1:M]
        corr[i + 1:M, i] = corr[i, i + 1:M]

    S[0] = np.sum(corr)


sdfg = correlation.to_sdfg()

sdfg.save("log_sdfgs/correlation_forward.sdfg")

add_backward_pass(sdfg=sdfg, inputs=["data"], outputs=["S"], autooptimize=True)
# load sdfg
# sdfg = dc.SDFG.from_file("/scratch/aboudaou/workspace/dace/npbench_ad/validating/log_sdfgs/correlation_no_simp.sdfg")
# sdfg.simplify()
# sdfg.save("log_sdfgs/correlation_no_simp.sdfg")
sdfg.save("log_sdfgs/correlation_after_simp.sdfg")

float_n = dc.float64(N)
# base_data = np.random.normal(loc=0.0, scale=1.0, size=(N, M))

# # Compute current stddev per column (to scale precisely)
# col_std = base_data.std(axis=0, ddof=1)

# # Rescale each column to have exactly 0.5 std dev
# for m in range(M):
#     if col_std[m] > 0:
#         base_data[:, m] = (base_data[:, m] / col_std[m]) * 0.5

# data = base_data
data = np.fromfunction(lambda i, j: (i * j) / M + i, (N, M), dtype=np.float64)
S = np.zeros(shape=[1], dtype=np.float64)
gradient_S = np.ones(shape=[1], dtype=np.float64)
gradient_data = np.zeros(shape=(N, M), dtype=np.float64)

sdfg(float_n, data, S, gradient_data=gradient_data, gradient_S=gradient_S)
print(gradient_data)


def k2mm_jax(float_n, data):
    mean = jnp.mean(data, axis=0)

    stddev = jnp.sqrt(jnp.mean(jnp.subtract(data, mean)**2, axis=0))
    stddev = stddev.at[stddev <= 0.1].set(1.0)

    data = jnp.subtract(data, mean)

    data = jnp.divide(data, jnp.sqrt(float_n) * stddev)
    corr = jnp.eye(M, dtype=data.dtype)
    for i in range(M - 1):
        corr = corr.at[i, i + 1:M].set(data[:, i] @ data[:, i + 1:M])
        corr = corr.at[i + 1:M, i].set(corr[i, i + 1:M])

    return jnp.sum(corr)


jax_grad = jax.grad(k2mm_jax, argnums=[1])
data_j = jnp.fromfunction(lambda i, j: (i * j) / M + i, (N, M), dtype=jnp.float64)
float_n = jnp.copy(float_n)

gradient_A_jax = jax_grad(float_n, data_j)
print(gradient_A_jax)
print(gradient_A_jax - gradient_data)
assert np.allclose(gradient_A_jax, gradient_data, atol=1e-6)
