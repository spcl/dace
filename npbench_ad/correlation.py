import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass

N = 32
M = 32


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

add_backward_pass(sdfg=sdfg, inputs=["data"], outputs=["S"])

sdfg.save("log_sdfgs/correlation_backward.sdfg")

float_n = dc.float64(N)
data = np.fromfunction(lambda i, j: (i * j) / M + i, (N, M), dtype=np.float64)
S = np.zeros(shape=[1])
gradient_S = np.ones(shape=[1])
gradient_data = np.zeros(shape=(N, M))

sdfg(float_n, data, S, gradient_data=gradient_data, gradient_S=gradient_S)
print(gradient_data)

# JAX
import jax
import jax.numpy as jnp


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

float_n = jnp.copy(float_n)
data = jnp.copy(data)

gradient_A_jax = jax_grad(float_n, data)
print(gradient_A_jax)
assert np.allclose(gradient_A_jax, gradient_data)
