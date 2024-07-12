import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass

N1, N2 = (dc.symbol(s, dtype=dc.int64) for s in ('N1', 'N2'))
C_in, N, S0, S1, S2 = 3, 8, 30000, 2000, 2000


@dc.program
def relu(x: dc.float64[N1, N2]):
    return np.maximum(x, 0)


# Numerically-stable version of softmax
@dc.program
def softmax(x: dc.float64[N1, N2]):
    # tmp_max = np.max(x, axis=-1, keepdims=True)
    tmp_max = np.maximum.reduce(x, axis=-1, keepdims=True)
    tmp_out = np.exp(x - tmp_max)
    # tmp_sum = np.sum(tmp_out, axis=-1, keepdims=True)
    tmp_sum = np.add.reduce(tmp_out, axis=-1, keepdims=True)
    return tmp_out / tmp_sum


# 3-layer MLP
@dc.program
def mlp(input: dc.float64[N, C_in], w1: dc.float64[C_in, S0], b1: dc.float64[S0], w2: dc.float64[S0, S1],
        b2: dc.float64[S1], w3: dc.float64[S1, S2], b3: dc.float64[S2], S: dc.float64[1]):
    x1 = relu(input @ w1 + b1)
    x2 = relu(x1 @ w2 + b2)
    x3 = softmax(x2 @ w3 + b3)  # Softmax call can be omitted if necessary
    S[0] = np.sum(x3)

    return x3


def initialize(C_in, N, S0, S1, S2):
    from numpy.random import default_rng
    rng = default_rng(42)

    mlp_sizes = [S0, S1, S2]  # [300, 100, 10]
    # Inputs
    input = np.random.rand(N, C_in).astype(np.float64)
    # Weights
    w1 = rng.random((C_in, mlp_sizes[0]), dtype=np.float64)
    b1 = rng.random((mlp_sizes[0], ), dtype=np.float64)
    w2 = rng.random((mlp_sizes[0], mlp_sizes[1]), dtype=np.float64)
    b2 = rng.random((mlp_sizes[1], ), dtype=np.float64)
    w3 = rng.random((mlp_sizes[1], mlp_sizes[2]), dtype=np.float64)
    b3 = rng.random((mlp_sizes[2], ), dtype=np.float64)

    return input, w1, b1, w2, b2, w3, b3


input, w1, b1, w2, b2, w3, b3 = initialize(C_in, N, S0, S1, S2)
S = np.zeros(shape=[1])
gradient_S = np.ones(shape=[1])
gradient_input = np.zeros(shape=[N, C_in])

sdfg = mlp.to_sdfg()

sdfg(input, w1, b1, w2, b2, w3, b3, S)
# sdfg.save("log_sdfgs/mlp_forward.sdfg")

# add_backward_pass(sdfg=sdfg, inputs=["input"], outputs=["S"])

# sdfg.save("log_sdfgs/mlp_backward.sdfg")

# sdfg(input,
#      w1,
#      b1,
#      w2,
#      b2,
#      w3,
#      b3,
#      S,
#      gradient_input=gradient_input,
#      gradient_S=gradient_S,
#      N=N,
#      S0=S0,
#      S1=S1,
#      S2=S2,
#      C_in=C_in)

# # JAX
# import jax
# import jax.numpy as jnp

# def relu_jax(x: dc.float64[N1, N2]):
#     return jnp.maximum(x, 0)

# def softmax_jax(x: dc.float64[N1, N2]):
#     # tmp_max = jnp.max(x, axis=-1, keepdims=True)
#     tmp_max = jnp.maximum.reduce(x, axis=-1, keepdims=True)
#     tmp_out = jnp.exp(x - tmp_max)
#     # tmp_sum = jnp.sum(tmp_out, axis=-1, keepdims=True)
#     tmp_sum = jnp.add.reduce(tmp_out, axis=-1, keepdims=True)
#     return tmp_out / tmp_sum

# def mlp_jax(input: dc.float64[N, C_in], w1: dc.float64[C_in, S0], b1: dc.float64[S0], w2: dc.float64[S0, S1],
#             b2: dc.float64[S1], w3: dc.float64[S1, S2], b3: dc.float64[S2], S: dc.float64[1]):
#     x1 = relu_jax(input @ w1 + b1)
#     x2 = relu_jax(x1 @ w2 + b2)
#     x3 = softmax_jax(x2 @ w3 + b3)  # Softmax call can be omitted if necessary

#     @dc.map(_[0:N, 0:N])
#     def summap(i, j):
#         s >> S(1, lambda x, y: x + y)[0]
#         z << x3[i, j]
#         s = z

#     return x3

# jax_grad = jax.grad(mlp_jax, argnums=[0])
# gradient_A_jax = jax_grad(input, w1, b1, w2, b2, w3, b3)
# assert np.allclose(gradient_A_jax, gradient_input)
