import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass
import jax
import jax.numpy as jnp
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.dtypes import DeviceType

C_in, C_out, H, K, N, W = 3, 4, 4, 2, 8, 4
np.set_printoptions(linewidth=100)


@dc.program
def conv2d(input: dc.float32[N, H, W, C_in], weights: dc.float32[K, K, C_in, C_out], S: dc.float32[1]):

    output = np.ndarray((N, H - K + 1, W - K + 1, C_out), dtype=np.float32)

    # Loop structure adapted from https://github.com/SkalskiP/ILearnDeepLearning.py/blob/ba0b5ba589d4e656141995e8d1a06d44db6ce58d/01_mysteries_of_neural_networks/06_numpy_convolutional_neural_net/src/layers/convolutional.py#L88
    # for i, j in dc.map[0:H-K+1, 0:W-K+1]:
    for i in range(H - K + 1):
        for j in range(W - K + 1):
            output[:, i, j, :] = np.sum(
                input[:, i:i + K, j:j + K, :, np.newaxis] * weights[np.newaxis, :, :, :],
                axis=(1, 2, 3),
            )

    S[0] = np.sum(output)


sdfg = conv2d.to_sdfg()
sdfg = auto_optimize(sdfg, DeviceType.CPU)
sdfg.save("log_sdfgs/conv2d_forward.sdfg")
assert False
add_backward_pass(sdfg=sdfg, inputs=["input"], outputs=["S"])

sdfg.save("log_sdfgs/conv2d_backward_works.sdfg")

input = np.random.default_rng(42).random((N, H, W, C_in), dtype=np.float32)
input_j = jnp.array(input)
weights = np.random.default_rng(42).random((K, K, C_in, C_out), dtype=np.float32)
weights_j = jnp.array(weights)
bias = np.random.default_rng(42).random((C_out, ), dtype=np.float32)
bias_j = jnp.array(bias)
S = np.zeros(shape=[1], dtype=np.float32)
S_j = jnp.array(S)
gradient_S = np.ones(shape=[1], dtype=np.float32)
gradient_input = np.zeros(shape=[N, H, W, C_in], dtype=np.float32)
sdfg(input, weights, S, gradient_input=gradient_input, gradient_S=gradient_S)


def jax_conv2d(input, weights):
    K = weights.shape[0]  # Assuming square kernel
    N = input.shape[0]
    H_out = input.shape[1] - K + 1
    W_out = input.shape[2] - K + 1
    C_out = weights.shape[3]
    output = jnp.empty((N, H_out, W_out, C_out), dtype=np.float32)

    # Loop structure adapted from https://github.com/SkalskiP/ILearnDeepLearning.py/blob/ba0b5ba589d4e656141995e8d1a06d44db6ce58d/01_mysteries_of_neural_networks/06_numpy_convolutional_neural_net/src/layers/convolutional.py#L88
    for i in range(H_out):
        for j in range(W_out):
            output = output.at[:, i, j, :].set(
                jnp.sum(
                    input[:, i:i + K, j:j + K, :, np.newaxis] * weights[np.newaxis, :, :, :],
                    axis=(1, 2, 3),
                ))
    return output


# JAX Program
def jax_kernel(input, weights, bias, S):
    A = jax_conv2d(input, weights)
    S = S.at[0].set(jnp.sum(A))
    return S[0]


jax_grad = jax.grad(jax_kernel, argnums=0)
assert np.allclose(input_j, input)
gradient_input_j = jax_grad(input_j, weights_j, bias_j, S_j)

# print the indices of the non zero valeus
# print(np.nonzero(gradient_input))

print((gradient_input_j - gradient_input)[1, 0, :, :])
assert np.allclose(gradient_input, gradient_input_j)
