import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
# N
sizes = {'S': (8, 14, 14, 32, 8), 'M': (8, 28, 28, 64, 16), 'L': (8, 56, 56, 128, 32), 'paper': (8, 56, 56, 256, 64)}

N, H, W, C1, C2, S0, S1, S2, S3, S4, S5 = (dc.symbol(s, dtype=dc.int64)
                                           for s in ('N', 'H', 'W', 'C1', 'C2', 'S0', 'S1', 'S2', 'S3', 'S4', 'S5'))


@dc.program
def relu(x: dc.float64[S0, S1, S2, S3]):
    return np.maximum(x, 0)


# Deep learning convolutional operator (stride = 1)
@dc.program
def conv2d(input: dc.float64[S0, S1, S2, S3], weights: dc.float64[S4, S4, S3, S5]):
    # K = weights.shape[0]  # Assuming square kernel
    # N = input.shape[0]
    # H_out = input.shape[1] - K + 1
    # W_out = input.shape[2] - K + 1
    # C_out = weights.shape[3]
    # output = np.empty((N, H_out, W_out, C_out), dtype=np.float64)
    output = np.ndarray((S0, S1 - S4 + 1, S2 - S4 + 1, S5), dtype=np.float64)

    # Loop structure adapted from https://github.com/SkalskiP/ILearnDeepLearning.py/blob/ba0b5ba589d4e656141995e8d1a06d44db6ce58d/01_mysteries_of_neural_networks/06_numpy_convolutional_neural_net/src/layers/convolutional.py#L88
    # for i, j in dc.map[0:H-K+1, 0:W-K+1]:
    for i in range(S1 - S4 + 1):
        for j in range(S2 - S4 + 1):
            output[:, i, j, :] = np.sum(
                input[:, i:i + S4, j:j + S4, :, np.newaxis] * weights[np.newaxis, :, :, :],
                axis=(1, 2, 3),
            )

    return output


# Batch normalization operator, as used in ResNet
@dc.program
def batchnorm2d(x: dc.float64[S0, S1, S2, S3]):
    # mean = np.mean(x, axis=0, keepdims=True)
    mean = np.ndarray((1, S1, S2, S3), dtype=np.float64)
    mean[:] = np.mean(x, axis=0)
    # std = np.std(x, axis=0, keepdims=True)
    std = np.ndarray((1, S1, S2, S3), dtype=np.float64)
    # std[:] = np.sqrt(np.sum((x - mean) ** 2, axis=0) / np.float64(S0))
    std[:] = np.sqrt(np.sum((x - mean) * (x - mean), axis=0) / np.float64(S0))
    # return (x - mean) / np.sqrt(std + eps)
    return (x - mean) / np.sqrt(std + 1e-5)


# Bottleneck residual block (after initial convolution, without downsampling)
# in the ResNet-50 CNN (inference)
@dc.program
def resnet_basicblock(input: dc.float64[N, H, W, C1], conv1: dc.float64[1, 1, C1, C2], conv2: dc.float64[3, 3, C2, C2],
                      conv3: dc.float64[1, 1, C2, C1]):
    # Pad output of first convolution for second convolution
    # padded = np.zeros((input.shape[0], input.shape[1] + 2, input.shape[2] + 2,
    #                    conv1.shape[3]))
    padded = np.zeros((N, H + 2, W + 2, C2), dtype=np.float64)

    padded[:, 1:-1, 1:-1, :] = conv2d(input, conv1)
    x = batchnorm2d(padded)
    # x = relu(x)

    # x = conv2d(x, conv2)
    # x = batchnorm2d(x)
    # x = relu(x)
    # x = conv2d(x, conv3)
    # x = batchnorm2d(x)
    # return relu(x + input)
    x1 = relu(x)

    x2 = conv2d(x1, conv2)
    x3 = batchnorm2d(x2)
    x4 = relu(x3)
    x5 = conv2d(x4, conv3)
    x6 = batchnorm2d(x5)
    return relu(x6 + input)


# Bottleneck residual block (after initial convolution, without downsampling)
# in the ResNet-50 CNN (inference)
@dc.program
def resnet_basicblock_gpu(input: dc.float64[N, H, W, C1], conv1: dc.float64[1, 1, C1, C2],
                          conv2: dc.float64[3, 3, C2, C2], conv3: dc.float64[1, 1, C2, C1], S: dc.float64[1]):
    # Pad output of first convolution for second convolution
    # padded = np.zeros((input.shape[0], input.shape[1] + 2, input.shape[2] + 2,
    #                    conv1.shape[3]))
    padded = np.ndarray((N, H + 2, W + 2, C2), dtype=np.float64)
    padded[:] = 0

    # padded[:, 1:-1, 1:-1, :] = conv2d(input, conv1)
    padded[:, 1:H + 1, 1:W + 1, :] = conv2d(input, conv1)
    x = batchnorm2d(padded)
    # x = relu(x)

    # x = conv2d(x, conv2)
    # x = batchnorm2d(x)
    # x = relu(x)
    # x = conv2d(x, conv3)
    # x = batchnorm2d(x)
    # return relu(x + input)
    x1 = relu(x)

    x2 = conv2d(x1, conv2)
    x3 = batchnorm2d(x2)
    x4 = relu(x3)
    x5 = conv2d(x4, conv3)
    x6 = batchnorm2d(x5)
    x7 = relu(x6 + input)
    S[0] = np.sum(x7)
    return x7


# sdfg = resnet_basicblock_gpu.to_sdfg()

# sdfg.save("log_sdfgs/resnet_forward.sdfg")

# add_backward_pass(sdfg=sdfg, inputs=["input"], outputs=["S"], autooptimize=False)

# load the sdfg
sdfg = dc.SDFG.from_file("log_sdfgs/resnet_backward.sdfg")
# sdfg.save("log_sdfgs/resnet_backward.sdfg")
sdfg.simplify()


def fwd_initialize(size, datatype=np.float64):
    N, W, H, C1, C2 = size
    from numpy.random import default_rng
    rng = default_rng(42)

    # Input
    input = rng.random((N, H, W, C1), dtype=np.float64)
    # Weights
    conv1 = rng.random((1, 1, C1, C2), dtype=np.float64)
    conv2 = rng.random((3, 3, C2, C2), dtype=np.float64)
    conv3 = rng.random((1, 1, C2, C1), dtype=np.float64)
    S = np.zeros(shape=[1], dtype=datatype)
    # Prepare the sizes args dict
    keyword_args = {"N": N, "W": W, "H": H, "C1": C1, "C2": C2}

    return (input, conv1, conv2, conv3, S), keyword_args


def bwd_initialize(size, datatype=np.float64):
    N, W, H, C1, C2 = size
    fwd_args, fwd_keyword_args = fwd_initialize(size)
    gradient_input = np.zeros(shape=(N, H, W, C1), dtype=datatype)
    gradient_S = np.ones(shape=[1], dtype=datatype)

    # Prepare the sizes args dict
    fwd_keyword_args.update({f"gradient_input": gradient_input, "gradient_S": gradient_S})

    return fwd_args, fwd_keyword_args


# fwd_args, fwd_keyword_args = fwd_initialize(sizes['paper'])

fwd_args, fwd_keyword_args = bwd_initialize(sizes['S'])
sdfg(*fwd_args, **fwd_keyword_args)

gradient_input = fwd_keyword_args["gradient_input"]


# JAX Program
def jax_relu(x):
    return jnp.maximum(x, 0)


# Deep learning convolutional operator (stride = 1)
def jax_conv2d(input, weights):
    K = weights.shape[0]  # Assuming square kernel
    N = input.shape[0]
    H_out = input.shape[1] - K + 1
    W_out = input.shape[2] - K + 1
    C_out = weights.shape[3]
    output = jnp.empty((N, H_out, W_out, C_out), dtype=np.float64)

    # Loop structure adapted from https://github.com/SkalskiP/ILearnDeepLearning.py/blob/ba0b5ba589d4e656141995e8d1a06d44db6ce58d/01_mysteries_of_neural_networks/06_numpy_convolutional_neural_net/src/layers/convolutional.py#L88
    for i in range(H_out):
        for j in range(W_out):
            output = output.at[:, i, j, :].set(
                jnp.sum(
                    input[:, i:i + K, j:j + K, :, jnp.newaxis] * weights[jnp.newaxis, :, :, :],
                    axis=(1, 2, 3),
                ))

    return output


# Batch normalization operator, as used in ResNet
def jax_batchnorm2d(x, eps=1e-5):
    mean = jnp.mean(x, axis=0, keepdims=True)
    std = jnp.std(x, axis=0, keepdims=True)
    return (x - mean) / jnp.sqrt(std + eps)


# Bottleneck residual block (after initial convolution, without downsampling)
# in the ResNet-50 CNN (inference)
def jax_kernel(input, conv1, conv2, conv3, S):
    # Pad output of first convolution for second convolution
    padded = jnp.zeros((input.shape[0], input.shape[1] + 2, input.shape[2] + 2, conv1.shape[3]))

    padded = padded.at[:, 1:-1, 1:-1, :].set(jax_conv2d(input, conv1))
    x = jax_batchnorm2d(padded)
    x = jax_relu(x)

    x = jax_conv2d(x, conv2)
    x = jax_batchnorm2d(x)
    x = jax_relu(x)
    x = jax_conv2d(x, conv3)
    x = jax_batchnorm2d(x)
    return jnp.sum(jax_relu(x + input))


def numpy_array_to_jnp(dace_inputs):
    """
    Function to transform a set of numpy arrays to jax arrays
    """

    def convert_element(element):
        if isinstance(element, np.ndarray):
            return jnp.array(element)
        elif isinstance(element, (int, float, complex, np.float64, np.float64)):
            return element
        else:
            raise TypeError(f"Unsupported type {type(element)}")

    return tuple(convert_element(element) for element in dace_inputs)


def jax_initialize(size, datatype=np.float64):
    forward_inputs, _ = fwd_initialize(size)
    return numpy_array_to_jnp(forward_inputs)


jax_grad = jax.grad(jax_kernel, argnums=0)

forward_inputs = jax_initialize(sizes['S'])
gradient_input_jax = jax_grad(*forward_inputs)

print(gradient_input_jax - gradient_input)
assert np.allclose(gradient_input_jax, gradient_input, atol=1e-6)
