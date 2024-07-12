import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass

C_in, C_out, H, K, N, W = 3, 16, 32, 2, 8, 32


@dc.program
def conv2d(input: dc.float32[N, H, W, C_in], weights: dc.float32[K, K, C_in, C_out], S: dc.float64[1]):

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

sdfg.save("log_sdfgs/conv2d_forward.sdfg")

add_backward_pass(sdfg=sdfg, inputs=["input"], outputs=["S"])

sdfg.save("log_sdfgs/conv2d_backward.sdfg")

input = np.random.default_rng(42).random((N, H, W, C_in), dtype=np.float32)
weights = np.random.default_rng(42).random((K, K, C_in, C_out), dtype=np.float32)
bias = np.random.default_rng(42).random((C_out, ), dtype=np.float32)
S = np.zeros(shape=[1])
gradient_S = np.ones(shape=[1])
gradient_input = np.zeros(shape=[N, H, W, C_in])
sdfg(input, weights, S, gradient_input=gradient_input, gradient_S=gradient_S)

print(gradient_input)
