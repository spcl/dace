import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass

N, H, W, C_before_fc1, S0, S1, S2, S3, S4, S5 = (dc.symbol(
    s, dtype=dc.int64) for s in ('N', 'H', 'W', 'C_before_fc1', 'S0', 'S1',
                                 'S2', 'S3', 'S4', 'S5'))

@dc.program
def relu2(x: dc.float32[S0, S1]):
    return np.maximum(x, 0)


@dc.program
def relu4(x: dc.float32[S0, S1, S2, S3]):
    return np.maximum(x, 0)


# Deep learning convolutional operator (stride = 1)
@dc.program
def conv2d(input: dc.float32[S0, S1, S2, S3], weights: dc.float32[S4, S4, S3,
                                                                  S5]):
    # K = weights.shape[0]  # Assuming square kernel
    # N = input.shape[0]
    # H_out = input.shape[1] - K + 1
    # W_out = input.shape[2] - K + 1
    # C_out = weights.shape[3]
    # output = np.empty((N, H_out, W_out, C_out), dtype=np.float32)
    output = np.ndarray((S0, S1 - S4 + 1, S2 - S4 + 1, S5), dtype=np.float32)

    # Loop structure adapted from https://github.com/SkalskiP/ILearnDeepLearning.py/blob/ba0b5ba589d4e656141995e8d1a06d44db6ce58d/01_mysteries_of_neural_networks/06_numpy_convolutional_neural_net/src/layers/convolutional.py#L88
    # for i, j in dc.map[0:H-K+1, 0:W-K+1]:
    for i in range(S1 - S4 + 1):
        for j in range(S2 - S4 + 1):
            output[:, i, j, :] = np.sum(
                input[:, i:i + S4, j:j + S4, :, np.newaxis] *
                weights[np.newaxis, :, :, :],
                axis=(1, 2, 3),
            )
            # # TODO: View operations are needed
            # output[:, i, j, :] = np.sum(
            #     np.reshape(input[:, i:i+S4, j:j+S4, :], (S0, S4, S4, S3, 1)) *
            #     np.reshape(weights, (1, S4, S4, S3, S5)),
            #     axis=(1, 2, 3),
            # )

    return output


# 2x2 maxpool operator, as used in LeNet-5
@dc.program
def maxpool2d(x: dc.float32[S0, S1, S2, S3]):
    # output = np.empty(
    #     [x.shape[0], x.shape[1] // 2, x.shape[2] // 2, x.shape[3]],
    #     dtype=x.dtype)
    output = np.ndarray([S0, S1 // 2, S2 // 2, S3], dtype=np.float32)
    # for i in range(x.shape[1] // 2):
    #     for j in range(x.shape[2] // 2):
    for i in range(S1 // 2):
        for j in range(S2 // 2):
            output[:, i, j, :] = np.max(x[:, 2 * i:2 * i + 2,
                                          2 * j:2 * j + 2, :],
                                        axis=(1, 2))
    return output


# LeNet-5 Convolutional Neural Network (inference mode)
@dc.program
def lenet5(input: dc.float32[N, H, W, 1], conv1: dc.float32[5, 5, 1, 6],
           conv1bias: dc.float32[6], conv2: dc.float32[5, 5, 6, 16],
           conv2bias: dc.float32[16], fc1w: dc.float32[C_before_fc1, 120],
           fc1b: dc.float32[120], fc2w: dc.float32[120, 84],
           fc2b: dc.float32[84], fc3w: dc.float32[84,
                                                  10], fc3b: dc.float32[10], S:dc.float32[1]):


    x1 = relu4(conv2d(input, conv1) + conv1bias)
    x2 = maxpool2d(x1)
    x3 = relu4(conv2d(x2, conv2) + conv2bias)
    x4 = maxpool2d(x3)
    x5 = np.reshape(x4, (N, C_before_fc1))
    x6 = relu2(x5 @ fc1w + fc1b)
    x7 = relu2(x6 @ fc2w + fc2b)

    D = x7 @ fc3w + fc3b

    S[0] = np.sum(D)
    return D


sdfg = lenet5.to_sdfg()

sdfg.save("log_sdfgs/lenet_forward.sdfg")

add_backward_pass(sdfg=sdfg, inputs=["input"], outputs=["S"])

sdfg.save("log_sdfgs/lenet_backward.sdfg")
sdfg.compile()

# @dc.program
# def maxpool2d(x: dc.float32[S0, S1, S2, S3], S: dc.float64[1]):
#     # output = np.empty(
#     #     [x.shape[0], x.shape[1] // 2, x.shape[2] // 2, x.shape[3]],
#     #     dtype=x.dtype)
#     output = np.ndarray([S0, S1 // 2, S2 // 2, S3], dtype=np.float32)
#     # for i in range(x.shape[1] // 2):
#     #     for j in range(x.shape[2] // 2):
#     for i in range(S1 // 2):
#         for j in range(S2 // 2):
#             output[:, i, j, :] = np.max(x[:, 2 * i:2 * i + 2,
#                                           2 * j:2 * j + 2, :],
#                                         axis=(1, 2))
#     return output

# sdfg = maxpool2d.to_sdfg()

# sdfg.save("log_sdfgs/maxpool2d_forward.sdfg")

# add_backward_pass(sdfg=sdfg, inputs=["x"], outputs=["S"])

# sdfg.save("log_sdfgs/maxpool2d_backward.sdfg")
# sdfg.compile()
