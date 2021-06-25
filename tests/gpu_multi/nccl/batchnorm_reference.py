import numpy as np
import dace as dc

N, H, W, C1, C2, S0, S1, S2, S3, S4, S5 = (dc.symbol(s, dtype=dc.int64)
                                           for s in ('N', 'H', 'W', 'C1', 'C2',
                                                     'S0', 'S1', 'S2', 'S3',
                                                     'S4', 'S5'))
dc_dtype = dc.float32
np_dtype = np.float32


@dc.program
def relu(x: dc.float32[S0, S1, S2, S3]):
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

    return output


@dc.program
def batchnorm2d(x: dc_dtype[S0, S1, S2, S3]):
    # mean = np.mean(x, axis=0, keepdims=True)
    mean = np.ndarray((1, S1, S2, S3), dtype=np.float32)
    mean[:] = np.mean(x, axis=0)
    # std = np.std(x, axis=0, keepdims=True)
    std = np.ndarray((1, S1, S2, S3), dtype=np.float32)
    # std[:] = np.sqrt(np.sum((x - mean) ** 2, axis=0) / np.float32(S0))
    std[:] = np.sqrt(np.sum((x - mean) * (x - mean), axis=0) / np.float32(S0))
    # return (x - mean) / np.sqrt(std + eps)
    return (x - mean) / np.sqrt(std + 1e-5)


@dc.program
def resnet_basicblock_gpu(out: dc.float32[N, H, W,
                                          C1], input: dc.float32[N, H, W, C1],
                          conv1: dc.float32[1, 1, C1, C2],
                          conv2: dc.float32[3, 3, C2,
                                            C2], conv3: dc.float32[1, 1, C2,
                                                                   C1]):
    # Pad output of first convolution for second convolution
    # padded = np.zeros((input.shape[0], input.shape[1] + 2, input.shape[2] + 2,
    #                    conv1.shape[3]))
    padded = np.ndarray((N, H + 2, W + 2, C2), dtype=np.float32)
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
    return relu(x6 + input)


def test_batchnorm2d():
    bnsdfg: dc.SDFG = batchnorm2d.to_sdfg()
    bnsdfg.view()
    rnsdfg: dc.SDFG = resnet_basicblock_gpu.to_sdfg()
    rnsdfg.view()


if __name__ == "__main__":
    test_batchnorm2d()