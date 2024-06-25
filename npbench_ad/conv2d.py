import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass

C_in, C_out, H, K, N, W = 3, 4, 32, 32, 32, 32


@dc.program
def conv2d(input: dc.float32[N, H, W, C_in], weights: dc.float32[K, K, C_in,
                                                                 C_out], S: dc.float64[1]):

    output = np.ndarray((N, H - K + 1, W - K + 1, C_out), dtype=np.float32)

    # Loop structure adapted from https://github.com/SkalskiP/ILearnDeepLearning.py/blob/ba0b5ba589d4e656141995e8d1a06d44db6ce58d/01_mysteries_of_neural_networks/06_numpy_convolutional_neural_net/src/layers/convolutional.py#L88
    # for i, j in dc.map[0:H-K+1, 0:W-K+1]:
    for i in range(H - K + 1):
        for j in range(W - K + 1):
            output[:, i, j, :] = np.sum(
                input[:, i:i + K, j:j + K, :, np.newaxis] *
                weights[np.newaxis, :, :, :],
                axis=(1, 2, 3),
            )

    @dc.map(_[0:N, 0:H - K + 1, 0: W - K + 1, 0: C_out])
    def summap(i, j, k, l):
        s >> S(1, lambda x, y: x + y)[0]
        z << output[i, j, k, l]
        s = z


sdfg = conv2d.to_sdfg()

sdfg.save("log_sdfgs/conv2d_forward.sdfg")


add_backward_pass(sdfg=sdfg, inputs=["input"], outputs=["S"])

sdfg.save("log_sdfgs/conv2d_backward.sdfg")

