import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass
N = 32
M = 32

C_in, N, S0, S1, S2, N1, N2 = 32, 32, 32, 32, 32, 32, 32


@dc.program
def relu(x: dc.float32[N1, N2]):
    return np.maximum(x, 0)


# Numerically-stable version of softmax
@dc.program
def softmax(x: dc.float32[N1, N2]):
    # tmp_max = np.max(x, axis=-1, keepdims=True)
    tmp_max = np.maximum.reduce(x, axis=-1, keepdims=True)
    tmp_out = np.exp(x - tmp_max)
    # tmp_sum = np.sum(tmp_out, axis=-1, keepdims=True)
    tmp_sum = np.add.reduce(tmp_out, axis=-1, keepdims=True)
    return tmp_out / tmp_sum


# 3-layer MLP
@dc.program
def mlp(input: dc.float32[N, C_in], w1: dc.float32[C_in, S0],
        b1: dc.float32[S0], w2: dc.float32[S0, S1], b2: dc.float32[S1],
        w3: dc.float32[S1, S2], b3: dc.float32[S2], S: dc.float64[1]):
    x1 = relu(input @ w1 + b1)
    x2 = relu(x1 @ w2 + b2)
    x3 = softmax(x2 @ w3 + b3)  # Softmax call can be omitted if necessary

    @dc.map(_[0:N, 0:N])
    def summap(i, j):
        s >> S(1, lambda x, y: x + y)[0]
        z << x3[i, j]
        s = z
    return x3



sdfg = mlp.to_sdfg()

sdfg.save("log_sdfgs/mlp_forward.sdfg")


add_backward_pass(sdfg=sdfg, inputs=["input"], outputs=["S"])

sdfg.save("log_sdfgs/mlp_backward.sdfg")