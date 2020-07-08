import numpy as np
import dace
from dace.libraries.onnx.nodes.onnx_op import ONNXConv

import torch


def test_expansion():
    sdfg = dace.SDFG("test_expansion")
    sdfg.add_array("X_arr", (5, 3, 10, 10), dace.float32)
    sdfg.add_array("W_arr", (16, 3, 3, 3), dace.float32)
    sdfg.add_array("Z_arr", (5, 16, 4, 4), dace.float32)

    state = sdfg.add_state()
    access_X = state.add_access("X_arr")
    access_W = state.add_access("W_arr")
    access_Z = state.add_access("Z_arr")

    c = ONNXConv("Conv", strides=[2, 2])

    state.add_node(c)
    state.add_edge(access_X, None, c, "X", sdfg.get_array_memlet("X_arr"))
    state.add_edge(access_W, None, c, "W", sdfg.get_array_memlet("W_arr"))
    state.add_edge(c, "Y", access_Z, None, sdfg.get_array_memlet("Z_arr"))

    X = np.random.rand(5, 3, 10, 10).astype(np.float32)
    W = np.random.rand(16, 3, 3, 3).astype(np.float32)
    Z = np.zeros((5, 16, 4, 4)).astype(np.float32)

    sdfg(X_arr=X, W_arr=W, Z_arr=Z)

    Z_t = torch.nn.functional.conv2d(torch.tensor(X), torch.tensor(W), stride=2)

    assert np.allclose(Z, Z_t)


if __name__ == '__main__':
    test_expansion()
