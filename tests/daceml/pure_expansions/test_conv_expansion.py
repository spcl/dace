import pytest
import dace
import dace.libraries.onnx as donnx
import torch
import torch.nn.functional as F
import numpy as np


@pytest.mark.parametrize("num_in_channels, kernel_size, num_filters, bias",
                         [(1, (3, 3), 8, True), (8, (3, 3), 3, False), (8, (5, 5), 3, True), (8, (4, 4), 3, False)])
def test_conv_simple(num_in_channels, kernel_size, num_filters, bias):

    batch_size = 8

    X = np.random.rand(batch_size, num_in_channels, 32, 32).astype(np.float32)
    W = np.random.rand(num_filters, num_in_channels, *kernel_size).astype(np.float32)

    if bias:
        B = np.random.rand(num_filters).astype(np.float32)
        torch_Z = F.conv2d(torch.from_numpy(X), torch.from_numpy(W), bias=torch.from_numpy(B)).numpy()
    else:
        B = None
        torch_Z = F.conv2d(torch.from_numpy(X), torch.from_numpy(W)).numpy()

    dace_Z = np.zeros_like(torch_Z)

    if bias:

        @dace.program
        def conv(X_: dace.float32[tuple(X.shape)], W_: dace.float32[tuple(W.shape)], B_: dace.float32[tuple(B.shape)],
                 Z_: dace.float32[tuple(torch_Z.shape)]):
            donnx.ONNXConv(X=X_, W=W_, B=B_, Y=Z_)
    else:

        @dace.program
        def conv(X_: dace.float32[tuple(X.shape)], W_: dace.float32[tuple(W.shape)],
                 Z_: dace.float32[tuple(torch_Z.shape)]):
            donnx.ONNXConv(X=X_, W=W_, Y=Z_)

    sdfg = conv.to_sdfg()
    sdfg.expand_library_nodes()

    if bias:
        sdfg(X_=X, W_=W, Z_=dace_Z, B_=B)
    else:
        sdfg(X_=X, W_=W, Z_=dace_Z)

    print(torch_Z - dace_Z)
    assert np.allclose(torch_Z, dace_Z)
