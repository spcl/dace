# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import argparse
import numpy as np
from veclen_conversion_test import SIZE, VECTOR_LENGTH, make_sdfg
from dace.fpga_testing import fpga_test


@fpga_test()
def test_veclen_conversion_connector():

    size = 128
    vector_length = 4

    SIZE.set(size)
    VECTOR_LENGTH.set(vector_length)

    if size % vector_length != 0:
        raise ValueError("Size {} must be divisible by vector length {}.".format(size, vector_length))

    sdfg = make_sdfg(name="veclen_conversion_connector", vectorize_connector=True)
    sdfg.specialize({"W": vector_length})

    A = np.arange(size, dtype=np.float64)
    B = np.zeros((size, ), dtype=np.float64)

    sdfg(A=A, B=B, N=SIZE)

    mid = vector_length // 2

    for i in range(size // vector_length):
        expected = np.concatenate(
            (A[i * vector_length + mid:(i + 1) * vector_length], A[i * vector_length:i * vector_length + mid]))
        if any(B[i * vector_length:(i + 1) * vector_length] != expected):
            raise ValueError("Shuffle failed: {} (should be {})".format(B, expected))

    return sdfg


if __name__ == "__main__":
    test_veclen_conversion_connector(None)
