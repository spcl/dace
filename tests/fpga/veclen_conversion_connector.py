# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import argparse
import numpy as np
from veclen_conversion import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-size", type=int, default=128)
    parser.add_argument("-vector_length", type=int, default=4)
    args = parser.parse_args()

    SIZE.set(args.size)
    VECTOR_LENGTH.set(args.vector_length)

    if args.size % args.vector_length != 0:
        raise ValueError(
            "Size {} must be divisible by vector length {}.".format(
                args.size, args.vector_length))

    sdfg = make_sdfg(name="veclen_conversion_connector",
                     vectorize_connector=True)
    sdfg.specialize({"W": args.vector_length})

    A = np.arange(args.size, dtype=np.float64)
    B = np.zeros((args.size, ), dtype=np.float64)

    sdfg(A=A, B=B, N=SIZE)

    mid = args.vector_length // 2

    for i in range(args.size // args.vector_length):
        expected = np.concatenate(
            (A[i * args.vector_length + mid:(i + 1) * args.vector_length],
             A[i * args.vector_length:i * args.vector_length + mid]))
        if any(B[i * args.vector_length:(i + 1) *
                 args.vector_length] != expected):
            raise ValueError("Shuffle failed: {} (should be {})".format(
                B, expected))
