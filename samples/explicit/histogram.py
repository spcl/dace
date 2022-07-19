# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" 2D histogram sample that showcases memlets with write-conflict resolution and unknown element. """
import argparse
import dace
import math
import numpy as np

W = dace.symbol('W')
H = dace.symbol('H')
BINS = 256

@dace.program
def histogram(A: dace.float32[H, W], hist: dace.uint32[BINS]):
    for i, j in dace.map[0:H, 0:W]:
        with dace.tasklet:
            # Reading is performed normally
            a << A[i, j]
            # Writing to `out` is always done once (hence 1 in the first volume argument),
            # but to one of the locations between 0 and BINS. Also, writing to a certain
            # location would be conflicted with other tasklets in the map, so a summation
            # write-conflict resolution is used
            out >> hist(1, lambda x, y: x + y)[:]

            # The memlet `out` is an array. Writing the value 1 onto `out` would add 1
            # in the appropriate index
            out[min(int(a * BINS), BINS - 1)] = 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('W', type=int, nargs='?', default=32)
    parser.add_argument('H', type=int, nargs='?', default=32)
    args = parser.parse_args()

    W = args.W
    H = args.H

    # Initialize arrays
    A = np.random.rand(H, W).astype(dace.float32.type)
    hist = np.zeros([BINS], dtype=np.uint32)

    # Call dace program
    histogram(A, hist)

    # Check for correctness by calling numpy.histogram with the right arguments
    diff = np.linalg.norm(np.histogram(A, bins=BINS, range=(0.0, 1.0))[0][1:-1] - hist[1:-1])
    print('Difference:', diff)
    exit(0 if diff <= 1e-5 else 1)
