# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

W = dace.symbol('W')


@dace.program
def multi_output_scope(A, stats):

    @dace.map(_[0:W])
    def compute(i):
        inp << A[i]
        sum >> stats(1, lambda x, y: x + y)[0]
        ssq >> stats(1, lambda x, y: x + y)[1]

        sum = inp
        ssq = inp * inp


def test():
    W = 120

    A = dace.ndarray([W])
    stats = dace.ndarray([2])

    A[:] = np.random.normal(3.0, 5.0, W)
    stats[:] = 0.0

    multi_output_scope(A, stats, W=W)

    mean = stats[0] / W
    variance = stats[1] / W - mean * mean
    print("Mean: %f, Variance: %f" % (mean, variance))

    diff_mean = abs(mean - np.mean(A))
    print("Difference (mean):", diff_mean)
    diff_var = abs(variance - np.var(A))
    print("Difference (variance):", diff_var)
    assert diff_mean <= 1e-5 and diff_var <= 1e-4


if __name__ == "__main__":
    test()
