#!/usr/bin/env python
import dace
import numpy as np

W = dace.symbol()


@dace.program
def prog(A, stats):
    @dace.map(_[0:W])
    def compute(i):
        inp << A[i]
        sum >> stats(1, lambda x, y: x + y, 0)[0]
        ssq >> stats(1, lambda x, y: x + y, 0)[1]

        sum = inp
        ssq = inp * inp


if __name__ == '__main__':
    W.set(120)

    A = dace.ndarray([W])
    stats = dace.ndarray([2])

    A[:] = np.random.normal(3.0, 5.0, W.get())

    prog(A, stats)

    mean = stats[0] / W.get()
    variance = stats[1] / W.get() - mean * mean
    print("Mean: %f, Variance: %f" % (mean, variance))

    diff_mean = abs(mean - np.mean(A))
    print("Difference (mean):", diff_mean)
    diff_var = abs(variance - np.var(A))
    print("Difference (variance):", diff_var)
    print("==== Program end ====")
    exit(0 if diff_mean <= 1e-5 and diff_var <= 1e-4 else 1)
