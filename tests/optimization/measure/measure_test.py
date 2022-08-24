import dace
import math
import numpy as np

from dace.optimization.measure import measure


@dace.program
def matmul(A: dace.float64[10, 10], B: dace.float64[10, 10], C: dace.float64[10, 10]):
    C[:] = A @ B


def test_measure():
    measurements = 3
    warmup = 2

    sdfg = matmul.to_sdfg()
    sdfg.simplify()

    A = np.random.rand(10, 10).astype(np.float64)
    B = np.random.rand(10, 10).astype(np.float64)
    C = np.zeros_like(B)

    arguments = {"A": A, "B": B, "C": C}
    target = A @ B

    runtime, process_time = measure(sdfg, arguments, measurements=measurements, warmup=warmup)
    assert runtime != math.inf and process_time != math.inf
    assert (runtime / 1000) < process_time

    report = sdfg.get_latest_report()
    durations = list(report.durations.values())[0]
    durations = list(durations.values())[0]
    durations = list(durations.values())[0]

    assert len(durations) == measurements + warmup
    assert np.median(np.array(durations[warmup:])) == runtime


def test_measure_exception():
    sdfg = matmul.to_sdfg()
    sdfg.simplify()

    A = np.random.rand(10, 10).astype(np.float64)
    B = np.random.rand(10, 10).astype(np.float64)

    arguments = {"A": A, "B": B}

    runtime, process_time = measure(sdfg, arguments, measurements=1)
    assert runtime == math.inf
    assert process_time != math.inf


if __name__ == '__main__':
    test_measure()
