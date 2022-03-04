# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import math
import dace
import dace.performance as perf

import numpy as np

from pathlib import Path

N = dace.symbol("N")

machine_file = Path(__file__).parent / "SkylakeSP_Gold-6148.yml"

@dace.program
def sample(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
    for j, i in dace.map[1:N - 1, 1:N - 1]:
        with dace.tasklet:
            a << A[i - 1:i + 2, j - 1:j + 2]
            b >> B[i, j]

            b = a[0, 0] + a[0, 1] + a[0, 2] + a[1, 0] + a[1, 1] + a[1, 2] + a[2, 0] + a[2, 1] + a[2, 2]

    for i, j in dace.map[0:N, 0:N]:
        with dace.tasklet:
            a << A[i, j]
            c >> C[i, j]

            c = math.log(a)


if __name__ == '__main__':
    sdfg = sample.to_sdfg()

    model = perf.RooflineModel(machine_file_path=machine_file)
    kernels = model.kernels(sdfg)

    values = {"N": 1024}
    for kernel in kernels:
        report = model.analyze(kernel, values)

        print(report)    
