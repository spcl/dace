# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import math
import dace
from dace import optimization as optim
from dace.optimization.cutout_tuner import CutoutTuner
import numpy as np

from dace.transformation.auto.auto_optimize import auto_optimize
from dace.optimization import data_layout_tuner

N = 256


@dace.program
def sample(A, B, C):
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
    A = np.random.rand(N, N)
    B = np.random.rand(N, N)
    C = np.random.rand(N, N)

    sdfg = sample.to_sdfg(A, B, C)
    auto_optimize(sdfg, dace.DeviceType.CPU)

    CutoutTuner.dry_run(sdfg, A, B, C)

    tuner = optim.MapPermutationTuner(sdfg)
    dist = optim.DistributedCutoutTuner(tuner=tuner)
    dist.optimize()

    tuner = optim.MapTilingTuner(sdfg)
    dist = optim.DistributedCutoutTuner(tuner)
    dist.optimize()

    tuner = optim.DataLayoutTuner(sdfg)
    dist = optim.DistributedCutoutTuner(tuner)
    dist.optimize(group_by=data_layout_tuner.TuningGroups.Inputs_Outputs)
