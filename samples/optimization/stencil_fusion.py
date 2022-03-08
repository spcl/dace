# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import shutil
import os
import dace
from dace.dtypes import DeviceType
import numpy as np

from pathlib import Path

from dace.transformation.auto.auto_optimize import auto_optimize
from dace.optimization import cutout_tuner as ct
from dace import optimization as optim

if __name__ == '__main__':

    sdfg = dace.SDFG.from_file(Path(__file__).parent / "program.sdfg")
    #sdfg = randsdfg("stencil", max_stages=16, testing=True)
    auto_optimize(sdfg, DeviceType.CPU)

    cache_path = Path(__file__).parent / ".dacecache"
    shutil.rmtree(cache_path, ignore_errors=True)

    tuning_paths = Path(__file__).parent.rglob("*.tuning")
    for path in tuning_paths:
        os.remove(path)

    cutout_path = Path(__file__).parent / "cutouts"
    shutil.rmtree(cutout_path, ignore_errors=True)
    cutout_path.mkdir()

    arguments = {}
    for array in sdfg.arrays:
        if sdfg.arrays[array].transient:
            continue

        dtype = sdfg.arrays[array].dtype
        shape = sdfg.arrays[array].shape
    
        data = np.random.rand(*shape)
        arguments[array] = data

    result = ct.CutoutTuner.dry_run(sdfg, **arguments)


    print("Tuning")
    tuner = optim.StencilFusionTuner(sdfg)
    report = tuner.optimize()
    print(report)
