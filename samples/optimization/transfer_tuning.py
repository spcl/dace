# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import os
import dace
import numpy as np

from pathlib import Path

from dace.optimization import cutout_tuner as ct
from dace import optimization as optim

if __name__ == '__main__':

    sdfg_path = Path(os.environ["HOME"]) / "projects/tuning-dace/aha-expanded-2.sdfg"
    small_sample = dace.SDFG.from_file(sdfg_path)

    small_sample.apply_gpu_transformations()
    small_sample.simplify()

    arguments = {}
    for name, array in small_sample.arrays.items():
        if array.transient:
            continue

        data = dace.data.make_array_from_descriptor(array, np.random.rand(*array.shape))
        arguments[name] = data
    
    ct.CutoutTuner.dry_run(small_sample, **arguments)
    otf_tuner = optim.OnTheFlyMapFusionTuner(small_sample, measurement=dace.InstrumentationType.GPU_Events)
    otf_tuner.optimize(apply=False)

    sf_tuner = optim.SubgraphFusionTuner(small_sample, measurement=dace.InstrumentationType.GPU_Events)
    sf_tuner.optimize(apply=False)

    sdfg_path = Path(os.environ["HOME"]) / "projects/tuning-dace/aha-expanded-2.sdfg"
    big_sdfg = dace.SDFG.from_file(sdfg_path)

    print("Transfering")
    
    otf_tuner.transfer(big_sdfg)
    sf_tuner.transfer(big_sdfg)

    sdfg_path = Path(os.environ["HOME"]) / "projects/tuning-dace/aha-expanded-2-transfer.sdfg"
    big_sdfg.save(sdfg_path)
