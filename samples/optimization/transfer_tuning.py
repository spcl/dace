# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import os
import dace
import numpy as np

from pathlib import Path

from dace import optimization as optim
from dace.optimization import utils as optim_utils

if __name__ == '__main__':

    sdfg_path = Path(os.environ["HOME"]) / "projects/tuning-dace/aha-gpu.sdfg"
    small_sample = dace.SDFG.from_file(sdfg_path)
 
    sdfg_path = Path(os.environ["HOME"]) / "projects/tuning-dace/hG-transient.sdfg"
    big_sdfg = dace.SDFG.from_file(sdfg_path)

    dreport = big_sdfg.get_instrumented_data()
    if dreport is None:
        exit()
        arguments = {}
        for name, array in big_sdfg.arrays.items():
            if array.transient:
                continue

            data = dace.data.make_array_from_descriptor(array, np.random.rand(*array.shape))
            arguments[name] = data

        optim.CutoutTuner.dry_run(big_sdfg, **arguments)
        print("Goodbye")
        exit()

    print("We got it")

    otf_tuner = optim.OnTheFlyMapFusionTuner(small_sample, measurement=dace.InstrumentationType.GPU_Events)
    optim.OnTheFlyMapFusionTuner.transfer(big_sdfg, otf_tuner, k=5)
    
    big_sdfg.save(Path(os.environ["HOME"]) / "projects/tuning-dace/hG-transient-otf-transfer.sdfg")
    otf_tuner.optimize(apply=True)

    sf_tuner = optim.SubgraphFusionTuner(small_sample, measurement=dace.InstrumentationType.GPU_Events)
    optim.SubgraphFusionTuner.transfer(big_sdfg, sf_tuner, k=5)

    sdfg_path = Path(os.environ["HOME"]) / "projects/tuning-dace/hG-transient-subgraph-transfer.sdfg"
    big_sdfg.save(sdfg_path)
