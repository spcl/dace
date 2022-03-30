# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import os
import dace
import numpy as np

from pathlib import Path

from dace import optimization as optim
from dace.sdfg import infer_types

if __name__ == '__main__':

    sdfg_path = Path(os.environ["HOME"]) / "projects/tuning-dace/hg-fvt-384.sdfg"
    small_sample = dace.SDFG.from_file(sdfg_path)

    infer_types.infer_connector_types(small_sample)
    infer_types.set_default_schedule_and_storage_types(small_sample, None)

    sdfg_path = Path(os.environ["HOME"]) / "projects/tuning-dace/hG-prepared-384.sdfg"
    big_sdfg = dace.SDFG.from_file(sdfg_path)

    infer_types.infer_connector_types(big_sdfg)
    infer_types.set_default_schedule_and_storage_types(big_sdfg, None)

    dreport = big_sdfg.get_instrumented_data()
    if dreport is None:
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
    optim.OnTheFlyMapFusionTuner.transfer(big_sdfg, otf_tuner, k=2)
    
    big_sdfg.save(Path(os.environ["HOME"]) / "projects/tuning-dace/hG-384-otf-transfer.sdfg")
    otf_tuner.optimize(apply=True)

    sf_tuner = optim.SubgraphFusionTuner(small_sample, measurement=dace.InstrumentationType.GPU_Events)
    optim.SubgraphFusionTuner.transfer(big_sdfg, sf_tuner, k=1)

    sdfg_path = Path(os.environ["HOME"]) / "projects/tuning-dace/hG-384-SGF-transfer.sdfg"
    big_sdfg.save(sdfg_path)
