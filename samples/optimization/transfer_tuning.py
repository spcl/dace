# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import os
import dace
import numpy as np

from pathlib import Path

from dace import optimization as optim
from dace.optimization import utils as optim_utils
from dace.optimization import cutout_tuner as ct

if __name__ == '__main__':

    sdfg_path = Path(os.environ["HOME"]) / "projects/tuning-dace/aha.sdfg"
    small_sample = dace.SDFG.from_file(sdfg_path)

    small_sample.apply_gpu_transformations()
    small_sample.simplify()

    for state in small_sample.nodes():
        state.instrument = dace.InstrumentationType.GPU_Events
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                node.schedule = dace.ScheduleType.GPU_Device

    
    sdfg_path = Path(os.environ["HOME"]) / "projects/tuning-dace/hF2-transient.sdfg"
    big_sdfg = dace.SDFG.from_file(sdfg_path)

    dreport = big_sdfg.get_instrumented_data()
    if dreport is None:
        arguments = {}
        for name, array in big_sdfg.arrays.items():
            if array.transient:
                continue

            data = dace.data.make_array_from_descriptor(array, np.random.rand(*array.shape))
            arguments[name] = data

        ct.CutoutTuner.dry_run(big_sdfg, **arguments)
        print("Goodbye")
        exit()

    print("We got it")

    otf_tuner = optim.OnTheFlyMapFusionTuner(small_sample, measurement=dace.InstrumentationType.GPU_Events)
    optim.OnTheFlyMapFusionTuner.transfer(big_sdfg, otf_tuner, k=7)
    
    big_sdfg.save(Path(os.environ["HOME"]) / "projects/tuning-dace/hF2-transient-otf-transfer.sdfg")
    #otf_tuner.optimize(apply=True)

    #sf_tuner = optim.SubgraphFusionTuner(small_sample, measurement=dace.InstrumentationType.GPU_Events)
    #optim.SubgraphFusionTuner.transfer(big_sdfg, sf_tuner, k=1)

    #sdfg_path = Path(os.environ["HOME"]) / "projects/tuning-dace/hF-transfer2.sdfg"
    #big_sdfg.save(sdfg_path)
