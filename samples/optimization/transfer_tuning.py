# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import os
import dace
import numpy as np

from pathlib import Path

from dace import optimization as optim
from dace.optimization import utils as optim_utils
from dace.optimization import cutout_tuner as ct

if __name__ == '__main__':

    sdfg_path = Path(os.environ["HOME"]) / "projects/tuning-dace/aha-expanded-2.sdfg"
    small_sample = dace.SDFG.from_file(sdfg_path)

    small_sample.apply_gpu_transformations()
    small_sample.simplify()

    for state in small_sample.nodes():
        state.instrument = dace.InstrumentationType.GPU_Events
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                node.schedule = dace.ScheduleType.GPU_Device

    sdfg_path = Path(os.environ["HOME"]) / "projects/tuning-dace/aha-expanded-2-copy.sdfg"
    big_sdfg = dace.SDFG.from_file(sdfg_path)

    big_sdfg.apply_gpu_transformations()
    big_sdfg.simplify()

    for state in big_sdfg.nodes():
        state.instrument = dace.InstrumentationType.GPU_Events
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                node.schedule = dace.ScheduleType.GPU_Device


    arguments = {}
    for name, array in big_sdfg.arrays.items():
        if array.transient:
            continue

        data = dace.data.make_array_from_descriptor(array, np.random.rand(*array.shape))
        arguments[name] = data
    
    result = ct.CutoutTuner.dry_run(big_sdfg, **arguments)
    dreport = big_sdfg.get_instrumented_data()
    runtime = optim_utils.measure(big_sdfg, dreport)
    
    print("Transfer")

    otf_tuner = optim.OnTheFlyMapFusionTuner(small_sample, measurement=dace.InstrumentationType.GPU_Events)
    optim.OnTheFlyMapFusionTuner.transfer(big_sdfg, otf_tuner, k=10)
    
    otf_tuner.optimize(apply=True)

    sf_tuner = optim.SubgraphFusionTuner(small_sample, measurement=dace.InstrumentationType.GPU_Events)
    optim.SubgraphFusionTuner.transfer(big_sdfg, sf_tuner, k=10)

    tuned_runtime = optim_utils.measure(big_sdfg, dreport)
    print("Tuning speedup: ",  runtime / tuned_runtime)

    sdfg_path = Path(os.environ["HOME"]) / "projects/tuning-dace/aha-expanded-2-transfer.sdfg"
    big_sdfg.save(sdfg_path)
