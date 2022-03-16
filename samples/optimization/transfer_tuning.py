# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import os
import dace

from pathlib import Path

from dace import optimization as optim
from dace.optimization import utils as optim_utils

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

    runtime = optim_utils.measure(big_sdfg)
    
    print("Transfer")

    otf_tuner = optim.OnTheFlyMapFusionTuner(small_sample, measurement=dace.InstrumentationType.GPU_Events)
    optim.OnTheFlyMapFusionTuner.transfer(big_sdfg, otf_tuner, k=10)
    
    otf_tuner.optimize(apply=True)

    sf_tuner = optim.SubgraphFusionTuner(small_sample, measurement=dace.InstrumentationType.GPU_Events)
    optim.SubgraphFusionTuner.transfer(big_sdfg, sf_tuner, k=10)

    tuned_runtime = optim_utils.measure(big_sdfg)
    print("Tuning speedup: ",  runtime / tuned_runtime)

    sdfg_path = Path(os.environ["HOME"]) / "projects/tuning-dace/aha-expanded-2-transfer.sdfg"
    big_sdfg.save(sdfg_path)
