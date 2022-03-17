# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import os
import dace
import numpy as np

from pathlib import Path

from dace.optimization import cutout_tuner as ct
from dace import optimization as optim
from dace.optimization import utils as optim_utils

if __name__ == '__main__':

    sdfg_path = Path(os.environ["HOME"]) / "projects/tuning-dace/aha.sdfg"
    sdfg = dace.SDFG.from_file(sdfg_path)
    sdfg.instrument = dace.InstrumentationType.Timer

    sdfg.apply_gpu_transformations()
    sdfg.simplify()
    
    for state in sdfg.nodes():
        state.instrument = dace.InstrumentationType.GPU_Events
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                node.schedule = dace.ScheduleType.GPU_Device


    dreport = sdfg.get_instrumented_data()
    if dreport is None:
        arguments = {}
        for name, array in sdfg.arrays.items():
            if array.transient:
                continue

            data = dace.data.make_array_from_descriptor(array, np.random.rand(*array.shape))
            arguments[name] = data

        ct.CutoutTuner.dry_run(sdfg, **arguments)
        print("Goodbye")
        exit()

    print("We got it")

    tuner = optim.OnTheFlyMapFusionTuner(sdfg, measurement=dace.InstrumentationType.GPU_Events)
    tuner.optimize(apply=True)

    sdfg_path = Path(os.environ["HOME"]) / "projects/tuning-dace/aha-otf-fused.sdfg"
    sdfg.save(sdfg_path)

    tuner = optim.SubgraphFusionTuner(sdfg, measurement=dace.InstrumentationType.GPU_Events)
    tuner.optimize(apply=True)

    sdfg_path = Path(os.environ["HOME"]) / "projects/tuning-dace/aha-subgraph-fused.sdfg"
    sdfg.save(sdfg_path)

