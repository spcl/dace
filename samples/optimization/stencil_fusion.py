# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import os
import dace
import numpy as np

from pathlib import Path

from dace.optimization import cutout_tuner as ct
from dace import optimization as optim
from dace.optimization import utils as optim_utils

if __name__ == '__main__':

    sdfg_path = Path(os.environ["HOME"]) / "projects/tuning-dace/aha-expanded-2.sdfg"
    sdfg = dace.SDFG.from_file(sdfg_path)
    sdfg.instrument = dace.InstrumentationType.Timer

    sdfg.apply_gpu_transformations()
    sdfg.simplify()

    arguments = {}
    for name, array in sdfg.arrays.items():
        if array.transient:
            continue

        data = dace.data.make_array_from_descriptor(array, np.random.rand(*array.shape))
        arguments[name] = data
    
    
    for state in sdfg.nodes():
        state.instrument = dace.InstrumentationType.GPU_Events
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                node.schedule = dace.ScheduleType.GPU_Device

    result = ct.CutoutTuner.dry_run(sdfg, **arguments)

    print("Initial version")
    dreport = sdfg.get_instrumented_data()
    optim_utils.measure(sdfg, dreport)

    tuner = optim.OnTheFlyMapFusionTuner(sdfg, measurement=dace.InstrumentationType.GPU_Events)
    tuner.optimize(apply=True)
    optim_utils.measure(sdfg, dreport)

    sdfg_path = Path(os.environ["HOME"]) / "projects/tuning-dace/aha-expanded_otf_fused.sdfg"
    sdfg.save(sdfg_path)

    tuner = optim.SubgraphFusionTuner(sdfg, measurement=dace.InstrumentationType.GPU_Events)
    tuner.optimize(apply=True)
    optim_utils.measure(sdfg, dreport)

    sdfg_path = Path(os.environ["HOME"]) / "projects/tuning-dace/aha-expanded_sub_fused.sdfg"
    sdfg.save(sdfg_path)

    # print("Permutation")
    # tuner = optim.MapPermutationTuner(sdfg, measurement=dace.InstrumentationType.GPU_Events)
    # tuner.optimize(apply=True)
    # measure(sdfg, arguments)

    # sdfg_path = Path(os.environ["HOME"]) / "projects/tuning-dace/aha-expanded_fused_permuted.sdfg"
    # sdfg.save(sdfg_path)

    # print("Tiling")
    # tuner = optim.MapTilingTuner(sdfg, measurement=dace.InstrumentationType.GPU_Events)
    # tuner.optimize(apply=True)
    # measure(sdfg, arguments)

    # sdfg_path = Path(os.environ["HOME"]) / "projects/tuning-dace/aha-expanded_fused_permuted_tiled.sdfg"
    # sdfg.save(sdfg_path)