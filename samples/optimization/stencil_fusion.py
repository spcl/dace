# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import shutil
import os
import dace
import numpy as np

from pathlib import Path
from tqdm import tqdm

from dace.sdfg.analysis import cutout as cutter
from dace.optimization import cutout_tuner as ct
from dace import optimization as optim
from dace.transformation.subgraph import helpers
from dace.transformation import subgraph as sg

def measure(sdfg, arguments):
    with dace.config.set_temporary('debugprint', value=False):
        with dace.config.set_temporary('instrumentation', 'report_each_invocation', value=False):
            with dace.config.set_temporary('compiler', 'allow_view_arguments', value=True):
                csdfg = sdfg.compile()

                for _ in range(30):
                    csdfg(**arguments)

                csdfg.finalize()

    report = sdfg.get_latest_report()
    print(report)

if __name__ == '__main__':

    sdfg_path = Path(os.environ["HOME"]) / "projects/tuning-dace/aha-expanded-2.sdfg"
    sdfg = dace.SDFG.from_file(sdfg_path)
    sdfg.instrument = dace.InstrumentationType.Timer

    sdfg.apply_gpu_transformations()
    sdfg.simplify()

    # do_not_transient = set()
    # visited = set()
    # for state in sdfg.nodes():
    #     for node in state.data_nodes():
    #         in_degree = state.in_degree(node)
    #         out_degree = state.out_degree(node)

    #         if in_degree == 0 or out_degree == 0 or node.data in visited:
    #             do_not_transient.add(node.data)
    #         visited.add(node.data)

    arguments = {}
    for name, array in sdfg.arrays.items():
        # if name not in do_not_transient:
        #     array.transient = True
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
    measure(sdfg, arguments)

    tuner = optim.OnTheFlyMapFusionTuner(sdfg, measurement=dace.InstrumentationType.GPU_Events)
    tuner.optimize(apply=True)
    measure(sdfg, arguments)

    sdfg_path = Path(os.environ["HOME"]) / "projects/tuning-dace/aha-expanded_otf_fused.sdfg"
    sdfg.save(sdfg_path)

    tuner = optim.SubgraphFusionTuner(sdfg, measurement=dace.InstrumentationType.GPU_Events)
    tuner.optimize(apply=True)
    measure(sdfg, arguments)

    sdfg_path = Path(os.environ["HOME"]) / "projects/tuning-dace/aha-expanded_sub_fused.sdfg"
    sdfg.save(sdfg_path)

    print("Permutation")
    tuner = optim.MapPermutationTuner(sdfg, measurement=dace.InstrumentationType.GPU_Events)
    tuner.optimize(apply=True)
    measure(sdfg, arguments)

    sdfg_path = Path(os.environ["HOME"]) / "projects/tuning-dace/aha-expanded_fused_permuted.sdfg"
    sdfg.save(sdfg_path)

    print("Tiling")
    tuner = optim.MapTilingTuner(sdfg, measurement=dace.InstrumentationType.GPU_Events)
    tuner.optimize(apply=True)
    measure(sdfg, arguments)

    sdfg_path = Path(os.environ["HOME"]) / "projects/tuning-dace/aha-expanded_fused_permuted_tiled.sdfg"
    sdfg.save(sdfg_path)