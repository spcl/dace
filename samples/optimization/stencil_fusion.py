# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import shutil
import os
import dace
import numpy as np

from pathlib import Path

from dace.optimization import cutout_tuner as ct
from dace import optimization as optim

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

    sdfg_path = Path(os.environ["HOME"]) / "projects/tuning-dace/aha-expanded.sdfg"
    sdfg = dace.SDFG.from_file(sdfg_path)
    sdfg.instrument = dace.InstrumentationType.Timer
    # sdfg = dace.SDFG.from_file(Path("/home/lukas/projects/autodace/") / "test_stencil.sdfg")

    cache_path = Path(__file__).parent / ".dacecache"
    shutil.rmtree(cache_path, ignore_errors=True)

    tuning_paths = Path(__file__).parent.rglob("*.tuning")
    for path in tuning_paths:
        os.remove(path)

    cutout_path = Path(__file__).parent / "cutouts"
    shutil.rmtree(cutout_path, ignore_errors=True)
    cutout_path.mkdir()

    # values = {}
    # for v in sdfg.free_symbols:
    #     values[str(v)] = np.int32(64)

    # sdfg.specialize(values)

    arguments = {}
    for name, array in sdfg.arrays.items():
        # dtype = array.dtype
        # shape_ = tuple(map(lambda dim: dace.symbolic.evaluate(dim, values), array.shape))
        array.storage = dace.StorageType.GPU_Shared

        # array.shape = shape_
        # array.total_size = dace.symbolic.evaluate(array.total_size, values)
        # array.strides = tuple(map(lambda dim: dace.symbolic.evaluate(dim, values), array.strides))
        
        if not array.transient:
            data = dace.data.make_array_from_descriptor(array, np.random.rand(*array.shape))
            arguments[name] = data

    for state in sdfg.nodes():
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                node.schedule = dace.ScheduleType.GPU_Device

    result = ct.CutoutTuner.dry_run(sdfg, **arguments)

    print("Initial version")
    measure(sdfg, arguments)

    print("Fusing")
    tuner = optim.MapFusionTuner(sdfg)
    tuner.optimize(apply=True)
    measure(sdfg, arguments)

    print("Permutation")
    tuner = optim.MapPermutationTuner(sdfg)
    tuner.optimize(apply=True)
    measure(sdfg, arguments)

    print("Tiling")
    tuner = optim.MapTilingTuner(sdfg)
    tuner.optimize(apply=True)
    measure(sdfg, arguments)

    sdfg_path = Path(os.environ["HOME"]) / "projects/tuning-dace/aha-expanded_fused_permuted_tiled_cpu.sdfg"
    sdfg.save(sdfg_path)
