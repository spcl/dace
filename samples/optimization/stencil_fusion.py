# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
from math import prod
import shutil
import os
import dace
import json
import numpy as np

from pathlib import Path

from dace.transformation.auto.auto_optimize import auto_optimize
from dace.optimization import cutout_tuner as ct
from dace import optimization as optim
from dace.transformation.subgraph import helpers
from dace.transformation.subgraph import composite as comp
from dace.sdfg.analysis import cutout as cutter

if __name__ == '__main__':

    sdfg_path = Path("/home/lukas/projects/tuning-dace/") / "aha-expanded copy.sdfg"
    sdfg = dace.SDFG.from_file(sdfg_path)
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
        array.storage = dace.StorageType.CPU_Heap

        # array.shape = shape_
        # array.total_size = dace.symbolic.evaluate(array.total_size, values)
        # array.strides = tuple(map(lambda dim: dace.symbolic.evaluate(dim, values), array.strides))
        
        if not array.transient:
            data = dace.data.make_array_from_descriptor(array, np.random.rand(*array.shape))
            arguments[name] = data


    for state in sdfg.nodes():
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                node.schedule = dace.ScheduleType.CPU_Multicore

    result = ct.CutoutTuner.dry_run(sdfg, **arguments)

    tuner = optim.MapPermutationTuner(sdfg)
    tuner.optimize(apply=True)
    sdfg_path = Path("/home/lukas/projects/tuning-dace/") / "aha-expanded_copy_permuted.sdfg"
    sdfg.save(sdfg_path)

    tuning_paths = Path(__file__).parent.rglob("*.tuning")
    for path in tuning_paths:
        os.remove(path)

    tuner = optim.MapFusionTuner(sdfg)
    tuner.optimize(apply=True)

    tuning_paths = Path(__file__).parent.rglob("*.tuning")
    for path in tuning_paths:
        os.remove(path)

    sdfg_path = Path("/home/lukas/projects/tuning-dace/") / "aha-expanded_copy_permuted_fused.sdfg"
    sdfg.save(sdfg_path)

    tuner = optim.MapPermutationTuner(sdfg)
    tuner.optimize(apply=True)
    
    sdfg_path = Path("/home/lukas/projects/tuning-dace/") / "aha-expanded_copy_permuted_fused_permuted.sdfg"
    sdfg.save(sdfg_path)
