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

    sdfg = dace.SDFG.from_file(Path("/home/lukas/projects/tuning-dace/") / "aha-expanded.sdfg")
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
    
    tuner = optim.MapFusionTuner(sdfg)
    for i, (cutout, label) in enumerate(tuner.cutouts()):
        for i, map_ids in tuner.space(cutout):
            maps_ = list(map(lambda m: cutout.start_state.node(m), map_ids))
            subgraph = helpers.subgraph_from_maps(sdfg=cutout, graph=cutout.start_state, map_entries=maps_)

            if len(maps_) < 2:
                continue

            fused_subgraph = cutter.cutout_state(cutout.start_state, *(subgraph.nodes()))
            with open(cutout_path / f"cutout_{i}.sdfg", "w") as handle:
                json.dump(fused_subgraph.to_json(), handle)

    tuner.optimize()
