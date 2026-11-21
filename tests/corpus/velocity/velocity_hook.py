# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The velocity-tendencies pipeline, as a dace-fortran e2e harness hook (``OCEAN_E2E_SDFG_HOOK``).

The harness runs it on the freshly built SDFG in its child process: bake the configuration in ``VT_CONFIG``, drop
the branches it decides, parallelize, offload when ``VT_DEVICE`` is ``gpu``, and re-snapshot the Fortran signature
the binding is generated from.
"""

import json
import os

from dace import dtypes
from dace.sdfg import SDFG, nodes
from dace.transformation.passes.config_propagation import specialize_variant
from dace.transformation.passes.offloading import OffloadToAccelerator


def block_maps(sdfg: SDFG) -> list[nodes.MapEntry]:
    """The outermost maps over a block range (``i_startblk:i_endblk``): ICON launches its kernels from these, one block
    at a time, so they stay host code."""
    found = []
    for node, state in sdfg.all_nodes_recursive():
        if not isinstance(node, nodes.MapEntry) or not str(node.map.range[0][0]).endswith("startblk"):
            continue
        scope = state.entry_node(node)
        while scope is not None and not str(scope.map.range[0][0]).endswith("startblk"):
            scope = state.entry_node(scope)
        if scope is None:
            found.append(node)
    return found


def kernels(sdfg: SDFG) -> int:
    return sum(
        1
        for node, parent in sdfg.all_nodes_recursive()
        if isinstance(node, nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.GPU_Device
    )


def pipeline(sdfg: SDFG, values: dict, device: str) -> None:
    from dace_fortran.pipelines import num_loops, num_maps, optimize

    specialize_variant(sdfg, values)
    optimize(sdfg)
    print(f"VT_PIPELINE maps={num_maps(sdfg)} loops={num_loops(sdfg)}", flush=True)
    if device == "gpu":
        hosts = block_maps(sdfg)
        OffloadToAccelerator(host_maps=hosts).apply_pass(sdfg, {})
        print(f"VT_GPU host_maps={len(hosts)} kernels={kernels(sdfg)}", flush=True)


def run(sdfg: SDFG) -> None:
    from dace_fortran.bindings.frozen_signature import refreeze

    pipeline(sdfg, json.loads(os.environ["VT_CONFIG"]), os.environ.get("VT_DEVICE", "cpu"))
    refreeze(sdfg)
