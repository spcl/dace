# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``tile_every_kernel`` applies exactly what ``apply_transformations_once_everywhere(AddThreadBlockMap)``
applies, in one sweep instead of a full re-enumeration after every kernel it tiles."""
import copy
import json
from unittest import mock

import dace
from dace.transformation.dataflow.add_threadblock_map import AddThreadBlockMap
from dace.transformation.passes import pattern_matching
from dace.transformation.passes.gpu_specialization.codegen_preprocess_passes import tile_every_kernel

GPU = dace.dtypes.ScheduleType.GPU_Device
TB = dace.dtypes.ScheduleType.GPU_ThreadBlock
GLOBAL = dace.dtypes.StorageType.GPU_Global


@dace.program
def kernels(A: dace.float64[64] @ GLOBAL, B: dace.float64[64] @ GLOBAL, C: dace.float64[8, 8] @ GLOBAL):
    for i in dace.map[0:64] @ GPU:
        A[i] = 2.0 * B[i]
    for i, j in dace.map[0:8, 0:8] @ GPU:
        C[i, j] = A[i * 8 + j] + 1.0
    for i in dace.map[0:2] @ GPU:
        for j in dace.map[0:32] @ TB:
            B[i * 32 + j] = A[i * 32 + j]
    for i in dace.map[0:64]:
        A[i] = A[i] + B[i]


def kernels_sdfg() -> dace.SDFG:
    """``kernels`` with a block size on every kernel map, so tiling needs no configuration fallback."""
    sdfg = kernels.to_sdfg(simplify=True)
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry) and node.map.schedule == GPU:
            node.map.gpu_block_size = [32, 1, 1]
    return sdfg


def stripped(sdfg: dace.SDFG) -> str:

    def strip(obj):
        if isinstance(obj, dict):
            return {k: strip(v) for k, v in obj.items() if k not in ('guid', 'hash', 'debuginfo', 'source_files')}
        if isinstance(obj, list):
            return [strip(v) for v in obj]
        return obj

    return json.dumps(strip(sdfg.to_json()), sort_keys=True, default=str)


def test_sweep_matches_apply_once_everywhere():
    sdfg = kernels_sdfg()
    generic, swept = copy.deepcopy(sdfg), copy.deepcopy(sdfg)
    applied = generic.apply_transformations_once_everywhere(AddThreadBlockMap)
    assert applied == 2  # the two kernels without a thread-block map; the third has one, the fourth is host
    tile_every_kernel(swept)
    assert stripped(swept) == stripped(generic)


def test_sweep_enumerates_the_maps_once():
    sdfg = kernels_sdfg()
    original = pattern_matching.match_patterns
    with mock.patch.object(pattern_matching, 'match_patterns', side_effect=original) as spy:
        tile_every_kernel(sdfg)
    assert spy.call_count == 0
    tiled = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry) and n.map.schedule == TB]
    assert len(tiled) == 3


if __name__ == '__main__':
    test_sweep_matches_apply_once_everywhere()
    test_sweep_enumerates_the_maps_once()
