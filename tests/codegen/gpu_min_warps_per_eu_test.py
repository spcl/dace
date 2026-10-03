# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import pytest


@pytest.mark.gpu
def test_min_warps_per_eu() -> None:

    @dace.program
    def prog(a: dace.float64[100, 20] @ dace.StorageType.GPU_Global) -> None:
        for i, j in dace.map[0:100, 0:20] @ dace.ScheduleType.GPU_Device:
            a[i, j] = 1

    sdfg = prog.to_sdfg()
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, dace.nodes.MapEntry) and n.map.schedule == dace.ScheduleType.GPU_Device:
            mapentry = n

    # Expected launch bound is the default block's total thread count (product of its extents),
    # not a hard-coded number -- so the test tracks the configured default_block_size.
    block_threads = 1
    for extent in dace.Config.get('compiler', 'cuda', 'default_block_size').split(','):
        block_threads *= int(extent)

    assert f'__launch_bounds__({block_threads})' in sdfg.generate_code()[1].code
    mapentry.map.gpu_min_warps_per_eu = 4
    code = sdfg.generate_code()[1].code
    assert f'__launch_bounds__({block_threads},4)' in code and f'__launch_bounds__({block_threads})' not in code


if __name__ == '__main__':
    test_min_warps_per_eu()
