# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import pytest


@pytest.mark.gpu
def test_maxnreg():
    B = 2

    @dace.program
    def prog(a: dace.float64[100, 20] @ dace.StorageType.GPU_Global):
        for i, j in dace.map[0:100, 0:20] @ dace.ScheduleType.GPU_Device:
            a[i, j] = 1

    sdfg = prog.to_sdfg()
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, dace.nodes.MapEntry) and n.map.schedule == dace.ScheduleType.GPU_Device:
            mapentry = n

    assert '__launch_bounds__' in sdfg.generate_code()[1].code
    mapentry.map.gpu_maxnreg = 64
    assert '__maxnreg__(64)' in sdfg.generate_code()[1].code and '__launch_bounds__' not in sdfg.generate_code()[1].code


if __name__ == '__main__':
    test_maxnreg()
