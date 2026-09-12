# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import dace
from dace.transformation.passes import ResolveCondition
from dace.transformation.dataflow import MapFusion
import numpy as np
import pytest

@pytest.mark.parametrize("condition, after_pass, after_simplify, map_fusion_applications", [
    ('flag', 3, 1, 2),
    ('not flag', 3, 1, 2),
    ('dummy', 4, 4, 0)
])


def test_application_and_fusion(condition: str, after_pass: int, after_simplify: int, map_fusion_applications: int):
    N = dace.symbol('N', dtype=dace.int32)

    @dace.program
    def simple_program(flag: dace.bool, arr: dace.float32[N]):
        tmp1 = np.empty_like(arr)
        tmp2 = np.empty_like(arr)
        for i in dace.map[0:N]:
            tmp1[i] = arr[i] + 1
        if flag:
            for i in dace.map[0:N]:
                tmp2[i] = tmp1[i] * 2
        else:
            for i in dace.map[0:N]:
                tmp2[i] = tmp1[i] * 3
        for i in dace.map[0:N]:
            arr[i] = tmp2[i] + 1


    sdfg = simple_program.to_sdfg(simplify=True)

    assert len(sdfg.nodes()) == 4

    p = ResolveCondition()
    p.condition = condition
    p.apply_pass(sdfg, {})

    assert len(sdfg.nodes()) == after_pass

    sdfg.simplify()

    assert len(sdfg.nodes()) == after_simplify

    assert sdfg.apply_transformations_repeated([MapFusion]) == map_fusion_applications


if __name__ == '__main__':
    test_application_and_fusion('flag', 3, 1, 2)
    test_application_and_fusion('not flag', 3, 1, 2)
    test_application_and_fusion('dummy', 4, 4, 0)
