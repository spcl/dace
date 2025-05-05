# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import dace
from dace.transformation.dataflow import MapSplit
from dace.sdfg.nodes import NestedSDFG

N = dace.symbol('N', dtype=dace.int32)

@dace.program
def unapplicable(arr: dace.float32[N]):
    for i in dace.map[0:N]:
        arr[i] = 0


@dace.program
def simple_nested(arr: dace.float32[N, N]):
    for i in dace.map[0:N]:
        if i == 0:
            s = N/2
        elif i == N-1:
            s = N/3
        else:
            s = N
        for j in dace.map[0:s]:
            arr[i,j] = 0


def test_fail_application():
    sdfg = unapplicable.to_sdfg()
    applications = sdfg.apply_transformations_repeated([MapSplit])
    assert applications == 0


def test_multiple_applications():
    sdfg = simple_nested.to_sdfg()
    applications = sdfg.apply_transformations_repeated([MapSplit])
    assert applications == 2

    sdfg.simplify()

    for n in sdfg.start_state.nodes():
        assert not isinstance(n, NestedSDFG)


if __name__ == '__main__':
    test_fail_application()
    test_multiple_applications()