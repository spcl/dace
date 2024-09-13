# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import dace
from dace.transformation.interstate import IfExtraction
from dace.sdfg.nodes import NestedSDFG

N = dace.symbol('N', dtype=dace.int32)

@dace.program
def simple_application(flag: dace.bool, arr: dace.float32[N]):
    for i in dace.map[0:N]:
        if flag:
            outval = 1
        else:
            outval = 2
        arr[i] = outval


@dace.program
def dependant_application(flag: dace.bool, arr: dace.float32[N]):
    for i in dace.map[0:N]:
        if i == 0:
            outval = 1
        else:
            outval = 2
        arr[i] = outval


def test_simple_application():
    sdfg = simple_application.to_sdfg(simplify=True)

    assert len(sdfg.nodes()) == 1

    assert sdfg.apply_transformations_repeated([IfExtraction]) == 1

    assert len(sdfg.nodes()) == 4
    assert sdfg.start_state.is_empty()
    
    sdfg.simplify()
    
    for s in sdfg.nodes():
        for n in s.nodes():
            assert not isinstance(n, NestedSDFG)

def test_fails_due_to_dependency():
    sdfg = dependant_application.to_sdfg(simplify=True)

    assert sdfg.apply_transformations_repeated([IfExtraction]) == 0


if __name__ == '__main__':
    test_simple_application()
    test_fails_due_to_dependency()
