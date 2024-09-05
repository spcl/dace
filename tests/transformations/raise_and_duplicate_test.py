import dace
from dace.sdfg import nodes
from dace.transformation.interstate import IfRaising, StateReplication
from dace.transformation.dataflow import OTFMapFusion
import numpy as np


def test_raise_and_duplicate_and_fusions():
    N = dace.symbol('N', dace.int64)
    @dace.program
    def program(flag: dace.bool, in_arr: dace.float64[N], arr: dace.float64[N]):
        tmp1 = np.empty_like(arr)
        tmp2 = np.empty_like(arr)
        for i in dace.map[0:N]:
            tmp1[i] = in_arr[i]
        if flag:
            for i in dace.map[0:N]:
                tmp2[i] = tmp1[i]
        else:
            for i in dace.map[0:N]:
                tmp2[i] = tmp1[i]
        for i in dace.map[0:N]:
                arr[i] = tmp2[i]

    sdfg = program.to_sdfg()
    sdfg.apply_transformations([IfRaising, StateReplication])
    sdfg.simplify()
    sdfg.apply_transformations_repeated([OTFMapFusion])

    assert len(sdfg.nodes()) == 4
    assert sdfg.start_state.is_empty()
    
    entries = 0
    for state in sdfg.nodes():
        for node in state.nodes():
            if isinstance(node, nodes.MapEntry):
                entries += 1

    assert entries == 2


def test_if_raise_dependency():
    N = dace.symbol('N', dace.int64)
    @dace.program
    def program(arr: dace.float64[N]):
        flag = np.sum(arr)
        if flag:
            return 1
        else:
            return 0

    sdfg = program.to_sdfg()

    transform = IfRaising()
    transform.if_guard = sdfg.start_state

    assert not transform.can_be_applied(sdfg, 0, sdfg)


if __name__ == '__main__':
    test_raise_and_duplicate_and_fusions()
    test_if_raise_dependency()
