# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace import nodes
from dace.dtypes import ScheduleType
from dace.memlet import Memlet
from dace.transformation.change_strides import change_strides


def change_strides_test():
    sdfg = dace.SDFG('change_strides_test')
    N = dace.symbol('N')
    M = dace.symbol('M')
    sdfg.add_array('A', [N, M], dace.float64)
    sdfg.add_array('B', [N, M, 3], dace.float64)
    state = sdfg.add_state()

    task1, mentry1, mexit1 = state.add_mapped_tasklet(
            name="map1",
            map_ranges={'i': '0:N', 'j': '0:M'},
            inputs={'a': Memlet(data='A', subset='i, j')},
            outputs={'b': Memlet(data='B', subset='i, j, 0')},
            code='b = a + 1',
            external_edges=True,
            propagate=True)

    # Check that states are as expected
    changed_sdfg = change_strides(sdfg, ['N'], ScheduleType.Sequential)
    assert len(changed_sdfg.states()) == 3
    assert len(changed_sdfg.out_edges(changed_sdfg.start_state)) == 1
    work_state = changed_sdfg.out_edges(changed_sdfg.start_state)[0].dst
    nsdfg = None
    for node in work_state.nodes():
        if isinstance(node, nodes.NestedSDFG):
            nsdfg = node
    # Check shape and strides of data inside nested SDFG
    assert nsdfg is not None
    assert nsdfg.sdfg.data('A').shape == (N, M)
    assert nsdfg.sdfg.data('B').shape == (N, M, 3)
    assert nsdfg.sdfg.data('A').strides == (1, N)
    assert nsdfg.sdfg.data('B').strides == (1, N, M*N)


def main():
    change_strides_test()


if __name__ == '__main__':
    main()
