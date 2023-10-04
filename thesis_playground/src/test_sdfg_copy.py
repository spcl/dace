import copy

import dace
from dace.memlet import Memlet


def main():
    sdfg = dace.SDFG('sdfg_copy_test')
    KLEV = dace.symbol('KLEV')
    sdfg.add_array('A', [KLEV], dace.float64)
    sdfg.add_array('B', [KLEV], dace.float64)
    state = sdfg.add_state()

    nsdfg_sdfg = dace.SDFG('sdfg_copy_test_nested')
    nsdfg_sdfg.add_array('A', [KLEV], dace.float64)
    nsdfg_sdfg.add_array('B', [KLEV], dace.float64)
    nsdfg_state = nsdfg_sdfg.add_state()

    nsdfg_state.add_mapped_tasklet(
        name="block_map",
        map_ranges={"JK": "0:KLEV"},
        inputs={"A": Memlet(data='A', subset='JK')},
        outputs={"B": Memlet(data='B', subset='JK')},
        code='b = a'
        )

    state.add_nested_sdfg(nsdfg_sdfg, sdfg, {'A'}, {'B'})

    # Either works
    # sdfg.specialize({'KLEV': 13})
    sdfg.add_constant('KLEV', 13)
    print(nsdfg_sdfg.constants)
    sdfg_copy = copy.deepcopy(nsdfg_sdfg)
    print(sdfg_copy.constants)


if __name__ == '__main__':
    main()
