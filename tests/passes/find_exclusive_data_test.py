# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Dict, Set, Tuple
import dace
from dace.transformation.passes.analysis import FindExclusiveData

def perform_scan(sdfg: dace.SDFG) -> Dict[dace.SDFG, Set[str]]:
    scanner = FindExclusiveData()
    return scanner.apply_pass(sdfg, None)


def _make_all_exclusive_data_but_one_unused_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG('all_exclusive_data_but_one_unused_sdfg')
    state1 = sdfg.add_state(is_start_block=True)
    state2 = sdfg.add_state_after(state1)

    for name in 'abcde':
        sdfg.add_array(
                name,
                shape=(10, 10),
                dtype=dace.float64,
                transient=False,
        )

    state1.add_nedge(
            state1.add_access('a'),
            state1.add_access('b'),
            sdfg.make_array_memlet('a')
    )
    state2.add_nedge(
            state2.add_access('c'),
            state2.add_access('d'),
            sdfg.make_array_memlet('c')
    )
    sdfg.validate()
    return sdfg


def test_all_exclusive_data_but_one_unused():
    sdfg = _make_all_exclusive_data_but_one_unused_sdfg()
    assert len(sdfg.arrays) == 5

    # Because it is not used `e` is not considered to be exclusively used.
    #  This is a matter of definition.
    expected_exclusive_set = {aname for aname in sdfg.arrays.keys() if aname != 'e'}

    exclusive_set = perform_scan(sdfg)
    assert len(exclusive_set[sdfg]) == 4

    assert exclusive_set[sdfg] == expected_exclusive_set


def _make_multiple_access_same_state_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG('multiple_access_same_state_sdfg')
    state = sdfg.add_state(is_start_block=True)

    for name in 'abd':
        sdfg.add_array(
                name,
                shape=(10, 10),
                dtype=dace.float64,
                transient=False,
        )

    state.add_nedge(
            state.add_access('a'),
            state.add_access('b'),
            sdfg.make_array_memlet('a')
    )
    state.add_nedge(
            state.add_access('a'),
            state.add_access('d'),
            sdfg.make_array_memlet('a')
    )
    sdfg.validate()
    return sdfg


def test_multiple_access_same_state():
    sdfg = _make_multiple_access_same_state_sdfg()
    assert len(sdfg.arrays) == 3

    # `a` is not exclusive because there exists multiple access nodes in a single
    #  state for `a`.
    expected_exclusive_set = {aname for aname in sdfg.arrays.keys() if aname != 'a'}
    exclusive_set = perform_scan(sdfg)
    assert len(exclusive_set[sdfg]) == 2
    assert expected_exclusive_set == exclusive_set[sdfg]


def _make_multiple_single_access_node_same_state_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG('multiple_single_access_node_same_state_sdfg')
    state = sdfg.add_state(is_start_block=True)

    for name in 'abd':
        sdfg.add_array(
                name,
                shape=(10, 10),
                dtype=dace.float64,
                transient=False,
        )

    a = state.add_access('a')
    state.add_nedge(
            a,
            state.add_access('b'),
            sdfg.make_array_memlet('a')
    )
    state.add_nedge(
            a,
            state.add_access('d'),
            sdfg.make_array_memlet('a')
    )
    assert state.out_degree(a) == 2
    sdfg.validate()
    return sdfg


def test_multiple_single_access_node_same_state_sdfg() -> dace.SDFG:
    sdfg = _make_multiple_single_access_node_same_state_sdfg()
    assert len(sdfg.arrays) == 3

    # Unlike `test_multiple_access_same_state()` here `a` is included in the exclusive
    #  set, because, there is only a single AccessNode, that is used multiple times,
    #  i.e. has an output degree larger than one.
    expected_exclusive_set = sdfg.arrays.keys()
    exclusive_set = perform_scan(sdfg)
    assert len(exclusive_set[sdfg]) == 3
    assert expected_exclusive_set == exclusive_set[sdfg]


def _make_multiple_access_different_states_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG('multiple_access_different_states_sdfg')
    state1 = sdfg.add_state(is_start_block=True)
    state2 = sdfg.add_state_after(state1)

    for name in 'abd':
        sdfg.add_array(
                name,
                shape=(10, 10),
                dtype=dace.float64,
                transient=False,
        )

    # Note these edges are useless as `a` is written to twice. It is just to generate
    #  an additional case, i.e. the data are also written to.
    state1.add_nedge(
            state1.add_access('b'),
            state1.add_access('a'),
            sdfg.make_array_memlet('a')
    )
    state2.add_nedge(
            state2.add_access('d'),
            state2.add_access('a'),
            sdfg.make_array_memlet('a')
    )
    sdfg.validate()
    return sdfg


def test_multiple_access_different_states():
    sdfg = _make_multiple_access_different_states_sdfg()
    assert len(sdfg.arrays) == 3

    # `a` is not included in the exclusive set, because it is used in two different states.
    exclusive_set = perform_scan(sdfg)
    expected_exclusive_set = {aname for aname in sdfg.arrays.keys() if aname != 'a'}
    assert len(exclusive_set[sdfg]) == 2
    assert expected_exclusive_set == exclusive_set[sdfg]


def _make_access_only_on_interstate_edge_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG('access_on_interstate_edge_sdfg')

    for name in 'abcd':
        sdfg.add_array(
                name,
                shape=(10, 10),
                dtype=dace.float64,
                transient=False,
        )
    sdfg.add_scalar('e', dtype=dace.float64, transient=False)

    state1 = sdfg.add_state(is_start_block=True)
    state2 = sdfg.add_state_after(state1, assignments={'e_sym': 'e'})

    state1.add_nedge(
            state1.add_access('a'),
            state1.add_access('b'),
            sdfg.make_array_memlet('a')
    )
    state2.add_nedge(
            state2.add_access('c'),
            state2.add_access('d'),
            sdfg.make_array_memlet('c')
    )
    sdfg.validate()
    return sdfg


def test_access_only_on_interstate_edge():
    sdfg = _make_access_only_on_interstate_edge_sdfg()
    assert len(sdfg.arrays) == 5

    # `e` is part of the exclusive set, because it is only accessed on an interstate edge.
    expected_exclusive_set = sdfg.arrays.keys()
    exclusive_set = perform_scan(sdfg)
    assert len(exclusive_set[sdfg]) == 5
    assert exclusive_set[sdfg] == expected_exclusive_set


def _make_additional_access_on_interstate_edge_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG('additional_access_on_interstate_edge_sdfg')

    for name in 'abcd':
        sdfg.add_array(
                name,
                shape=(10, 10),
                dtype=dace.float64,
                transient=False,
        )
    sdfg.add_scalar('e', dtype=dace.float64, transient=False)
    sdfg.add_scalar('f', dtype=dace.float64, transient=False)

    state1 = sdfg.add_state(is_start_block=True)
    state2 = sdfg.add_state_after(state1, assignments={'e_sym': 'e'})

    state1.add_nedge(
            state1.add_access('a'),
            state1.add_access('b'),
            sdfg.make_array_memlet('a')
    )
    state2.add_nedge(
            state2.add_access('c'),
            state2.add_access('d'),
            sdfg.make_array_memlet('c')
    )
    state2.add_nedge(
            state2.add_access('e'),
            state2.add_access('f'),
            dace.Memlet('f[0] -> [0]')
    )
    sdfg.validate()
    return sdfg


def test_additional_access_on_interstate_edge():
    sdfg = _make_additional_access_on_interstate_edge_sdfg()
    assert len(sdfg.arrays) == 6

    # In this test `e` is not part of the exclusive set as it was in `test_access_only_on_interstate_edge`.
    #  The reason is because now there exists an AccessNode for `e`.
    expected_exclusive_set = {aname for aname in sdfg.arrays.keys() if aname != 'e'}
    exclusive_set = perform_scan(sdfg)
    assert len(exclusive_set[sdfg]) == 5
    assert exclusive_set[sdfg] == expected_exclusive_set


def _make_access_nested_nsdfg() -> dace.SDFG:
    sdfg = dace.SDFG('access_nested_nsdfg')

    for aname in 'ab':
        sdfg.add_array(
                aname,
                shape=(10,),
                dtype=dace.float64,
                transient=False,
        )

    state = sdfg.add_state(is_start_block=True)
    state.add_nedge(
            state.add_access('a'),
            state.add_access('b'),
            sdfg.make_array_memlet('a')
    )
    sdfg.validate()
    return sdfg


def _make_access_nested_sdfg() -> Tuple[dace.SDFG, dace.SDFG]:
    sdfg = dace.SDFG('access_nested_sdfg')
    nsdfg = _make_access_nested_nsdfg()

    for aname in 'ab':
        sdfg.add_array(
                aname,
                shape=(10,),
                dtype=dace.float64,
                transient=False,
        )

    state = sdfg.add_state(is_start_block=True)
    nsdfg_node = state.add_nested_sdfg(
            nsdfg,
            parent=sdfg,
            inputs={'a'},
            outputs={'b'},
            symbol_mapping={},
    )

    state.add_edge(
            state.add_access('a'),
            None,
            nsdfg_node,
            'a',
            sdfg.make_array_memlet('a'),
    )
    state.add_edge(
            nsdfg_node,
            'b',
            state.add_access('b'),
            None,
            sdfg.make_array_memlet('b'),
    )
    sdfg.validate()
    return sdfg, nsdfg


def test_access_nested_sdfg():
    sdfg, nested_sdfg = _make_access_nested_sdfg()
    assert all(len(nsdfg.arrays) == 2 for nsdfg in [sdfg, nested_sdfg])

    # In both SDFGs all data descriptors are exclusive.
    expected_exclusive_set = {'a', 'b'}
    exclusive_sets = perform_scan(sdfg)

    assert all(exclusive_sets[nsdfg] == expected_exclusive_set for nsdfg in [sdfg, nested_sdfg])


if __name__ == '__main__':
    test_all_exclusive_data_but_one_unused()
    test_multiple_access_same_state()
    test_multiple_single_access_node_same_state_sdfg()
    test_multiple_access_different_states()
    test_access_only_on_interstate_edge()
    test_additional_access_on_interstate_edge()
    test_access_nested_sdfg()
