# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Dict, Set, Tuple
import dace
from dace.transformation.passes.analysis import FindSingleUseData


def perform_scan(sdfg: dace.SDFG) -> Dict[dace.SDFG, Set[str]]:
    scanner = FindSingleUseData()
    return scanner.apply_pass(sdfg, None)


def _make_all_single_use_data_but_one_unused_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG('all_single_use_data_but_one_unused_sdfg')
    state1 = sdfg.add_state(is_start_block=True)
    state2 = sdfg.add_state_after(state1)

    for name in 'abcde':
        sdfg.add_array(
            name,
            shape=(10, 10),
            dtype=dace.float64,
            transient=False,
        )

    state1.add_nedge(state1.add_access('a'), state1.add_access('b'), sdfg.make_array_memlet('a'))
    state2.add_nedge(state2.add_access('c'), state2.add_access('d'), sdfg.make_array_memlet('c'))
    sdfg.validate()
    return sdfg


def test_all_single_use_data_but_one_unused():
    sdfg = _make_all_single_use_data_but_one_unused_sdfg()
    assert len(sdfg.arrays) == 5

    # Because `e` is not used inside the SDFG, it is not included in the returned set,
    #  all other descriptors are included because they appear once.
    expected_single_use_set = {aname for aname in sdfg.arrays.keys() if aname != 'e'}

    single_use_set = perform_scan(sdfg)

    assert len(single_use_set[sdfg]) == 4
    assert len(single_use_set) == 1
    assert single_use_set[sdfg] == expected_single_use_set


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

    state.add_nedge(state.add_access('a'), state.add_access('b'), sdfg.make_array_memlet('a'))
    state.add_nedge(state.add_access('a'), state.add_access('d'), sdfg.make_array_memlet('a'))
    sdfg.validate()
    return sdfg


def test_multiple_access_same_state():
    sdfg = _make_multiple_access_same_state_sdfg()
    assert len(sdfg.arrays) == 3

    # `a` is not single use data because there are multiple access nodes for it
    #  in a single state.
    expected_single_use_set = {aname for aname in sdfg.arrays.keys() if aname != 'a'}
    single_use_set = perform_scan(sdfg)
    assert len(single_use_set) == 1
    assert len(single_use_set[sdfg]) == 2
    assert expected_single_use_set == single_use_set[sdfg]


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
    state.add_nedge(a, state.add_access('b'), sdfg.make_array_memlet('a'))
    state.add_nedge(a, state.add_access('d'), sdfg.make_array_memlet('a'))
    assert state.out_degree(a) == 2
    sdfg.validate()
    return sdfg


def test_multiple_single_access_node_same_state_sdfg() -> dace.SDFG:
    sdfg = _make_multiple_single_access_node_same_state_sdfg()
    assert len(sdfg.arrays) == 3

    # Unlike `test_multiple_access_same_state()` here `a` is included in the single use
    #  set, because, there is only a single AccessNode, that is used multiple times,
    #  i.e. has an output degree larger than one.
    expected_single_use_set = sdfg.arrays.keys()
    single_use_set = perform_scan(sdfg)
    assert len(single_use_set) == 1
    assert len(single_use_set[sdfg]) == 3
    assert expected_single_use_set == single_use_set[sdfg]


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
    state1.add_nedge(state1.add_access('b'), state1.add_access('a'), sdfg.make_array_memlet('a'))
    state2.add_nedge(state2.add_access('d'), state2.add_access('a'), sdfg.make_array_memlet('a'))
    sdfg.validate()
    return sdfg


def test_multiple_access_different_states():
    sdfg = _make_multiple_access_different_states_sdfg()
    assert len(sdfg.arrays) == 3

    # `a` is not included in the single use set, because it is used in two different states.
    single_use_set = perform_scan(sdfg)
    expected_single_use_set = {aname for aname in sdfg.arrays.keys() if aname != 'a'}
    assert len(single_use_set) == 1
    assert len(single_use_set[sdfg]) == 2
    assert expected_single_use_set == single_use_set[sdfg]


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

    state1.add_nedge(state1.add_access('a'), state1.add_access('b'), sdfg.make_array_memlet('a'))
    state2.add_nedge(state2.add_access('c'), state2.add_access('d'), sdfg.make_array_memlet('c'))
    sdfg.validate()
    return sdfg


def test_access_only_on_interstate_edge():
    sdfg = _make_access_only_on_interstate_edge_sdfg()
    assert len(sdfg.arrays) == 5

    # `e` is only accessed on the interstate edge. So it is technically an single use
    #  data. But by definition we handle this case as non single_use.
    expected_single_use_set = {aname for aname in sdfg.arrays.keys() if aname != 'e'}
    single_use_set = perform_scan(sdfg)
    assert len(single_use_set) == 1
    assert len(single_use_set[sdfg]) == 4
    assert single_use_set[sdfg] == expected_single_use_set


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

    state1.add_nedge(state1.add_access('a'), state1.add_access('b'), sdfg.make_array_memlet('a'))
    state2.add_nedge(state2.add_access('c'), state2.add_access('d'), sdfg.make_array_memlet('c'))
    state2.add_nedge(state2.add_access('e'), state2.add_access('f'), dace.Memlet('f[0] -> [0]'))
    sdfg.validate()
    return sdfg


def test_additional_access_on_interstate_edge():
    sdfg = _make_additional_access_on_interstate_edge_sdfg()
    assert len(sdfg.arrays) == 6

    # There is one AccessNode for `a`, but as in `test_access_only_on_interstate_edge`
    #  `e` is also used on the inter state edge, so it is not included.
    expected_single_use_set = {aname for aname in sdfg.arrays.keys() if aname != 'e'}
    single_use_set = perform_scan(sdfg)
    assert len(single_use_set) == 1
    assert len(single_use_set[sdfg]) == 5
    assert single_use_set[sdfg] == expected_single_use_set


def _make_access_nested_nsdfg() -> dace.SDFG:
    sdfg = dace.SDFG('access_nested_nsdfg')

    for aname in 'ab':
        sdfg.add_array(
            aname,
            shape=(10, ),
            dtype=dace.float64,
            transient=False,
        )

    state = sdfg.add_state(is_start_block=True)
    state.add_nedge(state.add_access('a'), state.add_access('b'), sdfg.make_array_memlet('a'))
    sdfg.validate()
    return sdfg


def _make_access_nested_sdfg() -> Tuple[dace.SDFG, dace.SDFG]:
    sdfg = dace.SDFG('access_nested_sdfg')
    nsdfg = _make_access_nested_nsdfg()

    for aname in 'ab':
        sdfg.add_array(
            aname,
            shape=(10, ),
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

    # In the top and the nested SDFG `a` and `b` are both used once, so for
    #  both they are included in the single use set.
    #  Essentially tests if there is separation between the two.
    expected_single_use_set = {'a', 'b'}
    single_use_sets = perform_scan(sdfg)

    assert len(single_use_sets) == 2
    assert all(single_use_sets[nsdfg] == expected_single_use_set for nsdfg in [sdfg, nested_sdfg])


def _make_conditional_block_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG("conditional_block_sdfg")

    for name in ["a", "b", "c", "d", "cond", "cond2"]:
        sdfg.add_scalar(name, dtype=dace.bool_ if name.startswith("cond") else dace.float64, transient=False)
    sdfg.arrays["b"].transient = True
    sdfg.arrays["cond2"].transient = True

    entry_state = sdfg.add_state("entry", is_start_block=True)
    entry_state.add_nedge(entry_state.add_access("a"), entry_state.add_access("b"), sdfg.make_array_memlet("a"))
    cond_tasklet: dace.nodes.Tasklet = entry_state.add_tasklet(
        "cond_processing",
        inputs={"__in"},
        code="__out = not __in",
        outputs={"__out"},
    )
    entry_state.add_edge(entry_state.add_access("cond"), None, cond_tasklet, "__in", dace.Memlet("cond[0]"))
    entry_state.add_edge(cond_tasklet, "__out", entry_state.add_access("cond2"), None, dace.Memlet("cond2[0]"))

    if_region = dace.sdfg.state.ConditionalBlock("if")
    sdfg.add_node(if_region)
    sdfg.add_edge(entry_state, if_region, dace.InterstateEdge())

    then_body = dace.sdfg.state.ControlFlowRegion("then_body", sdfg=sdfg)
    tstate = then_body.add_state("true_branch", is_start_block=True)
    tstate.add_nedge(tstate.add_access("b"), tstate.add_access("c"), sdfg.make_array_memlet("b"))
    if_region.add_branch(dace.sdfg.state.CodeBlock("cond2"), then_body)

    else_body = dace.sdfg.state.ControlFlowRegion("else_body", sdfg=sdfg)
    fstate = else_body.add_state("false_branch", is_start_block=True)
    fstate.add_nedge(fstate.add_access("b"), fstate.add_access("d"), sdfg.make_array_memlet("d"))
    if_region.add_branch(dace.sdfg.state.CodeBlock("not (cond2)"), else_body)
    sdfg.validate()
    return sdfg


def test_conditional_block():
    sdfg = _make_conditional_block_sdfg()

    # `b` is not in no single use data, because there are three AccessNodes for it.
    #  `cond2` is no single use data, although there is exactly one AccessNode for
    #  it, it is used in the condition expression.
    expected_single_use_set = {a for a in sdfg.arrays.keys() if a not in ["b", "cond2"]}
    single_use_set = perform_scan(sdfg)

    assert len(single_use_set) == 1
    assert single_use_set[sdfg] == expected_single_use_set


if __name__ == '__main__':
    test_all_single_use_data_but_one_unused()
    test_multiple_access_same_state()
    test_multiple_single_access_node_same_state_sdfg()
    test_multiple_access_different_states()
    test_access_only_on_interstate_edge()
    test_additional_access_on_interstate_edge()
    test_access_nested_sdfg()
    test_conditional_block()
