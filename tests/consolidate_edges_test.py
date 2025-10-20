# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

from typing import Tuple
import dace
from dace import subsets as dace_sbs
from dace.sdfg import nodes as dace_nodes
from dace.sdfg.utils import consolidate_edges

import pytest

from .transformations import utility


def _make_cetest_sdfg():
    sdfg = dace.SDFG('cetest')
    sdfg.add_array('A', [50], dace.float32)
    sdfg.add_array('B', [48], dace.float32)
    state = sdfg.add_state()

    r = state.add_read('A')
    me, mx = state.add_map('map', dict(i='1:49'))
    t = state.add_tasklet('op', {'a', 'b', 'c'}, {'out'}, 'out = a + b + c')
    w = state.add_write('B')

    state.add_memlet_path(r, me, t, dst_conn='a', memlet=dace.Memlet.simple('A', 'i-1'))
    state.add_memlet_path(r, me, t, dst_conn='b', memlet=dace.Memlet.simple('A', 'i'))
    state.add_memlet_path(r, me, t, dst_conn='c', memlet=dace.Memlet.simple('A', 'i+1'))
    state.add_memlet_path(t, mx, w, src_conn='out', memlet=dace.Memlet.simple('B', 'i-1'))

    sdfg.validate()

    return sdfg, state


def test_consolidate_edges():
    sdfg, state = _make_cetest_sdfg()
    assert len(state.edges()) == 8
    consolidate_edges(sdfg)
    assert len(state.edges()) == 6


def _make_sdfg_multi_usage_input(
    N: int,
    use_inner_access_node: bool,
    use_non_standard_memlet: bool,
) -> Tuple[dace.SDFG, dace.SDFGState, dace_nodes.AccessNode, dace_nodes.MapEntry]:
    sdfg = dace.SDFG(utility.unique_name("multi_input_usage"))
    state = sdfg.add_state(is_start_block=True)

    if use_inner_access_node and use_non_standard_memlet:
        assert N >= 2, "Needed for alteration"

    multi_use_value_data, _ = sdfg.add_array(
        "multi_use_value",
        shape=(12, ),
        dtype=dace.float64,
        transient=False,
    )
    multi_use_value = state.add_access(multi_use_value_data)
    me, mx = state.add_map(
        "comp",
        ndrange={
            "__i": "0:10",
            "__j": "0:30",
        },
    )

    for i in range(N):
        input_data = f"input_{i}"
        output_data = f"output_{i}"
        offset_in_i = i % 3
        for name in [input_data, output_data]:
            sdfg.add_array(
                name,
                shape=(10, 30),
                dtype=dace.float64,
                transient=False,
            )

        if use_inner_access_node:
            inner_data = f"inner_data_{i}"
            sdfg.add_scalar(
                inner_data,
                dtype=dace.float64,
                transient=True,
            )

        iac, oac = (state.add_access(name) for name in [input_data, output_data])
        tlet = state.add_tasklet(
            f"tlet_{i}",
            inputs={"__in1", "__in2"},
            outputs={"__out"},
            code="__out = __in1 + __in2",
        )

        state.add_edge(multi_use_value, None, me, f"IN_muv_{i}",
                       dace.Memlet(f"{multi_use_value_data}[{offset_in_i}:{offset_in_i + 10}]"))

        if use_inner_access_node:
            inner_ac = state.add_access(inner_data)
            data = multi_use_value_data
            subset = f"__i + {offset_in_i}"
            other_subset = "0"

            # NOTE: If we use `(i % 2) == 0` then we hit a bug in Memlet propagation.
            #   Since it is only propagated to `0:11`, this is because the `__i + 2`
            #   is somehow overlooked.
            if use_non_standard_memlet and ((i % 2) == 1):
                data = inner_data
                subset, other_subset = other_subset, subset

            state.add_edge(me, f"OUT_muv_{i}", inner_ac, None,
                           dace.Memlet(
                               data=data,
                               subset=subset,
                               other_subset=other_subset,
                           ))
            state.add_edge(inner_ac, None, tlet, "__in1", dace.Memlet(f"{inner_data}[0]"))
        else:
            state.add_edge(me, f"OUT_muv_{i}", tlet, "__in1",
                           dace.Memlet(f"{multi_use_value_data}[__i + {offset_in_i}]"))
        me.add_scope_connectors(f"muv_{i}")

        state.add_edge(iac, None, me, f"IN_{input_data}", dace.Memlet(f"{input_data}[0:10, 0:30]"))
        state.add_edge(me, f"OUT_{input_data}", tlet, "__in2", dace.Memlet(f"{input_data}[__i, __j]"))
        me.add_scope_connectors(input_data)

        state.add_edge(tlet, "__out", mx, f"IN_{output_data}", dace.Memlet(f"{output_data}[__i, __j]"))
        state.add_edge(mx, f"OUT_{output_data}", oac, None, dace.Memlet(f"{output_data}[0:10, 0:30]"))
        mx.add_scope_connectors(output_data)

    sdfg.validate()

    return sdfg, state, multi_use_value, me


def _test_multi_use_value_input(
    use_inner_access_node: bool,
    use_non_standard_memlet: bool,
):
    if use_non_standard_memlet and (not use_inner_access_node):
        # This combination does not make sense.
        return

    N = 5
    sdfg, state, multi_use_value, me = _make_sdfg_multi_usage_input(N=N,
                                                                    use_inner_access_node=use_inner_access_node,
                                                                    use_non_standard_memlet=use_non_standard_memlet)

    initial_ac = utility.count_nodes(sdfg, dace_nodes.AccessNode, True)
    assert multi_use_value in initial_ac
    assert state.out_degree(multi_use_value) == N
    assert all((oedge.data.src_subset == dace_sbs.Range.from_string("0:10") or oedge.data.src_subset ==
                dace_sbs.Range.from_string("1:11") or oedge.data.src_subset == dace_sbs.Range.from_string("2:12"))
               for oedge in state.out_edges(multi_use_value))
    assert all(
        state.out_degree(ac) == 1 and isinstance(ac, dace_nodes.AccessNode) for ac in state.source_nodes()
        if ac is not multi_use_value)
    assert all(state.in_degree(ac) == 1 for ac in state.sink_nodes())

    ref, res = utility.make_sdfg_args(sdfg)
    utility.compile_and_run_sdfg(sdfg, **ref)

    consolidate_edges(sdfg)

    ac_after = utility.count_nodes(sdfg, dace_nodes.AccessNode, True)
    assert set(initial_ac) == set(ac_after)

    assert all(state.in_degree(ac) == 1 for ac in state.sink_nodes())
    assert state.out_degree(multi_use_value) == 1

    # NOTE: This test might fail because of a bug in Memlet propagation.
    #   The test was changed such that it is __not__ hit.
    assert all(oedge.data.src_subset == dace_sbs.Range.from_string("0:12")
               for oedge in state.out_edges(multi_use_value))

    utility.compile_and_run_sdfg(sdfg, **res)
    assert utility.compare_sdfg_res(ref=ref, res=res)


@pytest.mark.parametrize("use_inner_access_node", [True, False])
@pytest.mark.parametrize("use_non_standard_memlet", [True, False])
def test_multi_use_value_input(
    use_inner_access_node: bool,
    use_non_standard_memlet: bool,
):
    _test_multi_use_value_input(use_inner_access_node=use_inner_access_node,
                                use_non_standard_memlet=use_non_standard_memlet)


def test_multi_use_value_output():
    assert False, "Implement me."


if __name__ == '__main__':
    test_consolidate_edges()
    for use_non_standard_memlet in [True, False]:
        for use_inner_access_node in [True, False]:
            _test_multi_use_value_input(
                use_inner_access_node=use_inner_access_node,
                use_non_standard_memlet=use_non_standard_memlet,
            )
