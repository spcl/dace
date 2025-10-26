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
    use_inner_access_node: bool,
    use_non_standard_memlet: bool,
) -> Tuple[dace.SDFG, dace.SDFGState, dace_nodes.AccessNode, dace_nodes.MapEntry]:

    # Needs to be 5, to trigger the Memlet propagation bug (could actually also be
    #   less but greater than 2.
    N = 5

    sdfg = dace.SDFG(utility.unique_name("multi_input_usage"))
    state = sdfg.add_state(is_start_block=True)

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

            # NOTE: We need `(i % 2) == 0` the note in `_test_multi_use_value_input()`
            if use_non_standard_memlet and ((i % 2) == 0):
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

    sdfg, state, multi_use_value, me = _make_sdfg_multi_usage_input(use_inner_access_node=use_inner_access_node,
                                                                    use_non_standard_memlet=use_non_standard_memlet)

    initial_ac = utility.count_nodes(sdfg, dace_nodes.AccessNode, True)
    assert multi_use_value in initial_ac
    assert state.out_degree(multi_use_value) == 5
    assert all((oedge.data.src_subset == dace_sbs.Range.from_string("0:10") or oedge.data.src_subset ==
                dace_sbs.Range.from_string("1:11") or oedge.data.src_subset == dace_sbs.Range.from_string("2:12"))
               for oedge in state.out_edges(multi_use_value))
    assert all(
        state.out_degree(ac) == 1 and isinstance(ac, dace_nodes.AccessNode) for ac in state.source_nodes()
        if ac is not multi_use_value)
    assert all(state.in_degree(ac) == 1 for ac in state.sink_nodes())

    ref, res = utility.make_sdfg_args(sdfg)
    utility.compile_and_run_sdfg(sdfg, **ref)

    # NOTE: There is a bug in Memlet propagation that causes a test to fail if we
    #   use non-standard Memlets and inner AccessNode. The reason is that the largest
    #   subset, i.e. `__i + 2` is in a non-standard Memlet and the propagation fails
    #   to pick it up.
    ret = consolidate_edges(sdfg, propagate=False)
    sdfg.validate()
    assert ret > 0

    ac_after = utility.count_nodes(sdfg, dace_nodes.AccessNode, True)
    assert set(initial_ac) == set(ac_after)

    assert all(state.in_degree(ac) == 1 for ac in state.sink_nodes())
    assert state.out_degree(multi_use_value) == 1

    # Without `propagate=False` this test would fail if we use inner AccessNodes and
    #  non standard Memelts.
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


def _make_multi_use_value_output(
    use_inner_access_node: bool,
    use_non_standard_memlet: bool,
) -> Tuple[dace.SDFG, dace.SDFGState, dace_nodes.AccessNode]:

    sdfg = dace.SDFG(utility.unique_name("multi_input_usage"))
    state = sdfg.add_state(is_start_block=True)

    multi_output_data, _ = sdfg.add_array(
        "multi_output",
        shape=(12, 3),
        dtype=dace.float64,
        transient=False,
    )
    multi_output = state.add_access(multi_output_data)
    me, mx = state.add_map(
        "comp",
        ndrange={
            "__i": "0:10",
        },
    )

    for i in range(3):
        input_data = f"input_{i}"
        sdfg.add_array(
            input_data,
            shape=(10, ),
            dtype=dace.float64,
            transient=False,
        )
        tlet = state.add_tasklet(
            f"tlet_{i}",
            inputs={"__in1"},
            outputs={"__out"},
            code=f"__out = __in1 + 1.45 * ({i} + 1.3)",
        )

        state.add_edge(state.add_access(input_data), None, me, f"IN_{input_data}",
                       dace.Memlet(data=input_data, subset="0:10"))
        state.add_edge(me, f"OUT_{input_data}", tlet, "__in1", dace.Memlet(data=input_data, subset="__i"))
        me.add_scope_connectors(input_data)

        if use_inner_access_node:
            inner_data = f"inner_data_{i}"
            sdfg.add_scalar(
                inner_data,
                dtype=dace.float64,
                transient=True,
            )
            inner_ac = state.add_access(inner_data)

            data = multi_output_data
            subset = f"__i + {i}, {i}"
            other_subset = "0"

            if use_non_standard_memlet:
                data = inner_data
                subset, other_subset = other_subset, subset

            state.add_edge(tlet, "__out", inner_ac, None, dace.Memlet(f"{inner_data}[0]"))
            state.add_edge(inner_ac, None, mx, f"IN_output_{i}",
                           dace.Memlet(data=data, subset=subset, other_subset=other_subset))
        else:
            state.add_edge(tlet, "__out", mx, f"IN_output_{i}",
                           dace.Memlet(data=multi_output_data, subset=f"__i + {i}, {i}"))
        state.add_edge(mx, f"OUT_output_{i}", multi_output, None,
                       dace.Memlet(
                           data=multi_output_data,
                           subset=f"{i}:{i + 10}, {i}",
                       ))
        mx.add_scope_connectors(f"output_{i}")

    sdfg.validate()

    return sdfg, state, multi_output


def _test_multi_use_value_output(
    use_inner_access_node: bool,
    use_non_standard_memlet: bool,
):
    if use_non_standard_memlet and (not use_inner_access_node):
        # This combination is not useful.
        return

    sdfg, state, multi_output = _make_multi_use_value_output(
        use_inner_access_node=use_inner_access_node,
        use_non_standard_memlet=use_non_standard_memlet,
    )

    assert all(state.out_degree(sn) == 1 and isinstance(sn, dace_nodes.AccessNode) for sn in state.source_nodes())
    assert all(sn is multi_output and state.in_degree(sn) == 3 for sn in state.sink_nodes())
    assert all((iedge.data.dst_subset == dace_sbs.Range.from_string("0:10, 0") or iedge.data.dst_subset ==
                dace_sbs.Range.from_string("1:11, 1") or iedge.data.dst_subset == dace_sbs.Range.from_string("2:12, 2"))
               for iedge in state.in_edges(multi_output))
    initial_ac = utility.count_nodes(sdfg, dace_nodes.AccessNode, True)
    assert multi_output in initial_ac

    ref, res = utility.make_sdfg_args(sdfg)
    utility.compile_and_run_sdfg(sdfg, **ref)

    ret = consolidate_edges(sdfg, propagate=False)
    sdfg.validate()
    assert ret > 0

    ac_after = utility.count_nodes(sdfg, dace_nodes.AccessNode, True)
    assert set(ac_after) == set(initial_ac)
    assert all(state.out_degree(sn) == 1 and isinstance(sn, dace_nodes.AccessNode) for sn in state.source_nodes())

    assert state.in_degree(multi_output) == 1
    assert state.out_degree(multi_output) == 0
    assert all(iedge.data.dst_subset == dace_sbs.Range.from_string("0:12, 0:3")
               for iedge in state.in_edges(multi_output))

    utility.compile_and_run_sdfg(sdfg, **res)
    assert utility.compare_sdfg_res(ref=ref, res=res)


@pytest.mark.parametrize("use_non_standard_memlet", [True, False])
@pytest.mark.parametrize("use_inner_access_node", [True, False])
def test_multi_use_value_output(
    use_inner_access_node: bool,
    use_non_standard_memlet: bool,
):
    _test_multi_use_value_output(
        use_non_standard_memlet=use_non_standard_memlet,
        use_inner_access_node=use_inner_access_node,
    )


if __name__ == '__main__':
    test_consolidate_edges()
    for use_non_standard_memlet in [True, False]:
        for use_inner_access_node in [True, False]:
            _test_multi_use_value_input(
                use_inner_access_node=use_inner_access_node,
                use_non_standard_memlet=use_non_standard_memlet,
            )
            _test_multi_use_value_output(
                use_inner_access_node=use_inner_access_node,
                use_non_standard_memlet=use_non_standard_memlet,
            )
