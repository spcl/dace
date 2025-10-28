# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

from typing import Tuple, Union
import dace
from dace import subsets as dace_sbs
from dace.sdfg import nodes as dace_nodes
from dace.sdfg.utils import canonicalize_memlet_trees

import pytest

from .transformations import utility


def count_non_standard_memlets_in_scope(state: dace.SDFGState, ) -> int:

    nb_non_standard_memlets = 0
    scope_dict = state.scope_dict()
    for dnode in state.data_nodes():
        if scope_dict[dnode] is not None:
            continue

        for oedge in state.out_edges(dnode):
            if not isinstance(oedge.dst, dace_nodes.EntryNode):
                continue
            if not oedge.dst_conn.startswith("IN_"):
                continue
            nb_inspected = 0
            for mtree in state.memlet_tree(oedge).traverse_children():
                if mtree.edge.data.data != dnode.data:
                    nb_non_standard_memlets += 1
                nb_inspected += 1
            assert nb_inspected > 0

        for iedge in state.in_edges(dnode):
            if not isinstance(iedge.src, dace_nodes.ExitNode):
                continue
            nb_inspected = 0
            for mtree in state.memlet_tree(iedge).traverse_children():
                if mtree.edge.data.data != dnode.data:
                    nb_non_standard_memlets += 1
                nb_inspected += 1
            assert nb_inspected > 0

    return nb_non_standard_memlets


def _make_sdfg_multi_usage_input() -> Tuple[dace.SDFG, dace.SDFGState]:
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

        inner_ac = state.add_access(inner_data)
        data = multi_use_value_data
        subset = f"__i + {offset_in_i}"
        other_subset = "0"

        if (i % 2) == 0:
            data = inner_data
            subset, other_subset = other_subset, subset

        state.add_edge(me, f"OUT_muv_{i}", inner_ac, None,
                       dace.Memlet(
                           data=data,
                           subset=subset,
                           other_subset=other_subset,
                       ))
        state.add_edge(inner_ac, None, tlet, "__in1", dace.Memlet(f"{inner_data}[0]"))
        me.add_scope_connectors(f"muv_{i}")

        state.add_edge(iac, None, me, f"IN_{input_data}", dace.Memlet(f"{input_data}[0:10, 0:30]"))
        state.add_edge(me, f"OUT_{input_data}", tlet, "__in2", dace.Memlet(f"{input_data}[__i, __j]"))
        me.add_scope_connectors(input_data)

        state.add_edge(tlet, "__out", mx, f"IN_{output_data}", dace.Memlet(f"{output_data}[__i, __j]"))
        state.add_edge(mx, f"OUT_{output_data}", oac, None, dace.Memlet(f"{output_data}[0:10, 0:30]"))
        mx.add_scope_connectors(output_data)

    sdfg.validate()

    return sdfg, state


def test_multi_use_value_input():
    sdfg, state = _make_sdfg_multi_usage_input()

    initial_non_standard_memlets = count_non_standard_memlets_in_scope(state)
    assert initial_non_standard_memlets > 0

    ref, res = utility.make_sdfg_args(sdfg)
    utility.compile_and_run_sdfg(sdfg, **ref)

    ret = canonicalize_memlet_trees(sdfg)
    sdfg.validate()

    assert count_non_standard_memlets_in_scope(state) == 0
    utility.compile_and_run_sdfg(sdfg, **res)
    assert utility.compare_sdfg_res(ref=ref, res=res)
    assert ret == initial_non_standard_memlets


def _make_multi_use_value_output() -> Tuple[dace.SDFG, dace.SDFGState]:

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

        if i % 2 == 0:
            data = inner_data
            subset, other_subset = other_subset, subset

        state.add_edge(tlet, "__out", inner_ac, None, dace.Memlet(f"{inner_data}[0]"))
        state.add_edge(inner_ac, None, mx, f"IN_output_{i}",
                       dace.Memlet(data=data, subset=subset, other_subset=other_subset))
        state.add_edge(mx, f"OUT_output_{i}", multi_output, None,
                       dace.Memlet(
                           data=multi_output_data,
                           subset=f"{i}:{i + 10}, {i}",
                       ))
        mx.add_scope_connectors(f"output_{i}")
    sdfg.validate()

    return sdfg, state


def test_multi_use_value_output():
    sdfg, state = _make_multi_use_value_output()

    initial_non_standard_memlets = count_non_standard_memlets_in_scope(state)
    assert initial_non_standard_memlets > 0

    ref, res = utility.make_sdfg_args(sdfg)
    utility.compile_and_run_sdfg(sdfg, **ref)

    ret = canonicalize_memlet_trees(sdfg)
    sdfg.validate()

    assert count_non_standard_memlets_in_scope(state) == 0
    utility.compile_and_run_sdfg(sdfg, **res)
    assert utility.compare_sdfg_res(ref=ref, res=res)
    assert ret == initial_non_standard_memlets


if __name__ == '__main__':
    test_multi_use_value_input()
    test_multi_use_value_output()
