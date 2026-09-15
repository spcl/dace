# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

from typing import Tuple
import dace
import numpy as np
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


def _make_write_merge_sdfg(sub1: str, sub2: str, n1: int, n2: int,
                           array_size: int) -> Tuple[dace.SDFG, dace.SDFGState, dace_nodes.MapExit]:
    # Two exit connectors of the same map, both writing directly into B, one connector
    # per tasklet so consolidate_edges_scope sees them as two independent write paths
    # to the same outer data container.
    sdfg = dace.SDFG(utility.unique_name('write_merge'))
    sdfg.add_array('B', [array_size], dace.float64)
    state = sdfg.add_state(is_start_block=True)

    me, mx = state.add_map('trivial', dict(__i='0:1'))
    code1 = '\n'.join(f'out[{k}] = 1.0' for k in range(n1))
    code2 = '\n'.join(f'out[{k}] = 2.0' for k in range(n2))
    t1 = state.add_tasklet('t1', {}, {'out': None}, code1)
    t2 = state.add_tasklet('t2', {}, {'out': None}, code2)
    w = state.add_write('B')

    state.add_nedge(me, t1, dace.Memlet())
    state.add_nedge(me, t2, dace.Memlet())
    mx.add_scope_connectors('1')
    mx.add_scope_connectors('2')
    state.add_edge(t1, 'out', mx, 'IN_1', dace.Memlet(f'B[{sub1}]'))
    state.add_edge(t2, 'out', mx, 'IN_2', dace.Memlet(f'B[{sub2}]'))
    state.add_edge(mx, 'OUT_1', w, None, dace.Memlet(f'B[{sub1}]'))
    state.add_edge(mx, 'OUT_2', w, None, dace.Memlet(f'B[{sub2}]'))

    sdfg.validate()
    return sdfg, state, mx


def test_consolidate_edges_refuses_overlapping_writes():
    # B[0:6] and B[3:9] overlap on [3:6) -- consolidating would fold two independently
    # ordered writes into one connector, which can silently change which write lands
    # in the overlap. The pass must refuse the merge and leave the SDFG untouched.
    sdfg, state, mx = _make_write_merge_sdfg('0:6', '3:9', n1=6, n2=6, array_size=10)
    edges_before = len(state.edges())
    in_conn_before = dict(mx.in_connectors)
    out_conn_before = dict(mx.out_connectors)

    ref = np.zeros(10)
    sdfg(B=ref)

    ret = consolidate_edges(sdfg, propagate=False)

    assert ret is None or ret == 0
    assert len(state.edges()) == edges_before
    assert dict(mx.in_connectors) == in_conn_before
    assert dict(mx.out_connectors) == out_conn_before
    sdfg.validate()

    # A refused merge must not perturb execution either.
    res = np.zeros(10)
    sdfg(B=res)
    assert np.array_equal(ref, res)


def test_consolidate_edges_merges_disjoint_writes():
    # B[0:4] and B[6:10] cannot overlap -- the legitimate case must still consolidate,
    # otherwise the overlap guard is just disabling the pass outright.
    sdfg, state, mx = _make_write_merge_sdfg('0:4', '6:10', n1=4, n2=4, array_size=10)
    edges_before = len(state.edges())

    ret = consolidate_edges(sdfg, propagate=False)

    assert ret == 1
    assert len(state.edges()) == edges_before - 1
    assert len(mx.in_connectors) == 1
    assert len(mx.out_connectors) == 1
    sdfg.validate()

    res = np.zeros(10)
    sdfg(B=res)
    assert np.array_equal(res[0:4], np.ones(4))
    assert np.array_equal(res[6:10], np.full(4, 2.0))


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


def test_consolidate_edges_refuses_reads_from_two_access_nodes():
    """Two access nodes of one container are two program points; the second is what
    sequences its read after the write feeding it."""
    sdfg = dace.SDFG('read_merge')
    sdfg.add_array('A', [8], dace.float64)
    sdfg.add_array('B', [8], dace.float64)
    state = sdfg.add_state()

    src = state.add_access('A')
    me, mx = state.add_map('m', dict(i='1:8'))
    tasklet = state.add_tasklet('t', {'x': None, 'y': None}, {'z': None}, 'z = x + y')
    state.add_memlet_path(src, me, tasklet, dst_conn='x', memlet=dace.Memlet('A[i]'))
    state.add_memlet_path(tasklet, mx, state.add_access('B'), src_conn='z', memlet=dace.Memlet('B[i]'))

    # The write feeding 'written' is what orders the A[0] read after it.
    written = state.add_access('A')
    state.add_edge(state.add_tasklet('w', {}, {'o': None}, 'o = 100.0'), 'o', written, None, dace.Memlet('A[0]'))
    state.add_memlet_path(written, me, tasklet, dst_conn='y', memlet=dace.Memlet('A[0]'))
    sdfg.validate()

    ref = np.arange(8, dtype=np.float64)
    expected = np.zeros(8)
    sdfg(A=ref.copy(), B=expected)

    assert consolidate_edges(sdfg, propagate=False) in (None, 0)
    sdfg.validate()

    got = np.zeros(8)
    sdfg(A=ref.copy(), B=got)
    assert np.array_equal(expected, got)


def test_consolidate_edges_folds_reads_of_one_written_access_node():
    """Reads taken through the SAME written access node share its program point, so they fold.

    Refusing them (every read of a written container, regardless of which node it came from)
    left one scope connector per read, and the next pass to route the whole body through a single
    connector -- ``nest_state_subgraph`` -- stranded the rest as dangling out-connectors.
    """
    sdfg = dace.SDFG('read_fold')
    sdfg.add_array('A', [8], dace.float64)
    sdfg.add_array('B', [8], dace.float64, transient=True)
    sdfg.add_array('C', [8], dace.float64)
    state = sdfg.add_state()

    # 'b' is written here, so every read below is ordered after that write by this one node.
    b = state.add_access('B')
    fill_entry, fill_exit = state.add_map('fill', dict(j='0:8'))
    fill = state.add_tasklet('fill', {'a': None}, {'o': None}, 'o = a * 2.0')
    state.add_memlet_path(state.add_read('A'), fill_entry, fill, dst_conn='a', memlet=dace.Memlet('A[j]'))
    state.add_memlet_path(fill, fill_exit, b, src_conn='o', memlet=dace.Memlet('B[j]'))

    me, mx = state.add_map('stencil', dict(i='1:7'))
    tasklet = state.add_tasklet('t', {'l': None, 'm': None, 'r': None}, {'z': None}, 'z = l + m + r')
    for conn, index in (('l', 'i - 1'), ('m', 'i'), ('r', 'i + 1')):
        state.add_memlet_path(b, me, tasklet, dst_conn=conn, memlet=dace.Memlet(f'B[{index}]'))
    state.add_memlet_path(tasklet, mx, state.add_write('C'), src_conn='z', memlet=dace.Memlet('C[i]'))
    sdfg.validate()

    assert len([c for c in me.out_connectors if c.startswith('OUT_')]) == 3, 'test setup: expected three reads'

    ref = np.arange(8, dtype=np.float64)
    expected = np.zeros(8)
    sdfg(A=ref.copy(), C=expected)

    assert consolidate_edges(sdfg, propagate=False) == 2
    sdfg.validate()
    assert [c for c in me.out_connectors if c.startswith('OUT_')] == ['OUT_B']
    assert len(state.in_edges(me)) == 1
    assert state.in_edges(me)[0].data.subset == dace_sbs.Range.from_string('0:8')

    got = np.zeros(8)
    sdfg(A=ref.copy(), C=got)
    assert np.array_equal(expected, got)


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
