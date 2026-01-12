# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

from typing import Dict
import dace
from dace.sdfg.analysis.writeset_underapproximation import UnderapproximateWrites, UnderapproximateWritesDict
from dace.sdfg.utils import inline_control_flow_regions
from dace.subsets import Range
from dace.transformation.pass_pipeline import Pipeline

N = dace.symbol("N")
M = dace.symbol("M")
K = dace.symbol("K")


def test_2D_map_overwrites_2D_array():
    """
    2-dimensional map that fully overwrites 2-dimensional array
    --> Approximated write-set of Map to array equals shape of array
    """

    sdfg = dace.SDFG('twoD_map')
    sdfg.add_array('B', (M, N), dace.float64)
    map_state = sdfg.add_state('map')
    a1 = map_state.add_access('B')
    map_state.add_mapped_tasklet('overwrite_1',
                                 map_ranges={
                                     '_i': '0:N:1',
                                     '_j': '0:M:1'
                                 },
                                 inputs={},
                                 code='b = 5',
                                 outputs={'b': dace.Memlet('B[_j,_i]')},
                                 output_nodes={'B': a1},
                                 external_edges=True)

    pipeline = Pipeline([UnderapproximateWrites()])
    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]

    result = results[sdfg.cfg_id].approximation
    edge = map_state.in_edges(a1)[0]
    result_subset_list = result[edge].subset.subset_list
    result_subset = result_subset_list[0]
    expected_subset = Range.from_string('0:M, 0:N')
    assert (str(result_subset) == str(expected_subset))


def test_2D_map_added_indices():
    """
    2-dimensional array that writes to two-dimensional array with
    subscript expression that adds two indices
    --> Approximated write-set of Map is empty
    """

    sdfg = dace.SDFG("twoD_map")
    sdfg.add_array("B", (M, N), dace.float64)
    map_state = sdfg.add_state("map")
    a1 = map_state.add_access('B')
    map_state.add_mapped_tasklet("overwrite_1",
                                 map_ranges={
                                     '_i': '0:N:1',
                                     '_j': '0:M:1'
                                 },
                                 inputs={},
                                 code="b = 5",
                                 outputs={"b": dace.Memlet("B[_j,_i + _j]")},
                                 output_nodes={"B": a1},
                                 external_edges=True)

    pipeline = Pipeline([UnderapproximateWrites()])
    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]

    result = results[sdfg.cfg_id].approximation
    edge = map_state.in_edges(a1)[0]
    assert (len(result[edge].subset.subset_list) == 0)


def test_2D_map_multiplied_indices():
    """
    2-dimensional array that writes to two-dimensional array with
    subscript expression that multiplies two indices
    --> Approximated write-set of Map is empty
    """

    sdfg = dace.SDFG("twoD_map")
    sdfg.add_array("B", (M, N), dace.float64)
    map_state = sdfg.add_state("map")
    a1 = map_state.add_access('B')
    map_state.add_mapped_tasklet("overwrite_1",
                                 map_ranges={
                                     '_i': '0:N:1',
                                     '_j': '0:M:1'
                                 },
                                 inputs={},
                                 code="b = 5",
                                 outputs={"b": dace.Memlet("B[_j,_i * _j]")},
                                 output_nodes={"B": a1},
                                 external_edges=True)

    pipeline = Pipeline([UnderapproximateWrites()])
    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]

    result = results[sdfg.cfg_id].approximation
    edge = map_state.in_edges(a1)[0]
    assert (len(result[edge].subset.subset_list) == 0)


def test_1D_map_one_index_multiple_dims():
    """
    One-dimensional map that has the same index
    in two dimensions in a write-access
    --> Approximated write-set of Map is empty
    """

    sdfg = dace.SDFG("twoD_map")

    sdfg.add_array("B", (M, N), dace.float64)
    map_state = sdfg.add_state("map")
    a1 = map_state.add_access('B')
    map_state.add_mapped_tasklet("overwrite_1",
                                 map_ranges={'_j': '0:M:1'},
                                 inputs={},
                                 code="b = 5",
                                 outputs={"b": dace.Memlet("B[_j, _j]")},
                                 output_nodes={"B": a1},
                                 external_edges=True)

    pipeline = Pipeline([UnderapproximateWrites()])
    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]

    result = results[sdfg.cfg_id].approximation
    edge = map_state.in_edges(a1)[0]
    assert (len(result[edge].subset.subset_list) == 0)


def test_1D_map_one_index_squared():
    """
    One-dimensional map that multiplies the index
    in the subscript expression
    --> Approximated write-set of Map is empty
    """
    sdfg = dace.SDFG("twoD_map")
    sdfg.add_array("B", (M, ), dace.float64)
    map_state = sdfg.add_state("map")
    a1 = map_state.add_access('B')
    map_state.add_mapped_tasklet("overwrite_1",
                                 map_ranges={'_j': '0:M:1'},
                                 inputs={},
                                 code="b = 5",
                                 outputs={"b": dace.Memlet("B[_j * _j]")},
                                 output_nodes={"B": a1},
                                 external_edges=True)

    pipeline = Pipeline([UnderapproximateWrites()])
    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]

    result = results[sdfg.cfg_id].approximation
    edge = map_state.in_edges(a1)[0]
    assert (len(result[edge].subset.subset_list) == 0)


def test_map_tree_full_write():
    """
    Two maps nested in a map. Both nested maps overwrite the whole first dimension of the array
    together with the outer map the whole array is overwritten
    --> Approximated write-set of Map to array equals shape of array
    """

    sdfg = dace.SDFG("twoD_map")
    sdfg.add_array("B", (M, N), dace.float64)
    map_state = sdfg.add_state("map")
    a1 = map_state.add_access('B')
    map_entry, map_exit = map_state.add_map("outer_map", {"_i": '0:N:1'})
    map_exit.add_in_connector("IN_B")
    map_exit.add_out_connector("OUT_B")
    inner_map_entry_0, inner_map_exit_0 = map_state.add_map("inner_map_0", {"_j": '0:M:1'})
    inner_map_exit_0.add_in_connector("IN_B")
    inner_map_exit_0.add_out_connector("OUT_B")
    inner_map_entry_1, inner_map_exit_1 = map_state.add_map("inner_map_1", {"_j": '0:M:1'})
    inner_map_exit_1.add_in_connector("IN_B")
    inner_map_exit_1.add_out_connector("OUT_B")
    map_tasklet_0 = map_state.add_tasklet("map_tasklet_0", {}, {"b"}, "b = 1")
    map_tasklet_1 = map_state.add_tasklet("map_tasklet_1", {}, {"b"}, "b = 2")
    map_state.add_edge(map_entry, None, inner_map_entry_0, None, dace.Memlet())
    map_state.add_edge(inner_map_entry_0, None, map_tasklet_0, None, dace.Memlet())
    map_state.add_edge(map_tasklet_0, "b", inner_map_exit_0, "IN_B", dace.Memlet("B[_j, _i]"))
    inner_edge_0 = map_state.add_edge(inner_map_exit_0, "OUT_B", map_exit, "IN_B", dace.Memlet(data="B"))
    map_state.add_edge(map_entry, None, inner_map_entry_1, None, dace.Memlet())
    map_state.add_edge(inner_map_entry_1, None, map_tasklet_1, None, dace.Memlet())
    map_state.add_edge(map_tasklet_1, "b", inner_map_exit_1, "IN_B", dace.Memlet("B[_j, _i]"))
    inner_edge_1 = map_state.add_edge(inner_map_exit_1, "OUT_B", map_exit, "IN_B", dace.Memlet(data="B"))
    outer_edge = map_state.add_edge(map_exit, "OUT_B", a1, None, dace.Memlet(data="B"))

    pipeline = Pipeline([UnderapproximateWrites()])
    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]

    result = results[sdfg.cfg_id].approximation
    expected_subset_outer_edge = Range.from_string("0:M, 0:N")
    expected_subset_inner_edge = Range.from_string("0:M, _i")
    result_inner_edge_0 = result[inner_edge_0].subset.subset_list[0]
    result_inner_edge_1 = result[inner_edge_1].subset.subset_list[0]
    result_outer_edge = result[outer_edge].subset.subset_list[0]
    assert (str(result_inner_edge_0) == str(expected_subset_inner_edge))
    assert (str(result_inner_edge_1) == str(expected_subset_inner_edge))
    assert (str(result_outer_edge) == str(expected_subset_outer_edge))


def test_map_tree_no_write_multiple_indices():
    """
    Two maps nested in a map. Both nested writes contain an addition of
    indices in the subscript expression
    --> Approximated write-set of outer Map to array equals shape of array
    """

    sdfg = dace.SDFG("twoD_map")
    sdfg.add_array("B", (M, N), dace.float64)
    map_state = sdfg.add_state("map")
    a1 = map_state.add_access('B')
    map_entry, map_exit = map_state.add_map("outer_map", {"_i": '0:N:1'})
    map_exit.add_in_connector("IN_B")
    map_exit.add_out_connector("OUT_B")
    inner_map_entry_0, inner_map_exit_0 = map_state.add_map("inner_map_0", {"_j": '0:M:1'})
    inner_map_exit_0.add_in_connector("IN_B")
    inner_map_exit_0.add_out_connector("OUT_B")
    inner_map_entry_1, inner_map_exit_1 = map_state.add_map("inner_map_1", {"_j": '0:M:1'})
    inner_map_exit_1.add_in_connector("IN_B")
    inner_map_exit_1.add_out_connector("OUT_B")
    map_tasklet_0 = map_state.add_tasklet("map_tasklet_0", {}, {"b"}, "b = 1")
    map_tasklet_1 = map_state.add_tasklet("map_tasklet_1", {}, {"b"}, "b = 2")
    map_state.add_edge(map_entry, None, inner_map_entry_0, None, dace.Memlet())
    map_state.add_edge(inner_map_entry_0, None, map_tasklet_0, None, dace.Memlet())
    map_state.add_edge(map_tasklet_0, "b", inner_map_exit_0, "IN_B", dace.Memlet("B[_j + _i, _i]"))
    inner_edge_0 = map_state.add_edge(inner_map_exit_0, "OUT_B", map_exit, "IN_B", dace.Memlet(data="B"))
    map_state.add_edge(map_entry, None, inner_map_entry_1, None, dace.Memlet())
    map_state.add_edge(inner_map_entry_1, None, map_tasklet_1, None, dace.Memlet())
    map_state.add_edge(map_tasklet_1, "b", inner_map_exit_1, "IN_B", dace.Memlet("B[_j, _i + _j]"))
    inner_edge_1 = map_state.add_edge(inner_map_exit_1, "OUT_B", map_exit, "IN_B", dace.Memlet(data="B"))
    outer_edge = map_state.add_edge(map_exit, "OUT_B", a1, None, dace.Memlet(data="B"))

    pipeline = Pipeline([UnderapproximateWrites()])
    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]

    result = results[sdfg.cfg_id].approximation
    result_inner_edge_0 = result[inner_edge_0].subset.subset_list
    result_inner_edge_1 = result[inner_edge_1].subset.subset_list
    result_outer_edge = result[outer_edge].subset.subset_list
    assert (len(result_inner_edge_0) == 0)
    assert (len(result_inner_edge_1) == 0)
    assert (len(result_outer_edge) == 0)


def test_map_tree_multiple_indices_per_dimension():
    """
    Two maps nested in a map. One inner Map writes to array using multiple indices.
    The other inner map writes to array with affine indices
    --> Approximated write-set of outer Map to array equals shape of array
    """

    sdfg = dace.SDFG("twoD_map")
    sdfg.add_array("B", (M, N), dace.float64)
    map_state = sdfg.add_state("map")
    a1 = map_state.add_access('B')
    map_entry, map_exit = map_state.add_map("outer_map", {"_i": '0:N:1'})
    map_exit.add_in_connector("IN_B")
    map_exit.add_out_connector("OUT_B")
    inner_map_entry_0, inner_map_exit_0 = map_state.add_map("inner_map_0", {"_j": '0:M:1'})
    inner_map_exit_0.add_in_connector("IN_B")
    inner_map_exit_0.add_out_connector("OUT_B")
    inner_map_entry_1, inner_map_exit_1 = map_state.add_map("inner_map_1", {"_j": '0:M:1'})
    inner_map_exit_1.add_in_connector("IN_B")
    inner_map_exit_1.add_out_connector("OUT_B")
    map_tasklet_0 = map_state.add_tasklet("map_tasklet_0", {}, {"b"}, "b = 1")
    map_tasklet_1 = map_state.add_tasklet("map_tasklet_1", {}, {"b"}, "b = 2")
    map_state.add_edge(map_entry, None, inner_map_entry_0, None, dace.Memlet())
    map_state.add_edge(inner_map_entry_0, None, map_tasklet_0, None, dace.Memlet())
    map_state.add_edge(map_tasklet_0, "b", inner_map_exit_0, "IN_B", dace.Memlet("B[_j * _j, _i ]"))
    inner_edge_0 = map_state.add_edge(inner_map_exit_0, "OUT_B", map_exit, "IN_B", dace.Memlet(data="B"))
    map_state.add_edge(map_entry, None, inner_map_entry_1, None, dace.Memlet())
    map_state.add_edge(inner_map_entry_1, None, map_tasklet_1, None, dace.Memlet())
    map_state.add_edge(map_tasklet_1, "b", inner_map_exit_1, "IN_B", dace.Memlet("B[_j, _i]"))
    inner_edge_1 = map_state.add_edge(inner_map_exit_1, "OUT_B", map_exit, "IN_B", dace.Memlet(data="B"))
    outer_edge = map_state.add_edge(map_exit, "OUT_B", a1, None, dace.Memlet(data="B"))

    pipeline = Pipeline([UnderapproximateWrites()])
    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]

    result = results[sdfg.cfg_id].approximation
    expected_subset_outer_edge = Range.from_string("0:M, 0:N")
    expected_subset_inner_edge_1 = Range.from_string("0:M, _i")
    result_inner_edge_1 = result[inner_edge_1].subset.subset_list[0]
    result_outer_edge = result[outer_edge].subset.subset_list[0]
    assert (len(result[inner_edge_0].subset.subset_list) == 0)
    assert (str(result_inner_edge_1) == str(expected_subset_inner_edge_1))
    assert (str(result_outer_edge) == str(expected_subset_outer_edge))


def test_loop_in_map_multiplied_indices():
    """
    Loop nested in a map that writes to array. In the subscript expression
    of the write indices are multiplied
    --> Approximated write-set of Map to array is empty
    """

    @dace.program
    def loop(A: dace.float64[N, M]):
        for i in dace.map[0:N]:
            for j in range(M):
                A[i, j * i] = 0

    sdfg = loop.to_sdfg(simplify=True)

    # NOTE: Until the analysis is changed to make use of the new blocks, inline control flow for the analysis.
    inline_control_flow_regions(sdfg)

    pipeline = Pipeline([UnderapproximateWrites()])
    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]

    nsdfg = sdfg.cfg_list[1].parent_nsdfg_node
    map_state = sdfg.states()[0]
    result = results[sdfg.cfg_id].approximation
    edge = map_state.out_edges(nsdfg)[0]
    assert (len(result[edge].subset.subset_list) == 0)


def test_loop_in_map():
    """
    Loop nested in a map that writes to array. Outer map overwrites the array.
     --> Approximated write-set of Map to array equals shape of array
    """

    @dace.program
    def loop(A: dace.float64[N, M]):
        for i in dace.map[0:N]:
            for j in range(M):
                A[i, j] = 0

    sdfg = loop.to_sdfg(simplify=True)

    # NOTE: Until the analysis is changed to make use of the new blocks, inline control flow for the analysis.
    inline_control_flow_regions(sdfg)

    pipeline = Pipeline([UnderapproximateWrites()])
    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]

    map_state = sdfg.states()[0]
    edge = map_state.in_edges(map_state.data_nodes()[0])[0]
    result = results[sdfg.cfg_id].approximation
    expected_subset = Range.from_string("0:N, 0:M")
    assert (str(result[edge].subset.subset_list[0]) == str(expected_subset))


def test_map_in_loop():
    """
    Map nested in a loop that writes to array. Outer loop overwrites the array.
     --> Approximated write-set of Map to array equals shape of array
    """

    sdfg = dace.SDFG("nested")
    sdfg.add_array("B", (N, M), dace.float64)
    init = sdfg.add_state("init")
    guard = sdfg.add_state("guard")
    body = sdfg.add_state("body")
    end = sdfg.add_state("end")
    sdfg.add_edge(init, guard, dace.InterstateEdge(assignments={"j": "0"}))
    sdfg.add_edge(guard, body, dace.InterstateEdge(condition="j < N"))
    sdfg.add_edge(guard, end, dace.InterstateEdge(condition="not(j < N)"))
    sdfg.add_edge(body, guard, dace.InterstateEdge(assignments={"j": "j + 1"}))
    a1 = body.add_access("B")
    body.add_mapped_tasklet("overwrite_1",
                            map_ranges={'i': '0:M:1'},
                            inputs={},
                            code="b = 5",
                            outputs={"b": dace.Memlet("B[j, i]")},
                            output_nodes={"B": a1},
                            external_edges=True)

    pipeline = Pipeline([UnderapproximateWrites()])
    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]

    result = results[sdfg.cfg_id].loop_approximation
    expected_subset = Range.from_string("0:N, 0:M")
    assert (str(result[guard]["B"].subset.subset_list[0]) == str(expected_subset))


def test_map_in_loop_multiplied_indices_first_dimension():
    """
    Map nested in a loop that writes to array. Subscript expression
      of array access multiplies two indices in first dimension
    --> Approximated write-set of loop to array is empty
    """

    sdfg = dace.SDFG("nested")
    sdfg.add_array("B", (N, M), dace.float64)
    init = sdfg.add_state("init")
    guard = sdfg.add_state("guard")
    body = sdfg.add_state("body")
    end = sdfg.add_state("end")
    sdfg.add_edge(init, guard, dace.InterstateEdge(assignments={"j": "0"}))
    sdfg.add_edge(guard, body, dace.InterstateEdge(condition="j < N"))
    sdfg.add_edge(guard, end, dace.InterstateEdge(condition="not(j < N)"))
    sdfg.add_edge(body, guard, dace.InterstateEdge(assignments={"j": "j + 1"}))
    a1 = body.add_access("B")
    body.add_mapped_tasklet("overwrite_1",
                            map_ranges={'i': '0:M:1'},
                            inputs={},
                            code="b = 5",
                            outputs={"b": dace.Memlet("B[j * i, i]")},
                            output_nodes={"B": a1},
                            external_edges=True)

    pipeline = Pipeline([UnderapproximateWrites()])
    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]

    result = results[sdfg.cfg_id].loop_approximation
    assert (guard not in result.keys() or len(result[guard]) == 0)


def test_map_in_loop_multiplied_indices_second_dimension():
    """
    Map nested in a loop that writes to array. Subscript expression
      of array access multiplies two indices in second dimension
    --> Approximated write-set of loop to array is empty
    """
    sdfg = dace.SDFG("nested")
    sdfg.add_array("B", (N, M), dace.float64)
    init = sdfg.add_state("init")
    guard = sdfg.add_state("guard")
    body = sdfg.add_state("body")
    end = sdfg.add_state("end")
    sdfg.add_edge(init, guard, dace.InterstateEdge(assignments={"j": "0"}))
    sdfg.add_edge(guard, body, dace.InterstateEdge(condition="j < N"))
    sdfg.add_edge(guard, end, dace.InterstateEdge(condition="not(j < N)"))
    sdfg.add_edge(body, guard, dace.InterstateEdge(assignments={"j": "j + 1"}))
    a1 = body.add_access("B")
    body.add_mapped_tasklet("overwrite_1",
                            map_ranges={'i': '0:M:1'},
                            inputs={},
                            code="b = 5",
                            outputs={"b": dace.Memlet("B[j, i * j]")},
                            output_nodes={"B": a1},
                            external_edges=True)

    pipeline = Pipeline([UnderapproximateWrites()])
    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]

    result = results[sdfg.cfg_id].loop_approximation
    assert (guard not in result.keys() or len(result[guard]) == 0)


def test_nested_sdfg_in_map_nest():
    """
    Write in nested SDFG in two-dimensional map nest.
    --> should approximate write-set of map nest as shape of array."""

    @dace.program
    def nested_loop(A: dace.float64[M, N]):
        for i in dace.map[0:M]:
            for j in dace.map[0:N]:
                if A[0]:
                    A[i, j] = 1
                else:
                    A[i, j] = 2
                A[i, j] = A[i, j] * A[i, j]

    sdfg = nested_loop.to_sdfg(simplify=True)

    # NOTE: Until the analysis is changed to make use of the new blocks, inline control flow for the analysis.
    inline_control_flow_regions(sdfg)

    pipeline = Pipeline([UnderapproximateWrites()])
    result = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]
    write_approx = result[sdfg.cfg_id].approximation
    # find write set
    accessnode = None
    write_set = None
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.AccessNode):
            if node.data == "A":
                accessnode = node
    for edge, memlet in write_approx.items():
        if edge.dst is accessnode:
            write_set = memlet.subset

    assert (str(write_set) == "0:M, 0:N")


def test_loop_in_nested_sdfg_in_map_partial_write():
    """
    Write in nested SDFG in two-dimensional map nest.
    Nested map does not iterate over shape of second array dimension.
    --> should approximate write-set of map nest precisely."""

    @dace.program
    def nested_loop(A: dace.float64[M, N]):
        for i in dace.map[0:M]:
            for j in range(2, N, 1):
                if A[0]:
                    A[i, j] = 1
                else:
                    A[i, j] = 2
                A[i, j] = A[i, j] * A[i, j]

    sdfg = nested_loop.to_sdfg(simplify=True)

    # NOTE: Until the analysis is changed to make use of the new blocks, inline control flow for the analysis.
    inline_control_flow_regions(sdfg)

    pipeline = Pipeline([UnderapproximateWrites()])
    result = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]

    write_approx = result[sdfg.cfg_id].approximation
    # find write set
    accessnode = None
    write_set = None
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.AccessNode):
            if node.data == "A":
                accessnode = node
    for edge, memlet in write_approx.items():
        if edge.dst is accessnode:
            write_set = memlet.subset
    assert (str(write_set) == "0:M, 0:N - 2")


def test_map_in_nested_sdfg_in_map():
    """
    Write in Map nested in a nested SDFG nested in a map.
    --> should approximate write-set of loop nest precisely."""

    @dace.program
    def nested_loop(A: dace.float64[M, N]):
        for i in dace.map[0:M]:
            if A[0]:
                A[i, :] = 1
            else:
                A[i, :] = 2
            A[i, :] = 0

    sdfg = nested_loop.to_sdfg(simplify=True)

    # NOTE: Until the analysis is changed to make use of the new blocks, inline control flow for the analysis.
    inline_control_flow_regions(sdfg)

    pipeline = Pipeline([UnderapproximateWrites()])
    result = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]

    write_approx = result[sdfg.cfg_id].approximation
    # find write set
    accessnode = None
    write_set = None
    for node, parent in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.AccessNode):
            if node.data == "A" and parent.out_degree(node) == 0:
                accessnode = node
    for edge, memlet in write_approx.items():
        if edge.dst is accessnode:
            write_set = memlet.subset
    assert (str(write_set) == "0:M, 0:N")


def test_nested_sdfg_in_map_branches():
    """
    Nested SDFG that overwrites second dimension of array conditionally.
    --> should approximate write-set of map as empty
    """
    # No, should be approximated precisely - at least certainly with CF regions..?

    @dace.program
    def nested_loop(A: dace.float64[M, N]):
        for i in dace.map[0:M]:
            if A[0]:
                A[i, :] = 1
            else:
                A[i, :] = 2

    sdfg = nested_loop.to_sdfg(simplify=True)

    # NOTE: Until the analysis is changed to make use of the new blocks, inline control flow for the analysis.
    inline_control_flow_regions(sdfg)

    pipeline = Pipeline([UnderapproximateWrites()])
    result: Dict[int, UnderapproximateWritesDict] = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]

    write_approx = result[sdfg.cfg_id].approximation
    # find write set
    accessnode = None
    write_set = None
    for node, parent in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.AccessNode):
            if node.data == "A" and parent.out_degree(node) == 0:
                accessnode = node
    for edge, memlet in write_approx.items():
        if edge.dst is accessnode:
            write_set = memlet.subset.subset_list
    assert (not write_set)


def test_simple_loop_overwrite():
    """
    simple loop that overwrites a one-dimensional array
    --> should approximate write-set of loop as shape of array
    """

    sdfg = dace.SDFG("simple_loop")
    sdfg.add_array("A", [N], dace.int64)
    init = sdfg.add_state("init")
    end = sdfg.add_state("end")
    loop_body = sdfg.add_state("loop_body")
    _, guard, _ = sdfg.add_loop(init, loop_body, end, "i", "0", "i < N", "i + 1")
    a0 = loop_body.add_access("A")
    loop_tasklet = loop_body.add_tasklet("overwrite", {}, {"a"}, "a = 0")
    loop_body.add_edge(loop_tasklet, "a", a0, None, dace.Memlet("A[i]"))

    pipeline = Pipeline([UnderapproximateWrites()])
    result: UnderapproximateWritesDict = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__][sdfg.cfg_id]

    assert (str(result.loop_approximation[guard]["A"].subset) == str(Range.from_array(sdfg.arrays["A"])))


def test_loop_2D_overwrite():
    """
    Two-dimensional loop nest overwrites a two-dimensional array
    --> should approximate write-set of loop nest as shape of array
    """

    sdfg = dace.SDFG("loop_2D_overwrite")
    sdfg.add_array("A", [M, N], dace.int64)
    init = sdfg.add_state("init")
    end = sdfg.add_state("end")
    loop_body = sdfg.add_state("loop_body")
    loop_before_1 = sdfg.add_state("loop_before_1")
    loop_after_1 = sdfg.add_state("loop_after_1")
    _, guard2, _ = sdfg.add_loop(loop_before_1, loop_body, loop_after_1, "i", "0", "i < N", "i + 1")
    _, guard1, _ = sdfg.add_loop(init, loop_before_1, end, "j", "0", "j < M", "j + 1", loop_after_1)
    a0 = loop_body.add_access("A")
    loop_tasklet = loop_body.add_tasklet("overwrite", {}, {"a"}, "a = 0")
    loop_body.add_edge(loop_tasklet, "a", a0, None, dace.Memlet("A[j,i]"))

    pipeline = Pipeline([UnderapproximateWrites()])
    result = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__][sdfg.cfg_id].loop_approximation

    assert (str(result[guard1]["A"].subset) == str(Range.from_array(sdfg.arrays["A"])))
    assert (str(result[guard2]["A"].subset) == "j, 0:N")


def test_loop_2D_propagation_gap_symbolic():
    """
    Three nested loops that overwrite two dimensional array.
    Innermost loop is surrounded by loop that doesn't iterate
    over array range and is potentially empty.
    --> should approximate write-set to array of outer loop as empty
    """

    sdfg = dace.SDFG("loop_2D_no_overwrite")
    sdfg.add_array("A", [M, N], dace.int64)
    init = sdfg.add_state("init")
    end = sdfg.add_state("end")
    loop_body = sdfg.add_state("loop_body")
    loop_before_1 = sdfg.add_state("loop_before_1")
    loop_after_1 = sdfg.add_state("loop_after_1")
    loop_before_2 = sdfg.add_state("loop_before_2")
    loop_after_2 = sdfg.add_state("loop_after_2")
    _, guard3, _ = sdfg.add_loop(loop_before_1, loop_body, loop_after_1, "i", "0", "i < N", "i + 1")  # inner-most loop
    _, guard2, _ = sdfg.add_loop(loop_before_2, loop_before_1, loop_after_2, "k", "0", "k < K", "k + 1",
                                 loop_after_1)  # second-inner-most loop
    _, guard1, _ = sdfg.add_loop(init, loop_before_2, end, "j", "0", "j < M", "j + 1", loop_after_2)  # outer-most loop
    a0 = loop_body.add_access("A")
    loop_tasklet = loop_body.add_tasklet("overwrite", {}, {"a"}, "a = 0")
    loop_body.add_edge(loop_tasklet, "a", a0, None, dace.Memlet("A[j,i]"))

    pipeline = Pipeline([UnderapproximateWrites()])
    result = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__][sdfg.cfg_id].loop_approximation

    assert ("A" not in result[guard1].keys())
    assert ("A" not in result[guard2].keys())
    assert (str(result[guard3]["A"].subset) == "j, 0:N")


def test_2_loops_overwrite():
    """
    2 loops one after another overwriting an array
    --> should approximate write-set to array of both loops as shape of array
    """

    sdfg = dace.SDFG("two_loops_overwrite")
    sdfg.add_array("A", [N], dace.int64)
    init = sdfg.add_state("init")
    end = sdfg.add_state("end")
    loop_body_1 = sdfg.add_state("loop_body_1")
    loop_body_2 = sdfg.add_state("loop_body_2")
    _, guard_1, after_state = sdfg.add_loop(init, loop_body_1, None, "i", "0", "i < N", "i + 1")
    _, guard_2, _ = sdfg.add_loop(after_state, loop_body_2, end, "i", "0", "i < N", "i + 1")
    a0 = loop_body_1.add_access("A")
    loop_tasklet_1 = loop_body_1.add_tasklet("overwrite", {}, {"a"}, "a = 0")
    loop_body_1.add_edge(loop_tasklet_1, "a", a0, None, dace.Memlet("A[i]"))
    a1 = loop_body_2.add_access("A")
    loop_tasklet_2 = loop_body_2.add_tasklet("overwrite", {}, {"a"}, "a = 0")
    loop_body_2.add_edge(loop_tasklet_2, "a", a1, None, dace.Memlet("A[i]"))

    pipeline = Pipeline([UnderapproximateWrites()])
    result = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__][sdfg.cfg_id].loop_approximation

    assert (str(result[guard_1]["A"].subset) == str(Range.from_array(sdfg.arrays["A"])))
    assert (str(result[guard_2]["A"].subset) == str(Range.from_array(sdfg.arrays["A"])))


def test_loop_2D_overwrite_propagation_gap_non_empty():
    """
    Three nested loops that overwrite two-dimensional array.
    Innermost loop is surrounded by a loop that doesn't iterate
    over array range but over a non-empty constant range.
    --> should approximate write-set to array of loop nest as shape of array
    """

    sdfg = dace.SDFG("loop_2D_no_overwrite")
    sdfg.add_array("A", [M, N], dace.int64)
    init = sdfg.add_state("init")
    end = sdfg.add_state("end")
    loop_body = sdfg.add_state("loop_body")
    loop_before_1 = sdfg.add_state("loop_before_1")
    loop_after_1 = sdfg.add_state("loop_after_1")
    loop_before_2 = sdfg.add_state("loop_before_2")
    loop_after_2 = sdfg.add_state("loop_after_2")
    _, guard3, _ = sdfg.add_loop(loop_before_1, loop_body, loop_after_1, "i", "0", "i < N", "i + 1")
    _, guard2, _ = sdfg.add_loop(loop_before_2, loop_before_1, loop_after_2, "k", "0", "k < 10", "k + 1", loop_after_1)
    _, guard1, _ = sdfg.add_loop(init, loop_before_2, end, "j", "0", "j < M", "j + 1", loop_after_2)
    a0 = loop_body.add_access("A")
    loop_tasklet = loop_body.add_tasklet("overwrite", {}, {"a"}, "a = 0")
    loop_body.add_edge(loop_tasklet, "a", a0, None, dace.Memlet("A[j,i]"))

    pipeline = Pipeline([UnderapproximateWrites()])
    result = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__][sdfg.cfg_id].loop_approximation

    assert (str(result[guard1]["A"].subset) == str(Range.from_array(sdfg.arrays["A"])))
    assert (str(result[guard2]["A"].subset) == "j, 0:N")
    assert (str(result[guard3]["A"].subset) == "j, 0:N")


def test_loop_nest_multiplied_indices():
    """
    three nested loops that write to two dimensional array.
    The subscript expression is a multiplication of two indices
    -> should approximate write-sets of loops as empty
    """

    sdfg = dace.SDFG("loop_2D_no_overwrite")
    sdfg.add_array("A", [N, N], dace.int64)
    init = sdfg.add_state("init")
    end = sdfg.add_state("end")
    loop_body = sdfg.add_state("loop_body")
    loop_before_1 = sdfg.add_state("loop_before_1")
    loop_after_1 = sdfg.add_state("loop_after_1")
    loop_before_2 = sdfg.add_state("loop_before_2")
    loop_after_2 = sdfg.add_state("loop_after_2")
    _, guard3, _ = sdfg.add_loop(loop_before_1, loop_body, loop_after_1, "i", "0", "i < N", "i + 1")
    _, guard2, _ = sdfg.add_loop(loop_before_2, loop_before_1, loop_after_2, "k", "0", "k < 10", "k + 1", loop_after_1)
    _, guard1, _ = sdfg.add_loop(init, loop_before_2, end, "j", "0", "j < M", "j + 1", loop_after_2)
    a0 = loop_body.add_access("A")
    loop_tasklet = loop_body.add_tasklet("overwrite", {}, {"a"}, "a = 0")
    loop_body.add_edge(loop_tasklet, "a", a0, None, dace.Memlet("A[i,i*j]"))

    pipeline = Pipeline([UnderapproximateWrites()])
    result = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__][sdfg.cfg_id].loop_approximation

    assert (guard1 not in result.keys() or "A" not in result[guard1].keys())
    assert (guard2 not in result.keys() or "A" not in result[guard2].keys())
    assert (guard3 not in result.keys() or "A" not in result[guard3].keys() or not result[guard3]['A'].subset)


def test_loop_nest_empty_nested_loop():
    """
    three nested loops that write to two dimensional array.
    the innermost loop is surrounded by a loop that iterates over an empty range.
    --> Approximated write-set to array of outer loop is empty.
    Approximated write-set to array of innermost loop is equal to shape of array
    """

    sdfg = dace.SDFG("loop_2D_no_overwrite")
    sdfg.add_array("A", [M, N], dace.int64)
    init = sdfg.add_state("init")
    end = sdfg.add_state("end")
    loop_body = sdfg.add_state("loop_body")
    loop_before_1 = sdfg.add_state("loop_before_1")
    loop_after_1 = sdfg.add_state("loop_after_1")
    loop_before_2 = sdfg.add_state("loop_before_2")
    loop_after_2 = sdfg.add_state("loop_after_2")
    _, guard3, _ = sdfg.add_loop(loop_before_1, loop_body, loop_after_1, "i", "0", "i < N", "i + 1")
    _, guard2, _ = sdfg.add_loop(loop_before_2, loop_before_1, loop_after_2, "k", "0", "k < 0", "k + 1", loop_after_1)
    _, guard1, _ = sdfg.add_loop(init, loop_before_2, end, "j", "0", "j < M", "j + 1", loop_after_2)
    a0 = loop_body.add_access("A")
    loop_tasklet = loop_body.add_tasklet("overwrite", {}, {"a"}, "a = 0")
    loop_body.add_edge(loop_tasklet, "a", a0, None, dace.Memlet("A[j,i]"))

    pipeline = Pipeline([UnderapproximateWrites()])
    result = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__][sdfg.cfg_id].loop_approximation

    assert (guard1 not in result.keys() or "A" not in result[guard1].keys())
    assert (guard2 not in result.keys() or "A" not in result[guard2].keys())
    assert (str(result[guard3]["A"].subset) == "j, 0:N")


def test_loop_nest_inner_loop_conditional():
    """
    Loop nested in another loop. Nested loop is in a branch and overwrites the array.
        --> should approximate write-set to array of outer loop as empty
        and write-set to array of inner loop equal to shape of array
    """
    sdfg = dace.SDFG("loop_2D_branch")
    sdfg.add_array("A", [N], dace.int64)
    init = sdfg.add_state("init")
    end = sdfg.add_state("end")
    loop_body = sdfg.add_state("loop_body")
    if_guard = sdfg.add_state("if_guard")
    if_merge = sdfg.add_state("if_merge")
    loop_before_2 = sdfg.add_state("loop_before_2")
    loop_after_2 = sdfg.add_state("loop_after_2")
    _, guard2, _ = sdfg.add_loop(loop_before_2, loop_body, loop_after_2, "k", "0", "k < N", "k + 1")
    _, guard1, _ = sdfg.add_loop(init, if_guard, end, "j", "0", "j < M", "j + 1", if_merge)
    sdfg.add_edge(if_guard, loop_before_2, dace.InterstateEdge(condition="j % 2 == 0"))
    sdfg.add_edge(if_guard, if_merge, dace.InterstateEdge(condition="j % 2 == 1"))
    sdfg.add_edge(loop_after_2, if_merge, dace.InterstateEdge())
    a0 = loop_body.add_access("A")
    loop_tasklet = loop_body.add_tasklet("overwrite", {}, {"a"}, "a = 0")
    loop_body.add_edge(loop_tasklet, "a", a0, None, dace.Memlet("A[k]"))

    pipeline = Pipeline([UnderapproximateWrites()])
    result = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__][sdfg.cfg_id].loop_approximation

    assert (guard1 not in result.keys() or "A" not in result[guard1].keys())
    assert (guard2 in result.keys() and "A" in result[guard2].keys() and str(result[guard2]['A'].subset) == "0:N")


def test_loop_in_nested_sdfg_in_map_multiplied_indices():
    """
    Loop in nested SDFG nested in map. The subscript of the write multiplies two indices
    --> should approximate write-set of loop as empty
    """

    @dace.program
    def nested_loop(A: dace.float64[M, N]):
        for i in dace.map[0:M]:
            for j in range(N):
                A[i + 1, j * i] = 1

    sdfg = nested_loop.to_sdfg(simplify=True)

    # NOTE: Until the analysis is changed to make use of the new blocks, inline control flow for the analysis.
    inline_control_flow_regions(sdfg)

    pipeline = Pipeline([UnderapproximateWrites()])
    result = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]

    write_approx = result[sdfg.cfg_id].approximation
    write_set = None
    accessnode = None
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.AccessNode):
            if node.data == "A":
                accessnode = node
    for edge, memlet in write_approx.items():
        if edge.dst is accessnode:
            write_set = memlet.subset
    assert (not write_set.subset_list)


def test_loop_in_nested_sdfg_simple():
    """
    Loop nested in a map that overwrites two-dimensional array
    --> should approximate write-set of map to full shape of array
    """

    @dace.program
    def nested_loop(A: dace.float64[M, N]):
        for i in dace.map[0:M]:
            for j in range(N):
                A[i, j] = 1

    sdfg = nested_loop.to_sdfg(simplify=True)

    # NOTE: Until the analysis is changed to make use of the new blocks, inline control flow for the analysis.
    inline_control_flow_regions(sdfg)

    pipeline = Pipeline([UnderapproximateWrites()])
    result = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]

    # find write set
    write_approx = result[sdfg.cfg_id].approximation
    accessnode = None
    write_set = None
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.AccessNode):
            if node.data == "A":
                accessnode = node
    for edge, memlet in write_approx.items():
        if edge.dst is accessnode:
            write_set = memlet.subset

    assert (str(write_set) == "0:M, 0:N")


def test_loop_break():
    """
    Loop that has a break statement writing to array.
        --> Approximated write-set of loop to array is empty
    """

    sdfg = dace.SDFG("loop_2D_no_overwrite")
    sdfg.add_array("A", [N], dace.int64)
    init = sdfg.add_state("init", is_start_block=True)
    loop_body_0 = sdfg.add_state("loop_body_0")
    loop_body_1 = sdfg.add_state("loop_body_1")
    loop_after_1 = sdfg.add_state("loop_after_1")
    _, guard3, _ = sdfg.add_loop(init, loop_body_0, loop_after_1, "i", "0", "i < N", "i + 1", loop_body_1)
    sdfg.add_edge(loop_body_0, loop_after_1, dace.InterstateEdge(condition="i > 10"))
    sdfg.add_edge(loop_body_0, loop_body_1, dace.InterstateEdge(condition="not(i > 10)"))
    a0 = loop_body_1.add_access("A")
    loop_tasklet = loop_body_1.add_tasklet("overwrite", {}, {"a"}, "a = 0")
    loop_body_1.add_edge(loop_tasklet, "a", a0, None, dace.Memlet("A[i]"))

    pipeline = Pipeline([UnderapproximateWrites()])
    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]

    result = results[sdfg.cfg_id].loop_approximation
    assert (guard3 not in result.keys() or "A" not in result[guard3].keys())


def test_constant_multiplicative_2D():
    """
    Array is accessed via index that is multiplied with a constant.
    --> should approximate write-set precisely
    """

    A = dace.data.Array(dace.int64, (N, M))
    subset = Range.from_string("i,3*j")
    i_subset = Range.from_string("0:N:1")
    j_subset = Range.from_string("0:M:1")
    memlet = dace.Memlet(None, "A", subset)
    memlets = [memlet]

    propagated_memlet = UnderapproximateWrites()._underapproximate_subsets(memlets, A, ["j"], j_subset, None, True)
    propagated_memlet = UnderapproximateWrites()._underapproximate_subsets([propagated_memlet], A, ["i"], i_subset,
                                                                           None, True)

    propagated_subset = propagated_memlet.subset.subset_list[0]
    expected_subset = Range.from_string("0:N:1, 0:3*M - 2:3")
    propagated_string = str(propagated_subset)
    expected_string = str(expected_subset)
    assert (propagated_string == expected_string)


def test_affine_2D():
    """
    Array is accessed via affine subscript expresion.
    --> should approximate write-set precisely
    """

    A = dace.data.Array(dace.int64, (N, M))
    subset = Range.from_string("i,3 * j + 3")
    i_subset = Range.from_string("0:N:1")
    j_subset = Range.from_string("0:M:1")
    memlet = dace.Memlet(None, "A", subset)
    memlets = [memlet]

    propagated_memlet = UnderapproximateWrites()._underapproximate_subsets(memlets, A, ["j"], j_subset, None, True)
    propagated_memlet = UnderapproximateWrites()._underapproximate_subsets([propagated_memlet], A, ["i"], i_subset,
                                                                           None, True)

    propagated_subset = propagated_memlet.subset.subset_list[0]
    expected_subset = Range.from_string("0:N:1, 3 : 3 * M + 1 : 3")
    propagated_string = str(propagated_subset)
    expected_string = str(expected_subset)
    assert (propagated_string == expected_string)


def test_multiplied_variables():
    """
    Two indices are multiplied in subscript expression
    --> should fall back to empty subset
    """

    A = dace.data.Array(dace.int64, (M, ))
    subset = Range.from_string("i * j")
    i_subset = Range.from_string("0:N:1")
    j_subset = Range.from_string("0:M:1")
    memlet = dace.Memlet(None, "A", subset)
    memlets = [memlet]

    propagated_memlet = UnderapproximateWrites()._underapproximate_subsets(memlets, A, ["j"], j_subset, None, True)
    propagated_memlet = UnderapproximateWrites()._underapproximate_subsets([propagated_memlet], A, ["i"], i_subset,
                                                                           None, True)

    assert (not propagated_memlet.subset.subset_list)


def test_one_variable_in_2dimensions():
    """
    One index occurs in two dimensions
    --> should fall back to empty subset
    """

    A = dace.data.Array(dace.int64, (N, M))
    subset = Range.from_string("i, i")
    i_subset = Range.from_string("0:N:1")
    j_subset = Range.from_string("0:M:1")
    memlet = dace.Memlet(None, "A", subset)
    memlets = [memlet]

    propagated_memlet = UnderapproximateWrites()._underapproximate_subsets(memlets, A, ["j"], j_subset, None, True)
    propagated_memlet = UnderapproximateWrites()._underapproximate_subsets([propagated_memlet], A, ["i"], i_subset,
                                                                           None, True)

    assert (not propagated_memlet.subset.subset_list)


def test_negative_step():
    A = dace.data.Array(dace.int64, (N, M))
    subset = Range.from_string("i, j")
    i_subset = Range.from_string("0:N:1")
    j_subset = Range.from_string("M:0:-1")
    memlet = dace.Memlet(None, "A", subset)
    memlets = [memlet]

    propagated_memlet = UnderapproximateWrites()._underapproximate_subsets(memlets, A, ["j"], j_subset, None, True)
    propagated_memlet = UnderapproximateWrites()._underapproximate_subsets([propagated_memlet], A, ["i"], i_subset,
                                                                           None, True)

    propagated_subset = propagated_memlet.subset.subset_list[0]
    expected_subset = Range.from_string("0:N:1,M:0:-1")
    propagated_string = str(propagated_subset)
    expected_string = str(expected_subset)
    assert (propagated_string == expected_string)


def test_step_not_one():
    """
    Array is accessed via index that is defined
    over Range with stepsize > 1.
    --> should approximate write-set precisely
"""

    A = dace.data.Array(dace.int64, (N, M))
    subset = Range.from_string("i")
    i_subset = Range.from_string("0:N:3")
    memlet = dace.Memlet(None, "A", subset)
    memlets = [memlet]

    propagated_memlet = UnderapproximateWrites()._underapproximate_subsets(memlets, A, ["i"], i_subset, None, True)
    propagated_subset = propagated_memlet.subset.subset_list[0]

    expected_subset = Range.from_string("0:N:3")
    propagated_string = str(propagated_subset)
    expected_string = str(expected_subset)
    assert (propagated_string == expected_string)


if __name__ == '__main__':
    test_nested_sdfg_in_map_branches()
    test_map_in_nested_sdfg_in_map()
    test_loop_in_nested_sdfg_in_map_partial_write()
    test_nested_sdfg_in_map_nest()
    test_map_in_loop_multiplied_indices_first_dimension()
    test_map_in_loop_multiplied_indices_second_dimension()
    test_map_in_loop()
    test_loop_in_map_multiplied_indices()
    test_loop_in_map()
    test_map_tree_full_write()
    test_2D_map_overwrites_2D_array()
    test_2D_map_added_indices()
    test_2D_map_multiplied_indices()
    test_1D_map_one_index_multiple_dims()
    test_1D_map_one_index_squared()
    test_map_tree_multiple_indices_per_dimension()
    test_map_tree_no_write_multiple_indices()
    test_step_not_one()
    test_one_variable_in_2dimensions()
    test_affine_2D()
    test_constant_multiplicative_2D()
    test_multiplied_variables()
    test_loop_in_nested_sdfg_simple()
    test_loop_nest_inner_loop_conditional()
    test_loop_nest_empty_nested_loop()
    test_simple_loop_overwrite()
    test_loop_2D_overwrite()
    test_loop_2D_overwrite_propagation_gap_non_empty()
    test_2_loops_overwrite()
    test_loop_2D_propagation_gap_symbolic()
    test_loop_nest_multiplied_indices()
    test_loop_in_nested_sdfg_in_map_multiplied_indices()
    test_loop_break()
    test_negative_step()
