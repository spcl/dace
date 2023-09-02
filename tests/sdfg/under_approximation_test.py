import dace
from dace.sdfg.propagation_underapproximation import UnderapproximateWrites
from dace.subsets import Range
from dace.transformation.pass_pipeline import Pipeline

N = dace.symbol("N")
M = dace.symbol("M")
K = dace.symbol("K")

pipeline = Pipeline([UnderapproximateWrites()])


def test_2D_map_full_write():
    """2-dimensional map that fully overwrites 2-dimensional array --> propagate full range"""

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
                                 outputs={
                                     'b': dace.Memlet('B[_j,_i]')
                                 },
                                 output_nodes={'B': a1},
                                 external_edges=True
                                 )

    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]

    result = results['approximation']
    edge = map_state.in_edges(a1)[0]
    result_subset_list = result[edge].subset.subset_list
    result_subset = result_subset_list[0]
    expected_subset = Range.from_string('0:M, 0:N')
    assert (result_subset.__str__() == expected_subset.__str__())


def test_2D_map_added_indices_no_propagation():
    """2-dimensional array that writes to two-dimensional array with 
    subscript expression that adds two indices --> empty subset"""

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
                                 outputs={
                                     "b": dace.Memlet("B[_j,_i + _j]")
                                 },
                                 output_nodes={"B": a1},
                                 external_edges=True
                                 )

    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]

    result = results["approximation"]
    edge = map_state.in_edges(a1)[0]
    assert (len(result[edge].subset.subset_list) == 0)


def test_2D_map_multiplied_indices_no_propagation():
    """2-dimensional array that writes to two-dimensional array with 
    subscript expression that multiplies two indices --> empty subset"""

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
                                 outputs={
                                     "b": dace.Memlet("B[_j,_i * _j]")
                                 },
                                 output_nodes={"B": a1},
                                 external_edges=True
                                 )

    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]

    result = results["approximation"]
    edge = map_state.in_edges(a1)[0]
    assert (len(result[edge].subset.subset_list) == 0)


def test_1D_map_one_index_multiple_dims_no_propagation():
    """one dimensional map that has the same index 
    in two dimensions in a write-access --> no propagation"""

    sdfg = dace.SDFG("twoD_map")

    sdfg.add_array("B", (M, N), dace.float64)
    map_state = sdfg.add_state("map")
    a1 = map_state.add_access('B')
    map_state.add_mapped_tasklet("overwrite_1",
                                 map_ranges={
                                     '_j': '0:M:1'
                                 },
                                 inputs={},
                                 code="b = 5",
                                 outputs={
                                     "b": dace.Memlet("B[_j, _j]")
                                 },
                                 output_nodes={"B": a1},
                                 external_edges=True
                                 )

    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]

    result = results["approximation"]
    edge = map_state.in_edges(a1)[0]
    assert (len(result[edge].subset.subset_list) == 0)

def test_1D_map_one_index_squared_no_fission():
    """one dimensional map that multiplies the index 
    in the subscript expression --> no propagation"""
    sdfg = dace.SDFG("twoD_map")
    sdfg.add_array("B", (M,), dace.float64)
    map_state = sdfg.add_state("map")
    a1 = map_state.add_access('B')
    map_state.add_mapped_tasklet("overwrite_1",
                                 map_ranges={
                                     '_j': '0:M:1'
                                 },
                                 inputs={},
                                 code="b = 5",
                                 outputs={
                                     "b": dace.Memlet("B[_j * _j]")
                                 },
                                 output_nodes={"B": a1},
                                 external_edges=True
                                 )

    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]

    result = results["approximation"]
    edge = map_state.in_edges(a1)[0]
    assert (len(result[edge].subset.subset_list) == 0)

def test_map_tree_full_write():
    """two maps nested in map. both maps overwrite the whole first dimension of the array
    together with the outer map the whole array is overwritten"""

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

    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]

    result = results["approximation"]
    expected_subset_outer_edge = Range.from_string("0:M, 0:N")
    expected_subset_inner_edge = Range.from_string("0:M, _i")
    result_inner_edge_0 = result[inner_edge_0].subset.subset_list[0]
    result_inner_edge_1 = result[inner_edge_1].subset.subset_list[0]
    result_outer_edge = result[outer_edge].subset.subset_list[0]

    assert (result_inner_edge_0.__str__() == expected_subset_inner_edge.__str__())
    assert (result_inner_edge_1.__str__() == expected_subset_inner_edge.__str__())
    assert (result_outer_edge.__str__() == expected_subset_outer_edge.__str__())


def test_map_tree_no_write_multiple_indices():
    """same as test_map_tree_full_write but now the accesses 
    in both nested maps add two indices --> no propagation"""
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

    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]

    result = results["approximation"]
    result_inner_edge_0 = result[inner_edge_0].subset.subset_list
    result_inner_edge_1 = result[inner_edge_1].subset.subset_list
    result_outer_edge = result[outer_edge].subset.subset_list
    assert (len(result_inner_edge_0) == 0)
    assert (len(result_inner_edge_1) == 0)
    assert (len(result_outer_edge) == 0)


def test_map_tree_multiple_indices_per_dimension_full_write():
    """same as test_map_tree_full_write but one of the inner maps squares index _j --> full write"""

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

    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]

    result = results["approximation"]
    expected_subset_outer_edge = Range.from_string("0:M, 0:N")
    expected_subset_inner_edge_1 = Range.from_string("0:M, _i")
    result_inner_edge_1 = result[inner_edge_1].subset.subset_list[0]
    result_outer_edge = result[outer_edge].subset.subset_list[0]
    assert (len(result[inner_edge_0].subset.subset_list) == 0)
    assert (result_inner_edge_1.__str__() == expected_subset_inner_edge_1.__str__())
    assert (result_outer_edge.__str__() == expected_subset_outer_edge.__str__())


def test_loop_in_map_no_write():
    """loop in a map and indices are multiplied in subscript expression -> no propagation"""

    @dace.program
    def loop(A: dace.float64[N, M]):
        for i in dace.map[0:N]:
            for j in range(M):
                A[i, j * i] = 0
    sdfg = loop.to_sdfg()

    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]

    nsdfg = sdfg.sdfg_list[1].parent_nsdfg_node
    map_state = sdfg.states()[0]
    result = results["approximation"]
    edge = map_state.out_edges(nsdfg)[0]
    assert (len(result[edge].subset.subset_list) == 0)


def test_loop_in_map_full_write():
    """loop in map and both together fully overwrite the array"""

    @dace.program
    def loop(A: dace.float64[N, M]):
        for i in dace.map[0:N]:
            for j in range(M):
                A[i, j] = 0
    sdfg = loop.to_sdfg()

    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]

    map_state = sdfg.states()[0]
    edge = map_state.in_edges(map_state.data_nodes()[0])[0]
    result = results["approximation"]
    expected_subset = Range.from_string("0:N, 0:M")
    assert (result[edge].subset.subset_list[0].__str__() == expected_subset.__str__())


def test_map_in_loop_full_write():
    """map in loop, together they overwrite a two dimensional array"""

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
                            map_ranges={
                                'i': '0:M:1'
                            },
                            inputs={},
                            code="b = 5",
                            outputs={
                                "b": dace.Memlet("B[j, i]")
                            },
                            output_nodes={"B": a1},
                            external_edges=True
                            )

    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]

    result = results["loop_approximation"]
    expected_subset = Range.from_string("0:N, 0:M")
    assert (result[guard]["B"].subset.subset_list[0].__str__() == expected_subset.__str__())


def test_map_in_loop_no_write_0():
    """map in loop. Subscript expression of array access in loop multiplies two indicies --> no propagation"""

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
                            map_ranges={
                                'i': '0:M:1'
                            },
                            inputs={},
                            code="b = 5",
                            outputs={
                                "b": dace.Memlet("B[j * i, i]")
                            },
                            output_nodes={"B": a1},
                            external_edges=True
                            )

    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]

    result = results["loop_approximation"]
    assert (guard not in result.keys() or len(result[guard]) == 0)

def test_map_in_loop_no_write_1():
    """same as test_map_in_loop_no_write_0 but for other dimension"""
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
                            map_ranges={
                                'i': '0:M:1'
                            },
                            inputs={},
                            code="b = 5",
                            outputs={
                                "b": dace.Memlet("B[j, i * j]")
                            },
                            output_nodes={"B": a1},
                            external_edges=True
                            )

    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]

    result = results["loop_approximation"]
    assert (guard not in result.keys() or len(result[guard]) == 0)

def test_nested_sdfg_in_map_full_write():
    @dace.program
    def nested_loop(A: dace.float64[M, N]):
        for i in dace.map[0:M]:
            for j in dace.map[0:N]:
                if A[0]:
                    A[i, j] = 1
                else:
                    A[i, j] = 2
                A[i, j] = A[i, j] * A[i, j]
    sdfg = nested_loop.to_sdfg()

    result = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]

    write_approx = result["approximation"]
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

    assert (write_set.__str__() == "0:M, 0:N")


def test_loop_in_nested_sdfg_in_map_partial_write():
    @dace.program
    def nested_loop(A: dace.float64[M, N]):
        for i in dace.map[0:M]:
            for j in range(2, N, 1):
                if A[0]:
                    A[i, j] = 1
                else:
                    A[i, j] = 2
                A[i, j] = A[i, j] * A[i, j]
    sdfg = nested_loop.to_sdfg()

    result = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]

    write_approx = result["approximation"]
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
    assert (write_set.__str__() == "0:M, 0:N - 2")


def test_nested_sdfg_in_map_full_write_1():
    @dace.program
    def nested_loop(A: dace.float64[M, N]):
        for i in dace.map[0:M]:
            if A[0]:
                A[i, :] = 1
            else:
                A[i, :] = 2
            A[i, :] = 0
    sdfg = nested_loop.to_sdfg()

    result = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]

    write_approx = result["approximation"]
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
    assert (write_set.__str__() == "0:M, 0:N")


def test_nested_sdfg_in_map_branches_no_write():
    @dace.program
    def nested_loop(A: dace.float64[M, N]):
        for i in dace.map[0:M]:
            if A[0]:
                A[i, :] = 1
            else:
                A[i, :] = 2
    sdfg = nested_loop.to_sdfg()

    result = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]

    write_approx = result["approximation"]
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
    assert (not write_set.__str__() == "0:M, 0:N")


def test_nested_sdfg_in_map_branches_no_write_1():
    @dace.program
    def nested_loop(A: dace.float64[M, N]):
        for i in dace.map[0:M]:
            if A[0]:
                A[i + 1, :] = 1
            else:
                A[i + 1, :] = 2
    sdfg = nested_loop.to_sdfg()

    result = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]

    write_approx = result["approximation"]
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

    assert (not write_set.__str__() == "0:M, 0:N")


def test_constant_multiplicative_2D():
    A = dace.data.Array(dace.int64, (N, M))
    subset = Range.from_string("i,3*j")
    i_subset = Range.from_string("0:N:1")
    j_subset = Range.from_string("0:M:1")
    memlet = dace.Memlet(None, "A", subset)
    memlets = [memlet]

    propagated_memlet = UnderapproximateWrites().propagate_subset(memlets, A, ["j"], j_subset, None, True)
    propagated_memlet = UnderapproximateWrites().propagate_subset([propagated_memlet], A, ["i"], i_subset, None, True)

    propagated_subset = propagated_memlet.subset.subset_list[0]
    expected_subset = Range.from_string("0:N:1, 0:3*M - 2:3")
    propagated_string = propagated_subset.__str__()
    expected_string = expected_subset.__str__()
    assert (propagated_string == expected_string)


def test_affine_2D():
    A = dace.data.Array(dace.int64, (N, M))
    subset = Range.from_string("i,3 * j + 3")
    i_subset = Range.from_string("0:N:1")
    j_subset = Range.from_string("0:M:1")
    memlet = dace.Memlet(None, "A", subset)
    memlets = [memlet]

    propagated_memlet = UnderapproximateWrites().propagate_subset(memlets, A, ["j"], j_subset, None, True)
    propagated_memlet = UnderapproximateWrites().propagate_subset([propagated_memlet], A, ["i"], i_subset, None, True)

    propagated_subset = propagated_memlet.subset.subset_list[0]
    expected_subset = Range.from_string("0:N:1, 3 : 3 * M + 1 : 3")
    propagated_string = propagated_subset.__str__()
    expected_string = expected_subset.__str__()
    assert (propagated_string == expected_string)


def test_multiplied_variables():
    A = dace.data.Array(dace.int64, (M,))
    subset = Range.from_string("i * j")
    i_subset = Range.from_string("0:N:1")
    j_subset = Range.from_string("0:M:1")
    memlet = dace.Memlet(None, "A", subset)
    memlets = [memlet]

    propagated_memlet = UnderapproximateWrites().propagate_subset(memlets, A, ["j"], j_subset, None, True)
    propagated_memlet = UnderapproximateWrites().propagate_subset([propagated_memlet], A, ["i"], i_subset, None, True)

    assert (not propagated_memlet.subset.subset_list)


def test_one_variable_in_2dimensions():
    A = dace.data.Array(dace.int64, (N, M))
    subset = Range.from_string("i, i")
    i_subset = Range.from_string("0:N:1")
    j_subset = Range.from_string("0:M:1")
    memlet = dace.Memlet(None, "A", subset)
    memlets = [memlet]

    propagated_memlet = UnderapproximateWrites().propagate_subset(memlets, A, ["j"], j_subset, None, True)
    propagated_memlet = UnderapproximateWrites().propagate_subset([propagated_memlet], A, ["i"], i_subset, None, True)

    assert (not propagated_memlet.subset.subset_list)


def test_negative_step():
    A = dace.data.Array(dace.int64, (N, M))
    subset = Range.from_string("i, j")
    i_subset = Range.from_string("0:N:1")
    j_subset = Range.from_string("M:0:-1")
    memlet = dace.Memlet(None, "A", subset)
    memlets = [memlet]

    propagated_memlet = UnderapproximateWrites().propagate_subset(memlets, A, ["j"], j_subset, None, True)
    propagated_memlet = UnderapproximateWrites().propagate_subset([propagated_memlet], A, ["i"], i_subset, None, True)

    propagated_subset = propagated_memlet.subset.subset_list[0]
    expected_subset = Range.from_string("0:N:1,0:M:1")
    propagated_string = propagated_subset.__str__()
    expected_string = expected_subset.__str__()
    assert (propagated_string == expected_string)
    assert (not propagated_memlet.subset.subset_list)


def test_step_not_one():
    A = dace.data.Array(dace.int64, (N, M))
    subset = Range.from_string("i")
    i_subset = Range.from_string("0:N:3")
    memlet = dace.Memlet(None, "A", subset)
    memlets = [memlet]

    propagated_memlet = UnderapproximateWrites().propagate_subset(memlets, A, ["i"], i_subset, None, True)
    propagated_subset = propagated_memlet.subset.subset_list[0]

    expected_subset = Range.from_string("0:N:3")
    propagated_string = propagated_subset.__str__()
    expected_string = expected_subset.__str__()
    assert (propagated_string == expected_string)


def test_simple_loop_overwrite():
    "simple loop that overwrites a one-dimensional array"
    sdfg = dace.SDFG("simple_loop")
    sdfg.add_array("A", [N], dace.int64)
    init = sdfg.add_state("init")
    end = sdfg.add_state("end")
    loop_body = sdfg.add_state("loop_body")
    _, guard, _ = sdfg.add_loop(init, loop_body, end, "i", "0", "i < N", "i + 1")
    a0 = loop_body.add_access("A")
    loop_tasklet = loop_body.add_tasklet("overwrite", {}, {"a"}, "a = 0")
    loop_body.add_edge(loop_tasklet, "a", a0, None, dace.Memlet("A[i]"))

    result = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]["loop_approximation"]

    assert (result[guard]["A"].subset.__str__() == Range.from_array(sdfg.arrays["A"]).__str__())

def test_loop_2D_overwrite():
    """two nested loops that overwrite a two-dimensional array"""
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

    result = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]["loop_approximation"]

    assert (result[guard1]["A"].subset.__str__() == Range.from_array(sdfg.arrays["A"]).__str__())
    assert (result[guard2]["A"].subset.__str__() == "j, 0:N")


def test_loop_2D_no_overwrite_0():
    """three nested loops that overwrite two dimensional array.
    the innermost loop is surrounded by loop that doesn't iterate over array range.
    Therefore we don't want the full write in the approximation for the outermost loop,
    since the second innermost loop could just not execute"""
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

    result = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]["loop_approximation"]

    assert ("A" not in result[guard1].keys())
    assert ("A" not in result[guard2].keys())
    assert (result[guard3]["A"].subset.__str__() == "j, 0:N")

def test_2_loops_overwrite():
    """2 loops one after another overwriting an array"""
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

    result = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]["loop_approximation"]

    assert (result[guard_1]["A"].subset.__str__() == Range.from_array(sdfg.arrays["A"]).__str__())
    assert (result[guard_2]["A"].subset.__str__() == Range.from_array(sdfg.arrays["A"]).__str__())

def test_loop_2D_overwrite_propagation_gap():
    """three nested loops that overwrite two dimensional array.
    the innermost loop is surrounded by a loop that doesn't iterate over array range but over a non-empty constant range.
    Therefore the loop nest as a whole overwrites the array"""
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

    result = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]["loop_approximation"]

    assert (result[guard1]["A"].subset.__str__() == Range.from_array(sdfg.arrays["A"]).__str__())
    assert (result[guard2]["A"].subset.__str__() == "j, 0:N")
    assert (result[guard3]["A"].subset.__str__() == "j, 0:N")


def test_loop_2D_no_overwrite_1():
    """three nested loops that write to two dimensional array.
    the subscript expression is a multiplication of two indices -> return empty subset"""
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
    loop_body.add_edge(loop_tasklet, "a", a0, None, dace.Memlet("A[i,3*j]"))

    result = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]["loop_approximation"]

    assert (result[guard1]["A"].subset.__str__() == Range.from_string("0:N, 0:3*M - 2:3").__str__())
    assert (result[guard2]["A"].subset.__str__() == "0:N, 3*j")
    assert (result[guard3]["A"].subset.__str__() == "0:N, 3*j")

def test_loop_2D_no_overwrite_2():
    """three nested loops that write to two dimensional array.
    the innermost loop is surrounded by a loop that iterates over an empty range.
    Therefore the loop nest as a whole does not overwrite the array"""

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

    result = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]["loop_approximation"]

    assert (guard1 not in result.keys() or "A" not in result[guard1].keys())
    assert (guard2 not in result.keys() or "A" not in result[guard2].keys())
    assert (guard3 not in result.keys() or "A" not in result[guard3].keys())

def test_loop_2D_branch():
    """loop nested in another loop. nested loop is in a branch and overwrites the array.
        The propagation should return an empty subset for the outermost loop
        and the full subset for the loop in the branch"""
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

    result = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]["loop_approximation"]

    assert (guard1 not in result.keys() or "A" not in result[guard1].keys())
    assert (guard2 in result.keys() and "A" in result[guard2].keys())

def test_loop_nested_sdfg():
    @dace.program
    def nested_loop(A: dace.float64[M, N]):
        for i in dace.map[0:M]:
            for j in range(N):
                A[i + 1, j * i] = 1
    sdfg = nested_loop.to_sdfg()

    result = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]

    write_approx = result["approximation"]
    write_set = None
    accessnode = None
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.AccessNode):
            if node.data == "A":
                accessnode = node
    for edge, memlet in write_approx.items():
        if edge.dst is accessnode:
            write_set = memlet.subset
            candiate_memlet = memlet
    assert(len(write_set.subset_list) == 0)

def test_loop_in_nested_sdfg_simple():
    @dace.program
    def nested_loop(A: dace.float64[M, N]):
        for i in dace.map[0:M]:
            for j in range(N):
                A[i, j] = 1
    sdfg = nested_loop.to_sdfg()

    result = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]

    # find write set
    write_approx = result["approximation"]
    accessnode = None
    write_set = None
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.AccessNode):
            if node.data == "A":
                accessnode = node
    for edge, memlet in write_approx.items():
        if edge.dst is accessnode:
            write_set = memlet.subset

    assert (write_set.__str__() == "0:M, 0:N")


def test_loop_break():
    """loop that has a break statement. So the analysis should not propagate memlets outside of this loop"""
    sdfg = dace.SDFG("loop_2D_no_overwrite")
    sdfg.add_array("A", [N], dace.int64)
    init = sdfg.add_state("init", is_start_state=True)
    loop_body_0 = sdfg.add_state("loop_body_0")
    loop_body_1 = sdfg.add_state("loop_body_1")
    loop_after_1 = sdfg.add_state("loop_after_1")
    _, guard3, _ = sdfg.add_loop(init, loop_body_0, loop_after_1, "i", "0", "i < N", "i + 1", loop_body_1)
    sdfg.add_edge(loop_body_0, loop_after_1, dace.InterstateEdge(condition="i > 10"))
    sdfg.add_edge(loop_body_0, loop_body_1, dace.InterstateEdge(condition="not(i > 10)"))
    a0 = loop_body_1.add_access("A")
    loop_tasklet = loop_body_1.add_tasklet("overwrite", {}, {"a"}, "a = 0")
    loop_body_1.add_edge(loop_tasklet, "a", a0, None, dace.Memlet("A[i]"))

    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]

    result = results["loop_approximation"]
    assert (guard3 not in result.keys() or "A" not in result[guard3].keys())

if __name__ == '__main__':
    test_nested_sdfg_in_map_branches_no_write_1()
    test_nested_sdfg_in_map_branches_no_write()
    test_nested_sdfg_in_map_full_write_1()
    test_loop_in_nested_sdfg_in_map_partial_write()
    test_nested_sdfg_in_map_full_write()
    test_map_in_loop_no_write_0()
    test_map_in_loop_no_write_1()
    test_map_in_loop_full_write()
    test_loop_in_map_no_write()
    test_loop_in_map_full_write()
    test_map_tree_full_write()
    test_2D_map_full_write()
    test_2D_map_added_indices_no_propagation()
    test_2D_map_multiplied_indices_no_propagation()
    test_1D_map_one_index_multiple_dims_no_propagation()
    test_1D_map_one_index_squared_no_fission()
    test_map_tree_multiple_indices_per_dimension_full_write()
    test_map_tree_no_write_multiple_indices()
    test_step_not_one()
    test_one_variable_in_2dimensions()
    test_affine_2D()
    test_constant_multiplicative_2D()
    test_multiplied_variables()
    test_negative_step()
    test_loop_in_nested_sdfg_simple()
    test_loop_2D_branch()
    test_loop_2D_no_overwrite_2()
    test_simple_loop_overwrite()
    test_loop_2D_overwrite()
    test_loop_2D_overwrite_propagation_gap()
    test_2_loops_overwrite()
    test_loop_2D_no_overwrite_0()
    test_loop_2D_no_overwrite_1()
    test_loop_nested_sdfg()
    test_loop_break()
