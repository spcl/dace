# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests conversion of schedule trees to SDFGs.
"""
import dace
import numpy as np

from dace.sdfg.state import ConditionalBlock


def test_implicit_inline_and_constants():
    """
    Tests implicit inlining upon roundtrip conversion, as well as constants with conflicting names.
    """

    @dace
    def nester(A: dace.float64[20]):
        A[:] = 12

    @dace.program
    def tester(A: dace.float64[20, 20]):
        for i in dace.map[0:20]:
            nester(A[:, i])

    sdfg = tester.to_sdfg(simplify=False)

    # Inject constant into nested SDFG
    assert len(list(sdfg.all_sdfgs_recursive())) > 1
    sdfg.add_constant('cst', 13)  # Add an unused constant
    sdfg.cfg_list[-1].add_constant('cst', 1, dace.data.Scalar(dace.float64))
    tasklet = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet))
    tasklet.code.as_string = tasklet.code.as_string.replace('12', 'cst')

    # Perform a roundtrip conversion
    stree = sdfg.as_schedule_tree()
    new_sdfg = stree.as_sdfg()

    assert len(list(new_sdfg.all_sdfgs_recursive())) == 1
    assert new_sdfg.constants['cst_0'].dtype == np.float64

    # Test SDFG
    a = np.random.rand(20, 20)
    new_sdfg(A=a)  # Tests arg_names
    assert np.allclose(a, 1)


def test_name_propagation():
    name = "my_complicated_sdfg_test_name"
    sdfg = dace.SDFG(name)
    sdfg.add_state("empty", is_start_block=True)

    stree = sdfg.as_schedule_tree()
    assert stree.name == name

    sdfg = stree.as_sdfg()
    assert sdfg.name == name


def test_transients_and_nested_sdfg() -> None:

    def nestedSDFG() -> dace.SDFG:

        def get_start_state(sdfg: dace.SDFG) -> dace.SDFGState:
            state = sdfg.add_state("my_state", is_start_block=True)
            read = state.add_read("B")
            write = state.add_write("tmp_condition")
            tasklet = state.add_tasklet(
                "masklet",
                {"B0"},  # inputs
                {"out"},  # outputs
                "out = B0 < 0",
            )
            state.add_edge(read, None, tasklet, "B0", dace.Memlet("B[0]"))
            state.add_edge(tasklet, "out", write, None, dace.Memlet("tmp_condition[0]"))
            return state

        def get_if_block(sdfg: dace.SDFG) -> dace.sdfg.ControlFlowRegion:
            if_block = ConditionalBlock('if_region', sdfg=sdfg)
            then_body = dace.sdfg.ControlFlowRegion('then_body', sdfg=sdfg, parent=if_block)
            then_state = then_body.add_state('then_state', is_start_block=True)
            then_write = then_state.add_write('B')
            then_tasklet = then_state.add_tasklet('write_zero', {}, {'out'}, 'out = 0')
            then_state.add_edge(then_tasklet, 'out', then_write, None, dace.Memlet('B[0]'))
            if_block.add_branch("tmp_condition", then_body)
            return if_block

        def get_map_state(sdfg: dace.SDFG) -> dace.SDFGState:
            state = sdfg.add_state("map_state")
            access_A = state.add_access("A")
            write_B = state.add_write("B")
            state.add_mapped_tasklet("write_one", {"j": dace.subsets.Range.from_string("0:10")}, {},
                                     "out = 1.0", {"out": dace.Memlet("A[15*i + 3*j]")},
                                     external_edges=True,
                                     output_nodes={"A": access_A})
            state.add_mapped_tasklet("copy", {"k": dace.subsets.Range.from_string("10:20")},
                                     {"read": dace.Memlet("A[k]")},
                                     "write = read", {"write": dace.Memlet("B[k]")},
                                     external_edges=True,
                                     input_nodes={"A": access_A},
                                     output_nodes={"B": write_B})
            return state

        sdfg = dace.SDFG(name="nested")
        sdfg.add_scalar("tmp_condition", dace.bool, transient=True)
        sdfg.add_array("A", [60], dace.float32)
        sdfg.add_array("B", [60], dace.float32)

        start_state = get_start_state(sdfg)

        if_block = get_if_block(sdfg)
        sdfg.add_node(if_block)
        sdfg.add_edge(start_state, if_block, dace.InterstateEdge())

        map_state = get_map_state(sdfg)
        sdfg.add_edge(if_block, map_state, dace.InterstateEdge())

        return sdfg

    sdfg = dace.SDFG(name="tester")
    _, A_desc = sdfg.add_array("A", [60], dace.float32, transient=True)
    _, B_desc = sdfg.add_array("B", [60], dace.float32)
    state = sdfg.add_state("state")
    access_A = state.add_access("A")
    state.add_mapped_tasklet(
        "fill",
        {"i": dace.subsets.Range.from_string("0:60")},
        {},  # inputs
        "out = 42.42",
        {"out": dace.Memlet("A[i]")},  # outputs
        external_edges=True,
        output_nodes={"A": access_A})

    read_B = state.add_read("B")
    map_entry, map_exit = state.add_map("second_map", {"i": dace.subsets.Range.from_string("0:2")})

    # map_entry
    map_entry.add_in_connector("IN_A")
    map_entry.add_out_connector("OUT_A")
    map_entry.add_in_connector("IN_B")
    map_entry.add_out_connector("OUT_B")
    state.add_edge(access_A, None, map_entry, "IN_A", dace.Memlet.from_array("A", A_desc))
    state.add_edge(read_B, None, map_entry, "IN_B", dace.Memlet.from_array("B", B_desc))

    # nested SDFG
    nsdfg = nestedSDFG()
    nsdfg_node = state.add_nested_sdfg(
        nsdfg,
        {
            "A": None,
            "B": None
        },  # inputs
        {
            "A": None,
            "B": None
        },  # outputs
        name="nested_sdfg",
    )
    state.add_edge(map_entry, "OUT_A", nsdfg_node, "A", dace.Memlet.from_array("A", A_desc))
    state.add_edge(map_entry, "OUT_B", nsdfg_node, "B", dace.Memlet.from_array("B", B_desc))

    # map_exit
    map_exit.add_in_connector("IN_A")
    map_exit.add_out_connector("OUT_A")
    state.add_edge(nsdfg_node, "A", map_exit, "IN_A", dace.Memlet.from_array("A", A_desc))
    write_A = state.add_write("A")
    state.add_edge(map_exit, "OUT_A", write_A, None, dace.Memlet.from_array("A", A_desc))

    map_exit.add_in_connector("IN_B")
    map_exit.add_out_connector("OUT_B")
    state.add_edge(nsdfg_node, "B", map_exit, "IN_B", dace.Memlet.from_array("B", B_desc))
    write_B = state.add_write("B")
    state.add_edge(map_exit, "OUT_B", write_B, None, dace.Memlet.from_array("B", B_desc))

    sdfg.validate()
    dace.sdfg.propagation.propagate_memlets_sdfg(sdfg)
    stree = sdfg.as_schedule_tree()
    roundtrip_sdfg = stree.as_sdfg(validate=True)

    assert roundtrip_sdfg


if __name__ == '__main__':
    test_implicit_inline_and_constants()
    test_name_propagation()
    test_transients_and_nested_sdfg()
