# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Detection of the staged buffers of a loop, and multiple buffering through the frontend and code generation."""

import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.multiple_buffering import find_staged_buffers
from tests.passes.multiple_buffering.lower_multiple_buffering_test import TILE, expected_add, tiled_add, tiled_gemm


def _loop(sdfg: dace.SDFG) -> LoopRegion:
    loops = [region for region in sdfg.all_control_flow_regions(recursive=True) if isinstance(region, LoopRegion)]
    assert len(loops) == 1
    return loops[0]


def test_finds_both_tiles():
    sdfg = tiled_gemm("staged_gemm", 2)
    staged, rejected = find_staged_buffers(_loop(sdfg), sdfg)
    assert sorted(buffer.name for buffer in staged) == ["a_tile", "b_tile"]
    assert {copy.src.data for buffer in staged for copy in buffer.copies} == {"A", "B"}
    # The output is written by the computation, not by a copy
    assert "C" in rejected


def test_names_restrict_candidates():
    sdfg = tiled_gemm("staged_gemm_names", 2)
    staged, rejected = find_staged_buffers(_loop(sdfg), sdfg, {"b_tile", "C"})
    assert [buffer.name for buffer in staged] == ["b_tile"]
    assert set(rejected) == {"C"}


def test_copy_using_body_symbol_is_rejected():
    """A copy subset that uses a symbol assigned in the body cannot be moved ahead of that assignment."""
    sdfg = tiled_add("staged_body_symbol", 2)
    loop = _loop(sdfg)
    (copy_edge,) = [e for e in loop.start_block.edges() if e.data.data == "A"]
    copy_edge.data.subset = dace.subsets.Range.from_string(f"u:u+{TILE}")
    first = loop.add_state_before(loop.start_block, "assign", is_start_block=True)
    loop.edges_between(first, loop.out_edges(first)[0].dst)[0].data.assignments["u"] = "t"
    staged, rejected = find_staged_buffers(loop, sdfg)
    assert not staged
    assert "assigns" in rejected["buf"]


def test_codegen_lowers_pipeline_depth():
    """Compiling an SDFG with a pipelined loop lowers it without calling the pass."""
    sdfg = tiled_add("staged_codegen", 3, 28)
    code = sdfg.generate_code()[0].clean_code
    assert "% 3" in code
    A = np.random.rand(28)
    B = np.zeros(28)
    sdfg(A=A, B=B)
    assert np.allclose(B, expected_add(A, 0, TILE))


def test_python_frontend():
    N = dace.symbol("N")

    @dace.program
    def tiled(A: dace.float64[N], B: dace.float64[N]):
        # A staging buffer in another storage than its source, which simplification keeps
        buf = dace.define_local([TILE], dace.float64, storage=dace.StorageType.Register)
        for t in range(0, N, TILE):
            buf[:] = A[t : t + TILE]
            B[t : t + TILE] = 2 * buf + 1

    sdfg = tiled.to_sdfg(simplify=True)
    loop = _loop(sdfg)
    staged, rejected = find_staged_buffers(loop, loop.sdfg)
    assert [buffer.name for buffer in staged] == ["buf"], rejected
    loop.pipeline_depth = 3
    assert "% 3" in sdfg.generate_code()[0].clean_code
    A = np.random.rand(24)
    B = np.zeros(24)
    sdfg(A=A, B=B, N=24)
    assert np.allclose(B, 2 * A + 1)


def _tiled_add_loop(name: str, **kwargs) -> tuple[dace.SDFG, LoopRegion]:
    sdfg = tiled_add(name, 2, **kwargs)
    return sdfg, _loop(sdfg)


def test_copy_node_is_part_of_copy():
    sdfg, loop = _tiled_add_loop("staged_copy_node", copy_node=True)
    staged, rejected = find_staged_buffers(loop, sdfg)
    assert [buffer.name for buffer in staged] == ["buf"], rejected
    (staged_copy,) = staged[0].copies
    assert staged_copy.copy_node is not None
    assert len(staged_copy.edges) == 2
    assert staged_copy.src.data == "A" and staged_copy.dst.data == "buf"


def test_plain_copy_has_no_copy_node():
    sdfg, loop = _tiled_add_loop("staged_plain_copy")
    staged, _ = find_staged_buffers(loop, sdfg)
    (staged_copy,) = staged[0].copies
    assert staged_copy.copy_node is None
    assert len(staged_copy.edges) == 1


def test_later_state_use_is_staged():
    sdfg, loop = _tiled_add_loop("staged_two_states", two_states=True)
    staged, rejected = find_staged_buffers(loop, sdfg)
    assert [buffer.name for buffer in staged] == ["buf"], rejected


def test_loop_without_writes_has_nothing_staged():
    sdfg = dace.SDFG("staged_empty")
    sdfg.add_array("A", [8], dace.float64)
    loop = LoopRegion("loop", "i < 8", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop, is_start_block=True)
    loop.add_state("body", is_start_block=True)
    assert find_staged_buffers(loop, sdfg) == ([], {})


def test_unknown_name_is_ignored():
    sdfg, loop = _tiled_add_loop("staged_unknown_name")
    assert find_staged_buffers(loop, sdfg, {"does_not_exist"}) == ([], {})


def test_non_transient_destination_is_rejected():
    sdfg, loop = _tiled_add_loop("staged_non_transient")
    sdfg.arrays["buf"].transient = False
    staged, rejected = find_staged_buffers(loop, sdfg)
    assert not staged
    assert rejected["buf"] == "not a transient array"


def test_scalar_destination_is_rejected():
    sdfg = dace.SDFG("staged_scalar")
    sdfg.add_array("A", [8], dace.float64)
    sdfg.add_array("B", [8], dace.float64)
    sdfg.add_scalar("s", dace.float64, transient=True)
    loop = LoopRegion("loop", "i < 8", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop, is_start_block=True)
    state = loop.add_state("body", is_start_block=True)
    s = state.add_access("s")
    state.add_nedge(state.add_read("A"), s, dace.Memlet("A[i]"))
    state.add_nedge(s, state.add_write("B"), dace.Memlet("B[i]"))
    staged, rejected = find_staged_buffers(loop, sdfg)
    assert not staged
    assert rejected["s"] == "not a transient array"


def test_buffer_used_outside_loop_is_rejected():
    sdfg, loop = _tiled_add_loop("staged_used_outside")
    after = sdfg.add_state_after(loop, "after")
    after.add_nedge(after.add_read("buf"), after.add_write("B"), dace.Memlet(f"buf[0:{TILE}]", other_subset="0:4"))
    staged, rejected = find_staged_buffers(loop, sdfg)
    assert not staged
    assert rejected["buf"] == "accessed outside the loop"


def test_buffer_read_on_interstate_edge_is_rejected():
    sdfg, loop = _tiled_add_loop("staged_interstate", two_states=True)
    (edge,) = loop.out_edges(loop.start_block)
    edge.data.condition = dace.properties.CodeBlock("buf > 0")
    staged, rejected = find_staged_buffers(loop, sdfg)
    assert not staged
    assert rejected["buf"] == "read on an interstate edge"


def test_source_written_in_loop_is_rejected():
    sdfg, loop = _tiled_add_loop("staged_src_written")
    state = loop.add_state_after(loop.start_block, "write_a")
    tasklet = state.add_tasklet("w", {}, {"o"}, "o = 0")
    state.add_edge(tasklet, "o", state.add_write("A"), None, dace.Memlet("A[t]"))
    staged, rejected = find_staged_buffers(loop, sdfg)
    assert not staged
    assert "A is written in the loop" in rejected["buf"]


def test_view_source_is_rejected():
    sdfg, loop = _tiled_add_loop("staged_view_src")
    sdfg.add_view("A_view", [16], dace.float64)
    for node in loop.start_block.data_nodes():
        if node.data == "A":
            node.data = "A_view"
            for edge in loop.start_block.out_edges(node):
                edge.data.data = "A_view"
    staged, rejected = find_staged_buffers(loop, sdfg)
    assert not staged
    assert "view or reference" in rejected["buf"]


def test_write_by_tasklet_is_rejected():
    """A buffer a computation fills is not a staged copy."""
    sdfg = dace.SDFG("staged_tasklet_write")
    sdfg.add_array("B", [8], dace.float64)
    sdfg.add_transient("tmp", [1], dace.float64)
    loop = LoopRegion("loop", "i < 8", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop, is_start_block=True)
    state = loop.add_state("body", is_start_block=True)
    tmp = state.add_access("tmp")
    tasklet = state.add_tasklet("w", {}, {"o"}, "o = i")
    state.add_edge(tasklet, "o", tmp, None, dace.Memlet("tmp[0]"))
    state.add_nedge(tmp, state.add_write("B"), dace.Memlet("B[i]", other_subset="0"))
    staged, rejected = find_staged_buffers(loop, sdfg)
    assert not staged
    assert "not a copy" in rejected["tmp"]


def test_copy_inside_map_is_rejected():
    sdfg, loop = _tiled_add_loop("staged_scoped_copy")
    state = loop.start_block
    for edge in state.edges():
        if edge.data.data == "A":
            state.remove_node(edge.src)
            break
    buf = [node for node in state.data_nodes() if node.data == "buf" and state.in_degree(node) == 0][0]
    entry, exit_ = state.add_map("copy", {"j": f"0:{TILE}"})
    state.add_memlet_path(state.add_read("A"), entry, exit_, buf, memlet=dace.Memlet("A[t + j]"))
    staged, rejected = find_staged_buffers(loop, sdfg)
    assert not staged
    assert rejected["buf"]


def test_copies_in_two_states_are_rejected():
    sdfg, loop = _tiled_add_loop("staged_two_copy_states", two_states=True)
    second = [s for s in loop.states() if s.label == "compute"][0]
    second.add_nedge(second.add_read("A"), second.add_access("buf"), dace.Memlet(f"A[t:t+{TILE}]", other_subset="0:4"))
    staged, rejected = find_staged_buffers(loop, sdfg)
    assert not staged
    assert rejected["buf"] == "copied in more than one state"


def test_copy_in_nested_region_is_rejected():
    """A copy must be in a state that is a direct child of the loop, not one in a nested region."""
    sdfg, loop = _tiled_add_loop("staged_nested_region_copy")
    state = loop.start_block
    region = dace.sdfg.state.ControlFlowRegion("inner", sdfg)
    loop.remove_node(state)
    loop.add_node(region, is_start_block=True)
    region.add_node(state, is_start_block=True)
    staged, rejected = find_staged_buffers(loop, sdfg)
    assert not staged
    assert "not a direct child" in rejected["buf"]


def test_read_before_copy_in_earlier_state_is_rejected():
    sdfg, loop = _tiled_add_loop("staged_read_first")
    sdfg.add_array("C", [1], dace.float64)
    first = loop.add_state_before(loop.start_block, "first", is_start_block=True)
    first.add_nedge(first.add_read("buf"), first.add_write("C"), dace.Memlet("buf[0]", other_subset="0"))
    staged, rejected = find_staged_buffers(loop, sdfg)
    assert not staged
    assert "does not follow the copy" in rejected["buf"]


def test_read_before_copy_in_same_state_is_rejected():
    sdfg, loop = _tiled_add_loop("staged_read_first_same_state")
    sdfg.add_array("C", [1], dace.float64)
    state = loop.start_block
    # An independent read of the buffer's previous contents, with no path from the copy
    state.add_nedge(state.add_read("buf"), state.add_write("C"), dace.Memlet("buf[0]", other_subset="0"))
    staged, rejected = find_staged_buffers(loop, sdfg)
    assert not staged
    assert "before it is copied" in rejected["buf"]


def test_buffer_missing_the_copy_on_one_path_is_rejected():
    """A buffer read in a state the copy does not dominate (the copy sits in a branch) is not staged."""
    sdfg, loop = _tiled_add_loop("staged_undominated", two_states=True)
    load = loop.start_block
    compute = [s for s in loop.states() if s.label == "compute"][0]
    entry = loop.add_state_before(load, "entry", is_start_block=True)
    loop.edges_between(entry, load)[0].data.condition = dace.properties.CodeBlock("t > 0")
    loop.add_edge(entry, compute, dace.InterstateEdge("t <= 0"))
    staged, rejected = find_staged_buffers(loop, sdfg)
    assert not staged
    assert "does not follow the copy" in rejected["buf"]


def test_loop_variable_in_copy_is_allowed():
    """Only symbols the body assigns block staging, not the loop variable itself."""
    sdfg, loop = _tiled_add_loop("staged_loop_var")
    staged, _ = find_staged_buffers(loop, sdfg)
    assert any("t" in {str(sym) for sym in c.edges[0].data.free_symbols} for c in staged[0].copies)


def test_inner_loop_variable_blocks_copy():
    """A copy subset that uses the variable of a loop nested in the body is rejected like an assigned symbol."""
    sdfg, loop = _tiled_add_loop("staged_inner_loop_var")
    inner = LoopRegion("inner", "u < 2", "u", "u = 0", "u = u + 1")
    loop.add_node(inner)
    inner.add_state("noop", is_start_block=True)
    loop.add_edge(loop.start_block, inner, dace.InterstateEdge())
    (copy_edge,) = [e for e in loop.start_block.edges() if e.data.data == "A"]
    copy_edge.data.subset = dace.subsets.Range.from_string(f"u:u+{TILE}")
    staged, rejected = find_staged_buffers(loop, sdfg)
    assert not staged
    assert "assigns" in rejected["buf"]


def test_two_buffers_one_rejected():
    """Rejecting one candidate leaves the other staged."""
    sdfg = tiled_gemm("staged_gemm_partial", 2)
    sdfg.arrays["b_tile"].transient = False
    staged, rejected = find_staged_buffers(_loop(sdfg), sdfg)
    assert [buffer.name for buffer in staged] == ["a_tile"]
    assert rejected["b_tile"] == "not a transient array"


def test_large_sizes():
    """The analysis does not depend on the sizes, which are only symbols or numbers in the copy subsets."""
    sdfg, loop = _tiled_add_loop("staged_large", n=2**30)
    staged, _ = find_staged_buffers(loop, sdfg)
    assert [buffer.name for buffer in staged] == ["buf"]


if __name__ == "__main__":
    test_finds_both_tiles()
    test_names_restrict_candidates()
    test_copy_using_body_symbol_is_rejected()
    test_codegen_lowers_pipeline_depth()
    test_python_frontend()
    test_copy_node_is_part_of_copy()
    test_plain_copy_has_no_copy_node()
    test_later_state_use_is_staged()
    test_loop_without_writes_has_nothing_staged()
    test_unknown_name_is_ignored()
    test_non_transient_destination_is_rejected()
    test_scalar_destination_is_rejected()
    test_buffer_used_outside_loop_is_rejected()
    test_buffer_read_on_interstate_edge_is_rejected()
    test_source_written_in_loop_is_rejected()
    test_view_source_is_rejected()
    test_write_by_tasklet_is_rejected()
    test_copy_inside_map_is_rejected()
    test_copies_in_two_states_are_rejected()
    test_copy_in_nested_region_is_rejected()
    test_read_before_copy_in_earlier_state_is_rejected()
    test_read_before_copy_in_same_state_is_rejected()
    test_buffer_missing_the_copy_on_one_path_is_rejected()
    test_loop_variable_in_copy_is_allowed()
    test_inner_loop_variable_blocks_copy()
    test_two_buffers_one_rejected()
    test_large_sizes()
