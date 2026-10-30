# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Four defects ls3df_scf hit on the way to the device; each test fails without its fix."""

import dace
from dace import dtypes, nodes
from dace.libraries.linalg.environments import hiptensor
from dace.libraries.linalg.nodes.tensordot import TensorDot
from dace.libraries.linalg.nodes.ttranspose import TensorTranspose
from dace.sdfg.graph import SubgraphView
from dace.transformation import helpers as xfh
from dace.transformation.passes.scalar_to_symbol import find_promotable_scalars


def test_state_fission_drops_a_copy_between_two_boundary_nodes():
    """``offsets -> idx`` split off while both are read on: the copy ran in the first state, so the second state's
    copy of it goes (it asserted)."""
    sdfg = dace.SDFG("fission_boundary_copy")
    sdfg.add_array("offsets", [4], dace.int64)
    sdfg.add_array("out", [2], dace.int64)
    sdfg.add_scalar("idx", dace.int64, transient=True)
    state = sdfg.add_state()
    offsets, idx = state.add_read("offsets"), state.add_access("idx")
    state.add_edge(offsets, None, idx, None, dace.Memlet("offsets[1]"))
    use_idx = state.add_tasklet("use_idx", {"i"}, {"o"}, "o = i")
    use_offsets = state.add_tasklet("use_offsets", {"v"}, {"o"}, "o = v")
    state.add_edge(idx, None, use_idx, "i", dace.Memlet("idx[0]"))
    state.add_edge(offsets, None, use_offsets, "v", dace.Memlet("offsets[0]"))
    out = state.add_write("out")
    state.add_edge(use_idx, "o", out, None, dace.Memlet("out[0]"))
    state.add_edge(use_offsets, "o", out, None, dace.Memlet("out[1]"))
    sdfg.validate()

    xfh.state_fission(SubgraphView(state, {offsets, idx}))
    sdfg.validate()
    assert not any(isinstance(e.src, nodes.AccessNode) and isinstance(e.dst, nodes.AccessNode) for e in state.edges())


def test_a_scalar_copied_from_device_memory_is_not_promoted():
    """The promoted symbol is read on host interstate edges, which cannot read ``GPU_Global`` memory."""
    sdfg = dace.SDFG("promote_from_device")
    sdfg.add_array("offsets", [4], dace.int64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_scalar("idx", dace.int64, transient=True, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()
    state.add_edge(state.add_read("offsets"), None, state.add_write("idx"), None, dace.Memlet("offsets[1]"))
    assert "idx" not in find_promotable_scalars(sdfg)


def test_ttgt_on_device_operands_calls_the_vendor_gemm():
    """The GEMM TTGT creates is past implementation selection: on device operands it must not be ``pure``, whose
    host maps cannot read device memory."""
    sdfg = dace.SDFG("ttgt_device")
    gpu = dtypes.StorageType.GPU_Global
    sdfg.add_array("A", [4, 5], dace.float64, storage=gpu)
    sdfg.add_array("B", [5, 3], dace.float64, storage=gpu)
    sdfg.add_array("C", [4, 3], dace.float64, storage=gpu)
    state = sdfg.add_state()
    node = TensorDot("tdot", left_axes=[1], right_axes=[0])
    node.implementation = "TTGT"
    state.add_node(node)
    state.add_edge(state.add_read("A"), None, node, "_left_tensor", dace.Memlet("A[0:4, 0:5]"))
    state.add_edge(state.add_read("B"), None, node, "_right_tensor", dace.Memlet("B[0:5, 0:3]"))
    state.add_edge(node, "_out_tensor", state.add_write("C"), None, dace.Memlet("C[0:4, 0:3]"))
    with dace.config.set_temporary("compiler", "cuda", "backend", value="hip"):
        node.expand(state)
    gemms = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.LibraryNode) and n.label == "_GEMM_"]
    assert gemms and all(g.implementation == "hipBLAS" for g in gemms)


def test_ttgt_on_device_operands_transposes_with_a_device_library():
    """The transposes TTGT creates for inner contraction axes and an output permutation are past implementation
    selection too: on device operands they take the device implementation, not ``pure``."""
    sdfg = dace.SDFG("ttgt_device_transposes")
    gpu = dtypes.StorageType.GPU_Global
    sdfg.add_array("A", [4, 3, 5], dace.float64, storage=gpu)
    sdfg.add_array("B", [4, 6, 5], dace.float64, storage=gpu)
    sdfg.add_array("C", [6, 3], dace.float64, storage=gpu)
    state = sdfg.add_state()
    node = TensorDot("tdot", left_axes=[0, 2], right_axes=[0, 2], permutation=[1, 0])
    node.implementation = "TTGT"
    state.add_node(node)
    state.add_edge(state.add_read("A"), None, node, "_left_tensor", dace.Memlet("A[0:4, 0:3, 0:5]"))
    state.add_edge(state.add_read("B"), None, node, "_right_tensor", dace.Memlet("B[0:4, 0:6, 0:5]"))
    state.add_edge(node, "_out_tensor", state.add_write("C"), None, dace.Memlet("C[0:6, 0:3]"))
    with dace.config.set_temporary("compiler", "cuda", "backend", value="hip"):
        node.expand(state)
        tensor = "hipTENSOR" if hiptensor.hipTensor.is_installed() else "CUDA"
    transposes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TensorTranspose)]
    assert len(transposes) == 3 and all(t.implementation == tensor for t in transposes)


def test_a_container_named_in_tasklet_code_inside_the_subgraph_is_no_argument():
    """A tasklet that names its container in code (``InlineTaskletConnectors``) makes it a used symbol; when the
    container has its access node inside the subgraph, its allocation is the subgraph's, not the caller's."""
    sdfg = dace.SDFG("arglist_named_container")
    sdfg.add_array("out", [8], dace.float64)
    sdfg.add_transient("t", [1], dace.float64)
    state = sdfg.add_state()
    entry, exit_ = state.add_map("m", dict(i="0:8"))
    write_t = state.add_tasklet("write_t", {}, {"o"}, "o = 1.0")
    t = state.add_access("t")
    read_t = state.add_tasklet("read_t", {"x"}, {"o"}, "o = x + t[0]")
    state.add_nedge(entry, write_t, dace.Memlet())
    state.add_edge(write_t, "o", t, None, dace.Memlet("t[0]"))
    state.add_edge(t, None, read_t, "x", dace.Memlet("t[0]"))
    state.add_memlet_path(read_t, exit_, state.add_write("out"), src_conn="o", memlet=dace.Memlet("out[i]"))
    assert "t" in {str(s) for s in state.scope_subgraph(entry).used_symbols(all_symbols=False)}
    assert "t" not in state.scope_subgraph(entry).arglist()


def test_a_gpu_dot_over_a_symbolic_power_length_is_cpp():
    """The vendor dot's length went into C++ as Python text (``Lb**3``)."""
    from dace.libraries.blas.nodes.dot import Dot

    lb = dace.symbol("Lb")
    sdfg = dace.SDFG("gpu_dot_power_length")
    gpu = dtypes.StorageType.GPU_Global
    sdfg.add_array("x", [lb**3], dace.float64, storage=gpu)
    sdfg.add_array("y", [lb**3], dace.float64, storage=gpu)
    sdfg.add_array("r", [1], dace.float64, storage=gpu)
    state = sdfg.add_state()
    node = Dot("dot")
    node.implementation = "hipBLAS"
    state.add_node(node)
    state.add_edge(state.add_read("x"), None, node, "_x", dace.Memlet.from_array("x", sdfg.arrays["x"]))
    state.add_edge(state.add_read("y"), None, node, "_y", dace.Memlet.from_array("y", sdfg.arrays["y"]))
    state.add_edge(node, "_result", state.add_write("r"), None, dace.Memlet("r[0]"))
    with dace.config.set_temporary("compiler", "cuda", "backend", value="hip"):
        node.expand(state)
    code = " ".join(n.code.as_string for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.Tasklet))
    assert "**" not in code and "Ddot" in code, code


if __name__ == "__main__":
    test_state_fission_drops_a_copy_between_two_boundary_nodes()
    test_a_scalar_copied_from_device_memory_is_not_promoted()
    test_ttgt_on_device_operands_calls_the_vendor_gemm()
    test_ttgt_on_device_operands_transposes_with_a_device_library()
    test_a_container_named_in_tasklet_code_inside_the_subgraph_is_no_argument()
    test_a_gpu_dot_over_a_symbolic_power_length_is_cpp()
