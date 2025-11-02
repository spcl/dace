# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import numpy
import pytest
from dace.libraries.standard.nodes.copy_node import CopyLibraryNode
from dace.libraries.standard.nodes.memset_node import MemsetLibraryNode
from dace.transformation.passes.assignment_and_copy_kernel_to_memset_and_memcpy import AssignmentAndCopyKernelToMemsetAndMemcpy

# Global dimension size for all test arrays
DIM_SIZE = 10


def _get_sdfg(
    num_memcpies: int,
    num_memsets: int,
    extra_computation: bool,
    non_zero: bool,
    subset_in_first_dim: bool,
) -> dace.SDFG:
    """
    Construct an SDFG that performs a configurable number of memcpy and memset
    operations, possibly with extra computation or non-zero memsets.
    """

    sdfg = dace.SDFG("main")
    state = sdfg.add_state("memset_memcpy_maps")

    # Define the iteration space of the map (controls which indices are touched)
    map_entry, map_exit = state.add_map(
        name="memcpy_memset_map",
        ndrange={
            "i":
            dace.subsets.Range([(0, DIM_SIZE - 1,
                                 1)]) if not subset_in_first_dim else dace.subsets.Range([(2, DIM_SIZE - 1, 1)]),
            "j":
            dace.subsets.Range([(0, DIM_SIZE - 1, 1)]),
        },
    )

    # Select memset value: 0.0 or 1.0 depending on `non_zero`
    assign_value = "0" if not non_zero else "1"

    # Create each memcpy or memset node
    for i in range(num_memcpies + num_memsets):
        is_memcpy = i < num_memcpies
        ch = chr(ord("A") + i)  # Name arrays alphabetically: A, B, C, ...

        in_name, out_name = f"{ch}_IN", f"{ch}_OUT"

        # Add 2D arrays for input and output
        for name in (in_name, out_name):
            sdfg.add_array(
                name=name,
                shape=(DIM_SIZE, DIM_SIZE),
                dtype=dace.float64,
                transient=False,
            )

        # Build the tasklet: memcpy = pass-through, memset = constant assignment
        tasklet_name = f"{'memcpy' if is_memcpy else 'memset'}_{i}"
        tasklet_code = "_out = _in" if is_memcpy else f"_out = {assign_value}"

        tasklet = state.add_tasklet(
            name=tasklet_name,
            inputs={"_in"} if is_memcpy else set(),
            outputs={"_out"},
            code=tasklet_code,
        )
        tasklet.add_out_connector("_out")

        # Handle input connection for memcpy
        if is_memcpy:
            # Connect array → map → tasklet
            state.add_edge(
                state.add_access(in_name),
                None,
                map_entry,
                f"IN_{in_name}",
                dace.memlet.Memlet(f"{in_name}[2:{DIM_SIZE}, 0:{DIM_SIZE}]"
                                   if subset_in_first_dim else f"{in_name}[0:{DIM_SIZE}, 0:{DIM_SIZE}]"),
            )
            map_entry.add_in_connector(f"IN_{in_name}")
            map_entry.add_out_connector(f"OUT_{in_name}")
            tasklet.add_in_connector("_in")
            state.add_edge(
                map_entry,
                f"OUT_{in_name}",
                tasklet,
                "_in",
                dace.memlet.Memlet(f"{in_name}[i, j]"),
            )
        else:
            # Memset has no input, only output dependency
            state.add_edge(
                map_entry,
                None,
                tasklet,
                None,
                dace.memlet.Memlet(None),
            )

        # If enabled, add extra computation: double every other result
        if extra_computation and i % 2 == 0:
            sdfg.add_scalar(
                f"tmp_{i}",
                dace.float64,
                storage=dace.dtypes.StorageType.Register,
                transient=True,
            )
            tmp_access = state.add_access(f"tmp_{i}")

            # Store tasklet result in temporary
            state.add_edge(tasklet, "_out", tmp_access, None, dace.memlet.Memlet(f"tmp_{i}[0]"))

            # Add extra tasklet that doubles the value
            extra_tasklet = state.add_tasklet(
                name=f"{tasklet_name}_extra_work",
                inputs={"_in"},
                outputs={"_out"},
                code="_out = 2 * _in",
            )
            extra_tasklet.add_in_connector("_in")
            extra_tasklet.add_out_connector("_out")

            state.add_edge(
                tmp_access,
                None,
                extra_tasklet,
                "_in",
                dace.memlet.Memlet(f"tmp_{i}[0]"),
            )
            state.add_edge(
                extra_tasklet,
                "_out",
                map_exit,
                f"IN_{out_name}",
                dace.memlet.Memlet(f"{out_name}[i, j]"),
            )
        else:
            # Normal write path: tasklet → map_exit
            state.add_edge(
                tasklet,
                "_out",
                map_exit,
                f"IN_{out_name}",
                dace.memlet.Memlet(f"{out_name}[i, j]"),
            )

        # Final output: map_exit → output array
        state.add_edge(
            map_exit,
            f"OUT_{out_name}",
            state.add_access(out_name),
            None,
            dace.memlet.Memlet(f"{out_name}[2:{DIM_SIZE}, 0:{DIM_SIZE}]"
                               if subset_in_first_dim else f"{out_name}[0:{DIM_SIZE}, 0:{DIM_SIZE}]"),
        )
        map_exit.add_in_connector(f"IN_{out_name}")
        map_exit.add_out_connector(f"OUT_{out_name}")

    # Save for debugging and validate SDFG correctness
    sdfg.validate()
    return sdfg


# --- Utility functions for counting nodes in an SDFG ---
def _get_num_memcpy_library_nodes(sdfg: dace.SDFG) -> int:
    """Return number of memcpy library nodes in an SDFG."""
    return sum(isinstance(node, CopyLibraryNode) for node, state in sdfg.all_nodes_recursive())


def _get_num_memset_library_nodes(sdfg: dace.SDFG) -> int:
    """Return number of memset library nodes in an SDFG."""
    return sum(isinstance(node, MemsetLibraryNode) for node, state in sdfg.all_nodes_recursive())


D = dace.symbol("D")


@dace.program
def double_memset_with_dynamic_connectors(kfdia: dace.int32, kidia: dace.int32, llindex3: dace.float64[D, D],
                                          zsinksum: dace.float64[D]):
    for i, j in dace.map[0:D:1, kidia - 1:kfdia:]:
        llindex3[i, j] = 0.0
    for j in dace.map[kidia - 1:kfdia:1]:
        zsinksum[j] = 0.0


def _set_lib_node_type(sdfg: dace.SDFG, expansion_type: str):
    for n, g in sdfg.all_nodes_recursive():
        if isinstance(n, (CopyLibraryNode, MemsetLibraryNode)):
            n.implementation = expansion_type


@dace.program
def nested_maps(kidia: dace.int64, kfdia: dace.int64, llindex: dace.float64[5, 5, D], zsinksum: dace.float64[5, D]):
    for i in dace.map[0:5]:
        sym_kidia = kidia
        sym_kfdia = kfdia
        for j, k in dace.map[0:5, sym_kidia:sym_kfdia:1]:
            llindex[i, j, k] = 0.0
        for k in dace.map[sym_kidia:sym_kfdia:1]:
            zsinksum[i, k] = 0.0


EXPANSION_TYPES = ["pure", "CPU", pytest.param("CUDA", marks=pytest.mark.gpu)]


@pytest.mark.parametrize("expansion_type", EXPANSION_TYPES)
def test_nested_maps(expansion_type: str):
    if expansion_type == "CUDA":
        import cupy
    xp = cupy if expansion_type == "CUDA" else numpy

    sdfg = nested_maps.to_sdfg()

    AssignmentAndCopyKernelToMemsetAndMemcpy().apply_pass(sdfg, {})
    # We should have 2 memset libnodes
    assert _get_num_memset_library_nodes(
        sdfg) == 2, f"Expected 2 memset library nodes, got {_get_num_memset_library_nodes(sdfg)}"

    kidia = 0
    kfdia = DIM_SIZE
    A_IN = xp.random.rand(5, 5, DIM_SIZE)
    B_IN = xp.random.rand(5, DIM_SIZE)
    _set_lib_node_type(sdfg, expansion_type)
    sdfg.expand_library_nodes(recursive=True)
    sdfg.validate()
    sdfg(llindex=A_IN, zsinksum=B_IN, kidia=kidia, kfdia=kfdia, D=DIM_SIZE)
    assert xp.allclose(A_IN, 0.0)
    assert xp.allclose(B_IN, 0.0)


@pytest.mark.parametrize("expansion_type", EXPANSION_TYPES)
def test_double_memset_with_dynamic_connectors(expansion_type: str):
    if expansion_type == "CUDA":
        import cupy
    xp = cupy if expansion_type == "CUDA" else numpy

    sdfg = double_memset_with_dynamic_connectors.to_sdfg()

    A_IN = xp.random.rand(DIM_SIZE, DIM_SIZE)
    B_IN = xp.ones(DIM_SIZE)

    sdfg.validate()
    p = AssignmentAndCopyKernelToMemsetAndMemcpy()
    p.overapproximate_first_dimension = True
    p.apply_pass(sdfg, {})
    for n, g in sdfg.all_nodes_recursive():
        if isinstance(n, dace.nodes.NestedSDFG):
            p.apply_pass(n.sdfg)
    sdfg.validate()

    assert _get_num_memcpy_library_nodes(
        sdfg) == 0, f"Expected 0 memcpy library node, got {_get_num_memcpy_library_nodes(sdfg)}"
    assert _get_num_memset_library_nodes(
        sdfg) == 2, f"Expected 2 memset library nodes, got {_get_num_memset_library_nodes(sdfg)}"

    sdfg.expand_library_nodes(recursive=True)
    sdfg.validate()
    _set_lib_node_type(sdfg, expansion_type)
    sdfg.expand_library_nodes(recursive=True)
    sdfg.validate()
    sdfg(llindex3=A_IN, zsinksum=B_IN, D=DIM_SIZE, kfdia=1, kidia=DIM_SIZE)

    assert xp.all(B_IN == 0.0), f"zsinksum should be fully zeroed {B_IN}"
    assert xp.all(A_IN == 0.0), f"llindex3 should be fully zeroed {A_IN}"


@pytest.mark.parametrize("expansion_type", EXPANSION_TYPES)
def test_simple_memcpy(expansion_type: str):
    if expansion_type == "CUDA":
        import cupy
    xp = cupy if expansion_type == "CUDA" else numpy

    sdfg = _get_sdfg(1, 0, False, False, False)
    sdfg.validate()
    AssignmentAndCopyKernelToMemsetAndMemcpy().apply_pass(sdfg, {})
    sdfg.validate()
    assert _get_num_memcpy_library_nodes(sdfg) == 1
    assert _get_num_memset_library_nodes(sdfg) == 0

    A_IN = xp.random.rand(DIM_SIZE, DIM_SIZE)
    A_OUT = xp.zeros_like(A_IN)
    _set_lib_node_type(sdfg, expansion_type)
    sdfg.expand_library_nodes(recursive=True)
    sdfg.validate()
    sdfg(A_IN=A_IN, A_OUT=A_OUT)

    assert xp.allclose(A_IN, A_OUT)


@pytest.mark.parametrize("expansion_type", EXPANSION_TYPES)
def test_simple_memset(expansion_type: str):
    if expansion_type == "CUDA":
        import cupy
    xp = cupy if expansion_type == "CUDA" else numpy

    sdfg = _get_sdfg(0, 1, False, False, False)
    AssignmentAndCopyKernelToMemsetAndMemcpy().apply_pass(sdfg, {})
    assert _get_num_memcpy_library_nodes(sdfg) == 0
    assert _get_num_memset_library_nodes(sdfg) == 1

    A_IN = xp.random.rand(DIM_SIZE, DIM_SIZE)
    A_OUT = xp.zeros_like(A_IN)
    _set_lib_node_type(sdfg, expansion_type)
    sdfg.expand_library_nodes(recursive=True)
    sdfg.validate()
    sdfg(A_IN=A_IN, A_OUT=A_OUT)

    assert xp.allclose(A_OUT, 0.0)


@pytest.mark.parametrize("expansion_type", EXPANSION_TYPES)
def test_multi_memcpy(expansion_type: str):
    if expansion_type == "CUDA":
        import cupy
    xp = cupy if expansion_type == "CUDA" else numpy

    sdfg = _get_sdfg(2, 0, False, False, False)
    AssignmentAndCopyKernelToMemsetAndMemcpy().apply_pass(sdfg, {})
    assert _get_num_memcpy_library_nodes(sdfg) == 2
    assert _get_num_memset_library_nodes(sdfg) == 0

    A_IN = xp.random.rand(DIM_SIZE, DIM_SIZE)
    A_OUT = xp.zeros_like(A_IN)
    B_IN = xp.random.rand(DIM_SIZE, DIM_SIZE)
    B_OUT = xp.zeros_like(B_IN)
    _set_lib_node_type(sdfg, expansion_type)
    sdfg.expand_library_nodes(recursive=True)
    sdfg.validate()
    sdfg(A_IN=A_IN, A_OUT=A_OUT, B_IN=B_IN, B_OUT=B_OUT)

    assert xp.allclose(A_IN, A_OUT)
    assert xp.allclose(B_IN, B_OUT)


@pytest.mark.parametrize("expansion_type", EXPANSION_TYPES)
def test_multi_memset(expansion_type: str):
    if expansion_type == "CUDA":
        import cupy
    xp = cupy if expansion_type == "CUDA" else numpy

    sdfg = _get_sdfg(0, 2, False, False, False)
    AssignmentAndCopyKernelToMemsetAndMemcpy().apply_pass(sdfg, {})
    assert _get_num_memcpy_library_nodes(sdfg) == 0
    assert _get_num_memset_library_nodes(sdfg) == 2

    A_IN = xp.random.rand(DIM_SIZE, DIM_SIZE)
    A_OUT = xp.zeros_like(A_IN)
    B_IN = xp.random.rand(DIM_SIZE, DIM_SIZE)
    B_OUT = xp.zeros_like(B_IN)
    _set_lib_node_type(sdfg, expansion_type)
    sdfg.expand_library_nodes(recursive=True)
    sdfg.validate()
    sdfg(A_IN=A_IN, A_OUT=A_OUT, B_IN=B_IN, B_OUT=B_OUT)

    assert xp.allclose(A_OUT, 0.0)
    assert xp.allclose(B_OUT, 0.0)


@pytest.mark.parametrize("expansion_type", EXPANSION_TYPES)
def test_multi_mixed(expansion_type: str):
    if expansion_type == "CUDA":
        import cupy
    xp = cupy if expansion_type == "CUDA" else numpy

    sdfg = _get_sdfg(1, 1, False, False, False)
    AssignmentAndCopyKernelToMemsetAndMemcpy().apply_pass(sdfg, {})
    assert _get_num_memcpy_library_nodes(sdfg) == 1
    assert _get_num_memset_library_nodes(sdfg) == 1

    A_IN = xp.random.rand(DIM_SIZE, DIM_SIZE)
    A_OUT = xp.zeros_like(A_IN)
    B_IN = xp.random.rand(DIM_SIZE, DIM_SIZE)
    B_OUT = xp.zeros_like(B_IN)
    _set_lib_node_type(sdfg, expansion_type)
    sdfg.expand_library_nodes(recursive=True)
    sdfg.validate()
    sdfg(A_IN=A_IN, A_OUT=A_OUT, B_IN=B_IN, B_OUT=B_OUT)

    assert xp.allclose(A_IN, A_OUT)
    assert xp.allclose(B_OUT, 0.0)


@pytest.mark.parametrize("expansion_type", EXPANSION_TYPES)
def test_simple_with_extra_computation(expansion_type: str):
    if expansion_type == "CUDA":
        import cupy
    xp = cupy if expansion_type == "CUDA" else numpy

    sdfg = _get_sdfg(2, 2, True, False, False)
    AssignmentAndCopyKernelToMemsetAndMemcpy().apply_pass(sdfg, {})

    A_IN = xp.random.rand(DIM_SIZE, DIM_SIZE)
    A_OUT = xp.zeros_like(A_IN)
    B_IN = xp.random.rand(DIM_SIZE, DIM_SIZE)
    B_OUT = xp.zeros_like(B_IN)
    C_IN = xp.random.rand(DIM_SIZE, DIM_SIZE)
    C_OUT = xp.zeros_like(C_IN)
    D_IN = xp.random.rand(DIM_SIZE, DIM_SIZE)
    D_OUT = xp.zeros_like(D_IN)
    _set_lib_node_type(sdfg, expansion_type)
    sdfg.expand_library_nodes(recursive=True)
    sdfg.validate()
    sdfg(A_IN=A_IN, A_OUT=A_OUT, B_IN=B_IN, B_OUT=B_OUT, C_IN=C_IN, C_OUT=C_OUT, D_IN=D_IN, D_OUT=D_OUT)

    assert xp.allclose(A_OUT, 2 * A_IN)
    assert xp.allclose(B_OUT, B_IN)
    assert xp.allclose(C_OUT, 0.0)
    assert xp.allclose(D_OUT, 0.0)


@pytest.mark.parametrize("expansion_type", EXPANSION_TYPES)
def test_simple_non_zero(expansion_type: str):
    if expansion_type == "CUDA":
        import cupy
    xp = cupy if expansion_type == "CUDA" else numpy

    sdfg = _get_sdfg(0, 1, False, True, False)
    AssignmentAndCopyKernelToMemsetAndMemcpy().apply_pass(sdfg, {})

    A_IN = xp.random.rand(DIM_SIZE, DIM_SIZE)
    A_OUT = xp.zeros_like(A_IN)
    _set_lib_node_type(sdfg, expansion_type)
    sdfg.expand_library_nodes(recursive=True)
    sdfg.validate()
    sdfg(A_IN=A_OUT, A_OUT=A_OUT)

    assert xp.allclose(A_OUT, 1.0)


@pytest.mark.parametrize("expansion_type", EXPANSION_TYPES)
def test_mixed_overapprox(expansion_type: str):
    if expansion_type == "CUDA":
        import cupy
    xp = cupy if expansion_type == "CUDA" else numpy

    sdfg = _get_sdfg(2, 2, False, False, True)
    AssignmentAndCopyKernelToMemsetAndMemcpy().apply_pass(sdfg, {})
    sdfg.validate()

    A_IN = xp.random.rand(DIM_SIZE, DIM_SIZE)
    A_OUT = xp.zeros_like(A_IN)
    B_IN = xp.random.rand(DIM_SIZE, DIM_SIZE)
    B_OUT = xp.zeros_like(B_IN)
    C_IN = xp.random.rand(DIM_SIZE, DIM_SIZE)
    C_OUT = xp.zeros_like(C_IN)
    D_IN = xp.random.rand(DIM_SIZE, DIM_SIZE)
    D_OUT = xp.zeros_like(D_IN)

    _set_lib_node_type(sdfg, expansion_type)
    sdfg.expand_library_nodes(recursive=True)
    sdfg.validate()
    sdfg(A_IN=A_IN, A_OUT=A_OUT, B_IN=B_IN, B_OUT=B_OUT, C_IN=C_IN, C_OUT=C_OUT, D_IN=D_IN, D_OUT=D_OUT)

    assert xp.allclose(C_OUT, 0.0)
    assert xp.allclose(D_OUT, 0.0)
    assert xp.allclose(B_OUT[2:10, 0:10], B_IN[2:10, 0:10])
    assert xp.allclose(A_IN[2:10, 0:10], A_OUT[2:10, 0:10])


if __name__ == "__main__":
    for expansion_type in ["CPU", "pure", "GPU"]:
        test_simple_memcpy(expansion_type)
        test_simple_memset(expansion_type)
        test_multi_memcpy(expansion_type)
        test_multi_memset(expansion_type)
        test_multi_mixed(expansion_type)
        test_simple_with_extra_computation(expansion_type)
        test_simple_non_zero(expansion_type)
        test_mixed_overapprox(expansion_type)
        test_nested_maps(expansion_type)
