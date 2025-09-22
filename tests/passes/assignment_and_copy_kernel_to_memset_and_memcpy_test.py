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
    sdfg.save("x.sdfg")
    sdfg.validate()
    return sdfg


# --- Utility functions for counting nodes in an SDFG ---
def _get_num_memcpy_library_nodes(sdfg: dace.SDFG) -> int:
    """Return number of memcpy library nodes in an SDFG."""
    return sum(isinstance(node, CopyLibraryNode) for state in sdfg.all_states() for node in state.nodes())


def _get_num_memset_library_nodes(sdfg: dace.SDFG) -> int:
    """Return number of memset library nodes in an SDFG."""
    return sum(isinstance(node, MemsetLibraryNode) for state in sdfg.all_states() for node in state.nodes())


# --- Tests ---
def test_simple_memcpy():
    """Single memcpy test: output should match input exactly."""
    sdfg = _get_sdfg(1, 0, False, False, False)
    sdfg.validate()
    AssignmentAndCopyKernelToMemsetAndMemcpy().apply_pass(sdfg, {})
    sdfg.validate()
    sdfg.save("s1.sdfg")
    assert _get_num_memcpy_library_nodes(sdfg) == 1, "Expected 1 memcpy library node"
    assert _get_num_memset_library_nodes(sdfg) == 0, "Expected 0 memset library nodes"

    A_IN = numpy.random.rand(DIM_SIZE, DIM_SIZE)
    A_OUT = numpy.zeros_like(A_IN)

    sdfg(A_IN=A_IN, A_OUT=A_OUT)

    assert numpy.allclose(A_IN, A_OUT), "A_OUT does not match A_IN in simple memcpy"


def test_simple_memset():
    """Single memset test: output should be all zeros."""
    sdfg = _get_sdfg(0, 1, False, False, False)
    AssignmentAndCopyKernelToMemsetAndMemcpy().apply_pass(sdfg, {})
    sdfg.save("x2.sdfg")
    assert _get_num_memcpy_library_nodes(sdfg) == 0, "Expected 0 memcpy library nodes"
    assert _get_num_memset_library_nodes(sdfg) == 1, "Expected 0 memset library node"

    A_IN = numpy.random.rand(DIM_SIZE, DIM_SIZE)
    A_OUT = numpy.zeros_like(A_IN)

    sdfg(A_IN=A_IN, A_OUT=A_OUT)

    assert numpy.allclose(A_OUT, 0.0), "A_OUT is not zero in simple memset"


def test_multi_memcpy():
    """Two memcpies: each output should equal its corresponding input."""
    sdfg = _get_sdfg(2, 0, False, False, False)
    sdfg.save("x1x.sdfg")
    AssignmentAndCopyKernelToMemsetAndMemcpy().apply_pass(sdfg, {})
    sdfg.save("x3x.sdfg")

    assert _get_num_memcpy_library_nodes(sdfg) == 2, "Expected 2 memcpy library nodes"
    assert _get_num_memset_library_nodes(sdfg) == 0, "Expected 0 memset library nodes"

    A_IN = numpy.random.rand(DIM_SIZE, DIM_SIZE)
    A_OUT = numpy.zeros_like(A_IN)
    B_IN = numpy.random.rand(DIM_SIZE, DIM_SIZE)
    B_OUT = numpy.zeros_like(B_IN)

    sdfg(A_IN=A_IN, A_OUT=A_OUT, B_IN=B_IN, B_OUT=B_OUT)

    assert numpy.allclose(A_IN, A_OUT), "A_OUT does not match A_IN in multi memcpy"
    assert numpy.allclose(B_IN, B_OUT), "B_OUT does not match B_IN in multi memcpy"


def test_multi_memset():
    """Two memsets: both outputs should be all zeros."""
    sdfg = _get_sdfg(0, 2, False, False, False)
    AssignmentAndCopyKernelToMemsetAndMemcpy().apply_pass(sdfg, {})

    assert _get_num_memcpy_library_nodes(sdfg) == 0, "Expected 0 memcpy library nodes"
    assert _get_num_memset_library_nodes(sdfg) == 2, "Expected 2 memset library nodes"

    A_IN = numpy.random.rand(DIM_SIZE, DIM_SIZE)
    A_OUT = numpy.zeros_like(A_IN)
    B_IN = numpy.random.rand(DIM_SIZE, DIM_SIZE)
    B_OUT = numpy.zeros_like(B_IN)

    sdfg(A_IN=A_IN, A_OUT=A_OUT, B_IN=B_IN, B_OUT=B_OUT)

    assert numpy.allclose(A_OUT, 0.0), "A_OUT is not zero in multi memset"
    assert numpy.allclose(B_OUT, 0.0), "B_OUT is not zero in multi memset"


def test_multi_mixed():
    """One memcpy and one memset: memcpy should copy, memset should zero output."""
    sdfg = _get_sdfg(1, 1, False, False, False)
    AssignmentAndCopyKernelToMemsetAndMemcpy().apply_pass(sdfg, {})

    assert _get_num_memcpy_library_nodes(sdfg) == 1, "Expected 1 memcpy library node"
    assert _get_num_memset_library_nodes(sdfg) == 1, "Expected 1 memset library node"

    A_IN = numpy.random.rand(DIM_SIZE, DIM_SIZE)
    A_OUT = numpy.zeros_like(A_IN)
    B_IN = numpy.random.rand(DIM_SIZE, DIM_SIZE)
    B_OUT = numpy.zeros_like(B_IN)

    sdfg(A_IN=A_IN, A_OUT=A_OUT, B_IN=B_IN, B_OUT=B_OUT)

    assert numpy.allclose(A_IN, A_OUT), "A_OUT does not match A_IN in mixed memcpy/memset"
    assert numpy.allclose(B_OUT, 0.0), "B_OUT is not zero in mixed memcpy/memset"


def test_simple_with_extra_computation():
    """
    Add extra computation (doubling) to every other tasklet.
    Expect memcpy results to be doubled in output, memsets remain zero.
    """
    sdfg = _get_sdfg(2, 2, True, False, False)
    AssignmentAndCopyKernelToMemsetAndMemcpy().apply_pass(sdfg, {})

    assert _get_num_memcpy_library_nodes(sdfg) == 1, "Expected 1 memcpy library node"
    assert _get_num_memset_library_nodes(sdfg) == 1, "Expected 1 memset library node"

    A_IN = numpy.random.rand(DIM_SIZE, DIM_SIZE)
    A_OUT = numpy.zeros_like(A_IN)
    B_IN = numpy.random.rand(DIM_SIZE, DIM_SIZE)
    B_OUT = numpy.zeros_like(B_IN)
    C_IN = numpy.random.rand(DIM_SIZE, DIM_SIZE)
    C_OUT = numpy.zeros_like(C_IN)
    D_IN = numpy.random.rand(DIM_SIZE, DIM_SIZE)
    D_OUT = numpy.zeros_like(D_IN)

    sdfg(A_IN=A_IN, A_OUT=A_OUT, B_IN=B_IN, B_OUT=B_OUT, C_IN=C_IN, C_OUT=C_OUT, D_IN=D_IN, D_OUT=D_OUT)

    # A_OUT should be double of A_IN
    assert numpy.allclose(A_OUT, 2 * A_IN), "A_OUT does not match expected doubled values"
    # memcpy B should be unchanged
    assert numpy.allclose(B_OUT, B_IN), "B_OUT does not match B_IN with extra computation"
    # memsets should zero the outputs
    assert numpy.allclose(C_OUT, 0.0), "C_OUT is not zero with extra computation"
    assert numpy.allclose(D_OUT, 0.0), "D_OUT is not zero with extra computation"


def test_simple_non_zero():
    """Memset with non_zero=True should fill with ones instead of zeros."""
    sdfg = _get_sdfg(0, 1, False, True, False)
    AssignmentAndCopyKernelToMemsetAndMemcpy().apply_pass(sdfg, {})

    assert _get_num_memcpy_library_nodes(sdfg) == 0, "Expected 0 memcpy library nodes"
    assert _get_num_memset_library_nodes(sdfg) == 0, "Expected 0 memset library nodes"

    A_IN = numpy.random.rand(DIM_SIZE, DIM_SIZE)
    A_OUT = numpy.zeros_like(A_IN)

    sdfg(A_IN=A_OUT, A_OUT=A_OUT)

    assert numpy.allclose(A_OUT, 1.0), "A_OUT is not filled with ones in non-zero memset"


def test_mixed_overapprox():
    """
    Overapproximation test (currently marked as TODO).
    Checks mixed memcpy and memset with more than one of each.
    """
    sdfg = _get_sdfg(2, 2, False, False, True)
    AssignmentAndCopyKernelToMemsetAndMemcpy().apply_pass(sdfg, {})

    assert _get_num_memcpy_library_nodes(sdfg) == 2, "Expected 2 memcpy library nodes"
    assert _get_num_memset_library_nodes(sdfg) == 2, "Expected 2 memset library nodes"

    A_IN = numpy.random.rand(DIM_SIZE, DIM_SIZE)
    A_OUT = numpy.zeros_like(A_IN)
    B_IN = numpy.random.rand(DIM_SIZE, DIM_SIZE)
    B_OUT = numpy.zeros_like(B_IN)
    C_IN = numpy.random.rand(DIM_SIZE, DIM_SIZE)
    C_OUT = numpy.zeros_like(C_IN)
    D_IN = numpy.random.rand(DIM_SIZE, DIM_SIZE)
    D_OUT = numpy.zeros_like(D_IN)

    sdfg(A_IN=A_IN, A_OUT=A_OUT, B_IN=B_IN, B_OUT=B_OUT, C_IN=C_IN, C_OUT=C_OUT, D_IN=D_IN, D_OUT=D_OUT)

    assert numpy.allclose(A_IN, A_OUT), f"A_OUT does not match A_IN in mixed overapprox {A_IN - A_OUT}"
    assert numpy.allclose(B_OUT, B_IN), "B_OUT does not match B_IN in mixed overapprox"
    assert numpy.allclose(C_OUT, 0.0), "C_OUT is not zero in mixed overapprox"
    assert numpy.allclose(D_OUT, 0.0), "D_OUT is not zero in mixed overapprox"


if __name__ == "__main__":
    test_simple_memcpy()
    test_simple_memset()
    test_multi_memcpy()
    test_multi_memset()
    test_multi_mixed()
    test_simple_with_extra_computation()
    test_simple_non_zero()
    test_mixed_overapprox()
