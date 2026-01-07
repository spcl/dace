# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import functools
import dace
import numpy
import pytest
from dace.libraries.standard.nodes.copy_node import CopyLibraryNode
from dace.libraries.standard.nodes.memset_node import MemsetLibraryNode
from dace.properties import CodeBlock
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.assignment_and_copy_kernel_to_memset_and_memcpy import AssignmentAndCopyKernelToMemsetAndMemcpy

# Global dimension size for all test arrays
DIM_SIZE = 10
D = dace.symbol("D")
EXPANSION_TYPES = ["pure", "CPU", pytest.param("CUDA", marks=pytest.mark.gpu)]


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


def _get_num_memcpy_library_nodes(sdfg: dace.SDFG) -> int:
    """Return number of memcpy library nodes in an SDFG."""
    return sum(isinstance(node, CopyLibraryNode) for node, state in sdfg.all_nodes_recursive())


def _get_num_memset_library_nodes(sdfg: dace.SDFG) -> int:
    """Return number of memset library nodes in an SDFG."""
    return sum(isinstance(node, MemsetLibraryNode) for node, state in sdfg.all_nodes_recursive())


def _set_lib_node_type(sdfg: dace.SDFG, expansion_type: str):
    for n, g in sdfg.all_nodes_recursive():
        if isinstance(n, (CopyLibraryNode, MemsetLibraryNode)):
            n.implementation = expansion_type


def temporarily_disable_autoopt_and_serialization(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Save original values
        orig_autoopt = dace.config.Config.get("optimizer", "autooptimize")
        orig_serialization = dace.config.Config.get("testing", "serialization")
        try:
            # Set both to False
            dace.config.Config.set("optimizer", "autooptimize", value=False)
            dace.config.Config.set("testing", "serialization", value=False)
            return func(*args, **kwargs)
        finally:
            # Restore original values
            dace.config.Config.set("optimizer", "autooptimize", value=orig_autoopt)
            dace.config.Config.set("testing", "serialization", value=orig_serialization)

    return wrapper


@dace.program
def double_memset_with_dynamic_connectors(kfdia: dace.int32, kidia: dace.int32, llindex3: dace.float64[D, D],
                                          zsinksum: dace.float64[D]):
    for i, j in dace.map[0:D:1, kidia - 1:kfdia:]:
        llindex3[i, j] = 0.0
    for j in dace.map[kidia - 1:kfdia:1]:
        zsinksum[j] = 0.0


@dace.program
def double_memcpy_with_dynamic_connectors(kfdia: dace.int32, kidia: dace.int32, llindex3_in: dace.float64[D, D],
                                          zsinksum_in: dace.float64[D], llindex3_out: dace.float64[D, D],
                                          zsinksum_out: dace.float64[D]):
    for i, j in dace.map[0:D:1, kidia - 1:kfdia:]:
        llindex3_out[i, j] = llindex3_in[i, j]
    for j in dace.map[kidia - 1:kfdia:1]:
        zsinksum_out[j] = zsinksum_in[j]


@dace.program
def nested_memset_maps_with_dynamic_connectors(kidia: dace.int64, kfdia: dace.int64, llindex: dace.float64[5, 5, D],
                                               zsinksum: dace.float64[5, D]):
    for i in dace.map[0:5]:
        sym_kidia = kidia
        sym_kfdia = kfdia
        for j, k in dace.map[0:5, sym_kidia:sym_kfdia:1]:
            llindex[i, j, k] = 0.0
        for k in dace.map[sym_kidia:sym_kfdia:1]:
            zsinksum[i, k] = 0.0


@dace.program
def nested_memcpy_maps_with_dynamic_connectors(kidia: dace.int64, kfdia: dace.int64, llindex_in: dace.float64[5, 5, D],
                                               zsinksum_in: dace.float64[5, D], llindex_out: dace.float64[5, 5, D],
                                               zsinksum_out: dace.float64[5, D]):
    for i in dace.map[0:5]:
        sym_kidia = kidia
        sym_kfdia = kfdia
        for j, k in dace.map[0:5, sym_kidia:sym_kfdia:1]:
            llindex_out[i, j, k] = llindex_in[i, j, k]
        for k in dace.map[sym_kidia:sym_kfdia:1]:
            zsinksum_out[i, k] = zsinksum_in[i, k]


@dace.program
def nested_memcpy_maps_with_dimension_change(kidia: dace.int64, kfdia: dace.int64, zcovptot: dace.float64[D],
                                             pcovptot: dace.float64[D, D]):
    for i in range(D):
        sym_kidia = kidia
        sym_kfdia = kfdia
        for j in dace.map[sym_kidia:sym_kfdia]:
            pcovptot[i, j] = zcovptot[j]


@dace.program
def nested_memset_maps_with_dimension_change(kidia: dace.int64, kfdia: dace.int64, pcovptot: dace.float64[D, D]):
    for i in range(D):
        sym_kidia = kidia
        sym_kfdia = kfdia
        for j in dace.map[sym_kidia:sym_kfdia]:
            pcovptot[i, j] = 0.0


def set_dtype_to_gpu_if_expansion_type_is_cuda(sdfg: dace.SDFG, expansion_type: str):
    if expansion_type != "CUDA":
        return

    for arr_name, arr in sdfg.arrays.items():
        if not isinstance(arr, dace.data.Scalar):
            arr.storage = dace.dtypes.StorageType.GPU_Global
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                set_dtype_to_gpu_if_expansion_type_is_cuda(node.sdfg, expansion_type)


@pytest.mark.parametrize("expansion_type", EXPANSION_TYPES)
@temporarily_disable_autoopt_and_serialization
def test_nested_memcpy_maps_with_dimension_change(expansion_type: str):
    if expansion_type == "CUDA":
        import cupy
    xp = cupy if expansion_type == "CUDA" else numpy

    sdfg = nested_memcpy_maps_with_dimension_change.to_sdfg()
    sdfg.name = sdfg.name + f"_expansion_type_{expansion_type}"
    set_dtype_to_gpu_if_expansion_type_is_cuda(sdfg, expansion_type)

    AssignmentAndCopyKernelToMemsetAndMemcpy(overapproximate_first_dimensions=True).apply_pass(sdfg, {})
    assert _get_num_memcpy_library_nodes(
        sdfg) == 1, f"Expected 1 memcpy library nodes, got {_get_num_memcpy_library_nodes(sdfg)}"
    assert _get_num_memset_library_nodes(
        sdfg) == 0, f"Expected 0 memset library nodes, got {_get_num_memset_library_nodes(sdfg)}"

    kidia = 0
    kfdia = DIM_SIZE
    A_IN = xp.random.rand(DIM_SIZE)
    B_IN = xp.random.rand(DIM_SIZE, DIM_SIZE)
    _set_lib_node_type(sdfg, expansion_type)
    sdfg.expand_library_nodes(recursive=True)
    sdfg.validate()
    sdfg(zcovptot=A_IN, pcovptot=B_IN, kidia=kidia, kfdia=kfdia, D=DIM_SIZE)
    assert xp.allclose(A_IN, B_IN)


@pytest.mark.parametrize("expansion_type", EXPANSION_TYPES)
@temporarily_disable_autoopt_and_serialization
def test_nested_memset_maps_with_dimension_change(expansion_type: str):
    if expansion_type == "CUDA":
        import cupy
    xp = cupy if expansion_type == "CUDA" else numpy

    sdfg = nested_memset_maps_with_dimension_change.to_sdfg()
    sdfg.name = sdfg.name + f"_expansion_type_{expansion_type}"
    set_dtype_to_gpu_if_expansion_type_is_cuda(sdfg, expansion_type)

    AssignmentAndCopyKernelToMemsetAndMemcpy(overapproximate_first_dimensions=True).apply_pass(sdfg, {})
    assert _get_num_memset_library_nodes(
        sdfg) == 1, f"Expected 1 memset library nodes, got {_get_num_memset_library_nodes(sdfg)}"
    assert _get_num_memcpy_library_nodes(
        sdfg) == 0, f"Expected 0 memcpy library nodes, got {_get_num_memcpy_library_nodes(sdfg)}"

    kidia = 0
    kfdia = DIM_SIZE
    B_IN = xp.random.rand(DIM_SIZE, DIM_SIZE)
    _set_lib_node_type(sdfg, expansion_type)
    sdfg.expand_library_nodes(recursive=True)
    sdfg.validate()
    sdfg(pcovptot=B_IN, kidia=kidia, kfdia=kfdia, D=DIM_SIZE)
    assert xp.allclose(B_IN, 0.0)


@pytest.mark.parametrize("expansion_type", EXPANSION_TYPES)
@temporarily_disable_autoopt_and_serialization
def test_nested_memset_maps_with_dynamic_connectors(expansion_type: str):
    if expansion_type == "CUDA":
        # CUDA expansion type is not possible for this kernel
        # because due to the nested nature we will have memcpy/memset inside a kernel
        # the "choose best expansion" logic needs to be implemented and tested separately
        return
    xp = cupy if expansion_type == "CUDA" else numpy

    sdfg = nested_memset_maps_with_dynamic_connectors.to_sdfg()
    sdfg.name = sdfg.name + f"_expansion_type_{expansion_type}"
    set_dtype_to_gpu_if_expansion_type_is_cuda(sdfg, expansion_type)

    AssignmentAndCopyKernelToMemsetAndMemcpy(overapproximate_first_dimensions=False).apply_pass(sdfg, {})
    # We should have 0 memset libnodes
    assert _get_num_memset_library_nodes(
        sdfg) == 1, f"Expected 1 memset library nodes, got {_get_num_memset_library_nodes(sdfg)}"
    AssignmentAndCopyKernelToMemsetAndMemcpy(overapproximate_first_dimensions=True).apply_pass(sdfg, {})
    # We should have 2 memset libnodes
    assert _get_num_memset_library_nodes(
        sdfg) == 2, f"Expected 2 memset library nodes, got {_get_num_memset_library_nodes(sdfg)}"

    kidia = 0
    kfdia = DIM_SIZE
    A_IN = xp.random.rand(5, 5, DIM_SIZE)
    B_IN = xp.random.rand(5, DIM_SIZE)

    _set_lib_node_type(sdfg, expansion_type)
    sdfg.expand_library_nodes(recursive=True)
    from dace.sdfg import infer_types
    infer_types.set_default_schedule_and_storage_types(sdfg, None)
    sdfg.validate()
    sdfg.save("a.sdfg")
    sdfg(llindex=A_IN, zsinksum=B_IN, kidia=kidia, kfdia=kfdia, D=DIM_SIZE)
    assert xp.allclose(A_IN, 0.0)
    assert xp.allclose(B_IN, 0.0)


@pytest.mark.parametrize("expansion_type", EXPANSION_TYPES)
@temporarily_disable_autoopt_and_serialization
def test_nested_memcpy_maps_with_dynamic_connectors(expansion_type: str):
    if expansion_type == "CUDA":
        # CUDA expansion type is not possible for this kernel
        # because due to the nested nature we will have memcpy/memset inside a kernel
        # the "choose best expansion" logic needs to be implemented and tested separately
        return
    xp = cupy if expansion_type == "CUDA" else numpy

    sdfg = nested_memcpy_maps_with_dynamic_connectors.to_sdfg()
    sdfg.name = sdfg.name + f"_expansion_type_{expansion_type}"
    set_dtype_to_gpu_if_expansion_type_is_cuda(sdfg, expansion_type)

    AssignmentAndCopyKernelToMemsetAndMemcpy(overapproximate_first_dimensions=False).apply_pass(sdfg, {})
    # We should have 0 memcpy libnodes
    assert _get_num_memcpy_library_nodes(
        sdfg) == 1, f"Expected 1 memcpy library nodes, got {_get_num_memcpy_library_nodes(sdfg)}"
    AssignmentAndCopyKernelToMemsetAndMemcpy(overapproximate_first_dimensions=True).apply_pass(sdfg, {})
    # We should have 2 memcpy libnodes
    assert _get_num_memcpy_library_nodes(
        sdfg) == 2, f"Expected 2 memcpy library nodes, got {_get_num_memcpy_library_nodes(sdfg)}"

    kidia = 0
    kfdia = DIM_SIZE
    A_IN = xp.random.rand(5, 5, DIM_SIZE)
    A_OUT = xp.random.rand(5, 5, DIM_SIZE)
    B_IN = xp.random.rand(5, DIM_SIZE)
    B_OUT = xp.random.rand(5, DIM_SIZE)
    _set_lib_node_type(sdfg, expansion_type)
    sdfg.expand_library_nodes(recursive=True)
    sdfg.validate()
    sdfg(llindex_in=A_IN, zsinksum_in=B_IN, llindex_out=A_OUT, zsinksum_out=B_OUT, kidia=kidia, kfdia=kfdia, D=DIM_SIZE)
    assert xp.allclose(A_IN, A_OUT)
    assert xp.allclose(B_IN, B_OUT)


@pytest.mark.parametrize("expansion_type", EXPANSION_TYPES)
@temporarily_disable_autoopt_and_serialization
def test_double_memset_with_dynamic_connectors(expansion_type: str):
    if expansion_type == "CUDA":
        import cupy
    xp = cupy if expansion_type == "CUDA" else numpy

    sdfg = double_memset_with_dynamic_connectors.to_sdfg()
    sdfg.name = sdfg.name + f"_expansion_type_{expansion_type}"
    set_dtype_to_gpu_if_expansion_type_is_cuda(sdfg, expansion_type)

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
@temporarily_disable_autoopt_and_serialization
def test_double_memcpy_with_dynamic_connectors(expansion_type: str):
    if expansion_type == "CUDA":
        import cupy
    xp = cupy if expansion_type == "CUDA" else numpy

    sdfg = double_memcpy_with_dynamic_connectors.to_sdfg()
    sdfg.name = sdfg.name + f"_expansion_type_{expansion_type}"
    set_dtype_to_gpu_if_expansion_type_is_cuda(sdfg, expansion_type)

    A_IN = xp.random.rand(DIM_SIZE, DIM_SIZE)
    B_IN = xp.random.rand(DIM_SIZE)
    A_OUT = xp.random.rand(DIM_SIZE, DIM_SIZE)
    B_OUT = xp.random.rand(DIM_SIZE)

    sdfg.validate()
    p = AssignmentAndCopyKernelToMemsetAndMemcpy(overapproximate_first_dimensions=True)
    p.overapproximate_first_dimension = True
    p.apply_pass(sdfg, {})
    for n, g in sdfg.all_nodes_recursive():
        if isinstance(n, dace.nodes.NestedSDFG):
            p.apply_pass(n.sdfg)
    sdfg.validate()
    assert _get_num_memcpy_library_nodes(
        sdfg) == 2, f"Expected 2 memcpy library node, got {_get_num_memcpy_library_nodes(sdfg)}"
    assert _get_num_memset_library_nodes(
        sdfg) == 0, f"Expected 0 memset library nodes, got {_get_num_memset_library_nodes(sdfg)}"

    sdfg.expand_library_nodes(recursive=True)
    sdfg.validate()
    _set_lib_node_type(sdfg, expansion_type)
    sdfg.expand_library_nodes(recursive=True)
    sdfg.validate()
    sdfg(llindex3_in=A_IN,
         zsinksum_in=B_IN,
         llindex3_out=A_OUT,
         zsinksum_out=B_OUT,
         D=DIM_SIZE,
         kfdia=1,
         kidia=DIM_SIZE)

    assert xp.all(B_IN == B_OUT)
    assert xp.all(A_IN == A_OUT)


@pytest.mark.parametrize("expansion_type", EXPANSION_TYPES)
@temporarily_disable_autoopt_and_serialization
def test_simple_memcpy(expansion_type: str):
    if expansion_type == "CUDA":
        import cupy
    xp = cupy if expansion_type == "CUDA" else numpy

    sdfg = _get_sdfg(1, 0, False, False, False)
    sdfg.validate()
    sdfg.name = sdfg.name + f"_simple_memcpy_expansion_type_{expansion_type}"
    set_dtype_to_gpu_if_expansion_type_is_cuda(sdfg, expansion_type)

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
@temporarily_disable_autoopt_and_serialization
def test_simple_memset(expansion_type: str):
    if expansion_type == "CUDA":
        import cupy
    xp = cupy if expansion_type == "CUDA" else numpy

    sdfg = _get_sdfg(0, 1, False, False, False)
    sdfg.name = sdfg.name + f"_simple_memset_expansion_type_{expansion_type}"
    set_dtype_to_gpu_if_expansion_type_is_cuda(sdfg, expansion_type)

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
@temporarily_disable_autoopt_and_serialization
def test_multi_memcpy(expansion_type: str):
    if expansion_type == "CUDA":
        import cupy
    xp = cupy if expansion_type == "CUDA" else numpy

    sdfg = _get_sdfg(2, 0, False, False, False)
    sdfg.validate()
    sdfg.name = sdfg.name + f"_multi_memcpy_expansion_type_{expansion_type}"
    set_dtype_to_gpu_if_expansion_type_is_cuda(sdfg, expansion_type)

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
@temporarily_disable_autoopt_and_serialization
def test_multi_memset(expansion_type: str):
    if expansion_type == "CUDA":
        import cupy
    xp = cupy if expansion_type == "CUDA" else numpy

    sdfg = _get_sdfg(0, 2, False, False, False)
    sdfg.validate()
    sdfg.name = sdfg.name + f"_multi_memset_expansion_type_{expansion_type}"
    set_dtype_to_gpu_if_expansion_type_is_cuda(sdfg, expansion_type)

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
@temporarily_disable_autoopt_and_serialization
def test_multi_mixed(expansion_type: str):
    if expansion_type == "CUDA":
        import cupy
    xp = cupy if expansion_type == "CUDA" else numpy

    sdfg = _get_sdfg(1, 1, False, False, False)
    sdfg.validate()
    sdfg.name = sdfg.name + f"_multi_mixed_expansion_type_{expansion_type}"
    set_dtype_to_gpu_if_expansion_type_is_cuda(sdfg, expansion_type)

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
@temporarily_disable_autoopt_and_serialization
def test_simple_with_extra_computation(expansion_type: str):
    if expansion_type == "CUDA":
        import cupy
    xp = cupy if expansion_type == "CUDA" else numpy

    sdfg = _get_sdfg(2, 2, True, False, False)
    sdfg.validate()
    sdfg.name = sdfg.name + f"_simple_with_extra_computation_expansion_type_{expansion_type}"
    set_dtype_to_gpu_if_expansion_type_is_cuda(sdfg, expansion_type)
    sdfg.save("x1.sdfg")
    AssignmentAndCopyKernelToMemsetAndMemcpy().apply_pass(sdfg, {})
    sdfg.save("x2.sdfg")

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
@temporarily_disable_autoopt_and_serialization
def test_simple_non_zero(expansion_type: str):
    if expansion_type == "CUDA":
        import cupy
    xp = cupy if expansion_type == "CUDA" else numpy

    sdfg = _get_sdfg(0, 1, False, True, False)
    sdfg.validate()
    sdfg.name = sdfg.name + f"_simple_nonzero_expansion_type_{expansion_type}"
    set_dtype_to_gpu_if_expansion_type_is_cuda(sdfg, expansion_type)

    AssignmentAndCopyKernelToMemsetAndMemcpy().apply_pass(sdfg, {})

    A_IN = xp.random.rand(DIM_SIZE, DIM_SIZE)
    A_OUT = xp.zeros_like(A_IN)
    _set_lib_node_type(sdfg, expansion_type)
    sdfg.expand_library_nodes(recursive=True)
    sdfg.validate()
    sdfg(A_IN=A_OUT, A_OUT=A_OUT)

    assert xp.allclose(A_OUT, 1.0)


@pytest.mark.parametrize("expansion_type", EXPANSION_TYPES)
@temporarily_disable_autoopt_and_serialization
def test_mixed_overapprox(expansion_type: str):
    if expansion_type == "CUDA":
        import cupy
    xp = cupy if expansion_type == "CUDA" else numpy

    sdfg = _get_sdfg(2, 2, False, False, True)
    sdfg.name = sdfg.name + f"_mixed_overapprox_expansion_type_{expansion_type}"
    set_dtype_to_gpu_if_expansion_type_is_cuda(sdfg, expansion_type)

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


def _get_nested_memcpy_with_dimension_change_and_fortran_strides(full_inner_range: bool = True,
                                                                 fortran_strides: bool = True):
    sdfg = dace.SDFG("nested_memcpy_with_dimension_change_and_fortran_strides")
    inner_sdfg = dace.SDFG(name="inner_sdfg")

    for sd in [sdfg, inner_sdfg]:
        sd.add_symbol("_for_it_0", dace.int64)
        sd.add_symbol("D", dace.int64)

    scl_names = ["kfdia", "kidia"]

    for sd in [sdfg, inner_sdfg]:
        for scl_name in scl_names:
            sd.add_scalar(name=scl_name, dtype=dace.int64)
        for arr_name, shape, strides in [("zcovptot", (D, ), (1, )),
                                         ("pcovptot", (D, D), (1, D) if fortran_strides else (D, 1))]:
            if not full_inner_range and arr_name == "pcovptot" and sd == inner_sdfg:
                sd.add_array(
                    name=arr_name,
                    shape=(D, ),
                    dtype=dace.float64,
                    transient=False,
                    strides=(1, ) if fortran_strides else (D, ),
                )
            else:
                sd.add_array(
                    name=arr_name,
                    shape=shape,
                    dtype=dace.float64,
                    transient=False,
                    strides=strides,
                )

    for_cfg = LoopRegion(label="for1",
                         condition_expr=CodeBlock("_for_it_0 < D"),
                         loop_var="_for_it_0",
                         initialize_expr=CodeBlock("_for_it_0 = 0"),
                         update_expr=CodeBlock("_for_it_0 = _for_it_0 + 1"))
    sdfg.add_node(for_cfg, True)
    inner_state = for_cfg.add_state(label="s1", is_start_block=True)
    nsdfg_node = inner_state.add_nested_sdfg(
        sdfg=inner_sdfg,
        inputs={"kfdia", "kidia", "zcovptot"},
        outputs={"pcovptot"},
        symbol_mapping={
            "_for_it_0": "_for_it_0",
            "D": "D"
        },
        name="inner_sdfg_node",
    )
    assert "_for_it_0" in inner_sdfg.symbols
    assert "_for_it_0" in sdfg.symbols
    assert "_for_it_0" not in sdfg.free_symbols
    assert "_for_it_0" in inner_sdfg.free_symbols

    inner_inner_state = inner_sdfg.add_state(label="s2", is_start_block=True)

    for in_name in {"kfdia", "kidia", "zcovptot"}:
        inner_state.add_edge(inner_state.add_access(in_name), None, nsdfg_node, in_name,
                             dace.memlet.Memlet.from_array(in_name, sdfg.arrays[in_name]))

    for out_name in {"pcovptot"}:
        inner_state.add_edge(
            nsdfg_node, out_name, inner_state.add_access(out_name), None,
            dace.memlet.Memlet("pcovptot[0:D, _for_it_0]" if not full_inner_range else "pcovptot[0:D, 0:D]"))

    inner_inner_state.add_mapped_tasklet(
        name="cpy",
        map_ranges={"i": dace.subsets.Range([(0, D - 1, 1)])},
        input_nodes={"zcovptot": inner_inner_state.add_access("zcovptot")},
        output_nodes={"pcovptot": inner_inner_state.add_access("pcovptot")},
        external_edges=True,
        code="_out = _in",
        inputs={"_in": dace.memlet.Memlet("zcovptot[i]")},
        outputs={"_out": dace.memlet.Memlet("pcovptot[i, _for_it_0]" if full_inner_range else "pcovptot[i]")},
    )
    sdfg.validate()
    sdfg.save("y.sdfg")
    return sdfg


@pytest.mark.parametrize("expansion_type", EXPANSION_TYPES)
@temporarily_disable_autoopt_and_serialization
def test_nested_memcpy_with_dimension_change_and_fortran_strides(expansion_type: str):
    if expansion_type == "CUDA":
        import cupy
    xp = cupy if expansion_type == "CUDA" else numpy

    sdfg = _get_nested_memcpy_with_dimension_change_and_fortran_strides(full_inner_range=True, fortran_strides=True)
    sdfg.name = sdfg.name + f"_full_inner_range_true_fortran_strides_true_expansion_type_{expansion_type}"
    set_dtype_to_gpu_if_expansion_type_is_cuda(sdfg, expansion_type)

    AssignmentAndCopyKernelToMemsetAndMemcpy(overapproximate_first_dimensions=True).apply_pass(sdfg, {})
    assert _get_num_memcpy_library_nodes(
        sdfg) == 1, f"Expected 1 memcpy library nodes, got {_get_num_memcpy_library_nodes(sdfg)}"
    assert _get_num_memset_library_nodes(
        sdfg) == 0, f"Expected 0 memset library nodes, got {_get_num_memset_library_nodes(sdfg)}"

    kidia = 0
    kfdia = DIM_SIZE
    A_IN = xp.fromfunction(lambda x: x, (DIM_SIZE, ), dtype=xp.float64).copy()
    B_IN = xp.fromfunction(lambda x, y: x * DIM_SIZE + y, (DIM_SIZE, DIM_SIZE), dtype=xp.float64).copy()
    _set_lib_node_type(sdfg, expansion_type)
    sdfg.save("x1.sdfg")
    sdfg.expand_library_nodes(recursive=True)
    sdfg.validate()
    sdfg(zcovptot=A_IN, pcovptot=B_IN, kidia=kidia, kfdia=kfdia, D=DIM_SIZE)
    assert xp.allclose(A_IN, B_IN)


@pytest.mark.parametrize("expansion_type", EXPANSION_TYPES)
@temporarily_disable_autoopt_and_serialization
def test_nested_memcpy_with_dimension_change_and_fortran_strides_with_subset(expansion_type: str):
    if expansion_type == "CUDA":
        import cupy
    xp = cupy if expansion_type == "CUDA" else numpy

    sdfg = _get_nested_memcpy_with_dimension_change_and_fortran_strides(full_inner_range=False, fortran_strides=True)
    sdfg.name = sdfg.name + f"_full_inner_range_false_fortran_strides_true_expansion_type_{expansion_type}"
    set_dtype_to_gpu_if_expansion_type_is_cuda(sdfg, expansion_type)

    AssignmentAndCopyKernelToMemsetAndMemcpy(overapproximate_first_dimensions=True).apply_pass(sdfg, {})
    assert _get_num_memcpy_library_nodes(
        sdfg) == 1, f"Expected 1 memcpy library nodes, got {_get_num_memcpy_library_nodes(sdfg)}"
    assert _get_num_memset_library_nodes(
        sdfg) == 0, f"Expected 0 memset library nodes, got {_get_num_memset_library_nodes(sdfg)}"

    kidia = 0
    kfdia = DIM_SIZE
    A_IN = xp.fromfunction(lambda x: x, (DIM_SIZE, ), dtype=xp.float64).copy()
    B_IN = xp.fromfunction(lambda x, y: x * DIM_SIZE + y, (DIM_SIZE, DIM_SIZE), dtype=xp.float64).copy()
    _set_lib_node_type(sdfg, expansion_type)
    sdfg.expand_library_nodes(recursive=True)
    sdfg.validate()
    sdfg(zcovptot=A_IN, pcovptot=B_IN, kidia=kidia, kfdia=kfdia, D=DIM_SIZE)
    assert xp.allclose(A_IN, B_IN)


@pytest.mark.parametrize("expansion_type", EXPANSION_TYPES)
@temporarily_disable_autoopt_and_serialization
def test_nested_memcpy_with_dimension_change_and_c_strides(expansion_type: str):
    if expansion_type == "CUDA":
        import cupy
    xp = cupy if expansion_type == "CUDA" else numpy

    sdfg = _get_nested_memcpy_with_dimension_change_and_fortran_strides(full_inner_range=True, fortran_strides=False)
    sdfg.name = sdfg.name + f"_full_inner_range_true_fortran_strides_false_expansion_type_{expansion_type}"
    set_dtype_to_gpu_if_expansion_type_is_cuda(sdfg, expansion_type)

    AssignmentAndCopyKernelToMemsetAndMemcpy(overapproximate_first_dimensions=True).apply_pass(sdfg, {})
    assert _get_num_memcpy_library_nodes(
        sdfg) == 0, f"Expected 0 memcpy library nodes, got {_get_num_memcpy_library_nodes(sdfg)}"
    assert _get_num_memset_library_nodes(
        sdfg) == 0, f"Expected 0 memset library nodes, got {_get_num_memset_library_nodes(sdfg)}"

    kidia = 0
    kfdia = DIM_SIZE
    A_IN = xp.fromfunction(lambda x: x, (DIM_SIZE, ), dtype=xp.float64).copy()
    B_IN = xp.fromfunction(lambda x, y: x * DIM_SIZE + y, (DIM_SIZE, DIM_SIZE), dtype=xp.float64).copy()
    _set_lib_node_type(sdfg, expansion_type)
    sdfg.expand_library_nodes(recursive=True)
    sdfg.validate()
    sdfg(zcovptot=A_IN, pcovptot=B_IN, kidia=kidia, kfdia=kfdia, D=DIM_SIZE)
    for j in range(DIM_SIZE):
        assert xp.allclose(B_IN[0:DIM_SIZE, j], A_IN), f"{j}: {B_IN[0:DIM_SIZE, j] - A_IN}"


@pytest.mark.parametrize("expansion_type", EXPANSION_TYPES)
@temporarily_disable_autoopt_and_serialization
def test_nested_memcpy_with_dimension_change_and_c_strides_with_subset(expansion_type: str):
    if expansion_type == "CUDA":
        import cupy
    xp = cupy if expansion_type == "CUDA" else numpy

    sdfg = _get_nested_memcpy_with_dimension_change_and_fortran_strides(full_inner_range=False, fortran_strides=False)
    sdfg.name = sdfg.name + f"_full_inner_range_false_fortran_strides_false_expansion_type_{expansion_type}"
    set_dtype_to_gpu_if_expansion_type_is_cuda(sdfg, expansion_type)

    AssignmentAndCopyKernelToMemsetAndMemcpy(overapproximate_first_dimensions=True).apply_pass(sdfg, {})
    assert _get_num_memcpy_library_nodes(
        sdfg) == 0, f"Expected 0 memcpy library nodes, got {_get_num_memcpy_library_nodes(sdfg)}"
    assert _get_num_memset_library_nodes(
        sdfg) == 0, f"Expected 0 memset library nodes, got {_get_num_memset_library_nodes(sdfg)}"

    kidia = 0
    kfdia = DIM_SIZE
    A_IN = xp.fromfunction(lambda x: x, (DIM_SIZE, ), dtype=xp.float64).copy()
    B_IN = xp.fromfunction(lambda x, y: x * DIM_SIZE + y, (DIM_SIZE, DIM_SIZE), dtype=xp.float64).copy()
    _set_lib_node_type(sdfg, expansion_type)
    sdfg.expand_library_nodes(recursive=True)
    sdfg.validate()
    sdfg(zcovptot=A_IN, pcovptot=B_IN, kidia=kidia, kfdia=kfdia, D=DIM_SIZE)
    for j in range(DIM_SIZE):
        assert xp.allclose(B_IN[0:DIM_SIZE, j], A_IN), f"{j}: {B_IN[0:DIM_SIZE, j] - A_IN}"


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
        test_nested_memset_maps_with_dynamic_connectors(expansion_type)
        test_nested_memcpy_maps_with_dynamic_connectors(expansion_type)
        test_double_memset_with_dynamic_connectors(expansion_type)
        test_double_memcpy_with_dynamic_connectors(expansion_type)
        test_nested_memset_maps_with_dimension_change(expansion_type)
        test_nested_memcpy_maps_with_dimension_change(expansion_type)
        test_nested_memcpy_with_dimension_change_and_fortran_strides(expansion_type)
        test_nested_memcpy_with_dimension_change_and_fortran_strides_with_subset(expansion_type)
        test_nested_memcpy_with_dimension_change_and_c_strides(expansion_type)
        test_nested_memcpy_with_dimension_change_and_c_strides_with_subset(expansion_type)
