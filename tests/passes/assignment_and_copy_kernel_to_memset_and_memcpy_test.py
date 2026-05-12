# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

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
# Not supported: the CUDA expansion emits cudaMemsetAsync/cudaMemcpyAsync, which are host-side
# runtime calls and cannot execute from device code, so nesting a memset/memcpy library node
# inside a GPU kernel has no valid CUDA expansion.
EXPANSION_TYPES_CPU_ONLY = [
    "pure", "CPU",
    pytest.param("CUDA",
                 marks=pytest.mark.skip(reason="nested memset/memcpy inside a GPU kernel is unsupported: "
                                        "cudaMemsetAsync/cudaMemcpyAsync cannot be called from device code"))
]


@pytest.fixture
def xp(expansion_type):
    if expansion_type == "CUDA":
        import cupy
        return cupy
    return numpy


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

    sdfg.validate()
    return sdfg


def _get_num_memcpy_library_nodes(sdfg: dace.SDFG) -> int:
    return sum(isinstance(node, CopyLibraryNode) for node, state in sdfg.all_nodes_recursive())


def _get_num_memset_library_nodes(sdfg: dace.SDFG) -> int:
    return sum(isinstance(node, MemsetLibraryNode) for node, state in sdfg.all_nodes_recursive())


# MemsetLibraryNode kept the legacy ``pure`` / ``CPU`` / ``CUDA`` impl names;
# CopyLibraryNode renamed to ``MappedTasklet`` / ``MemcpyCPU`` / ``MemcpyCUDA1D``.
# Tests still parametrize on the legacy label and translate per type here.
_COPY_IMPL_FROM_EXPANSION_TYPE = {
    "pure": "MappedTasklet",
    "CPU": "MemcpyCPU",
    "CUDA": "MemcpyCUDA1D",
}


def _set_lib_node_type(sdfg: dace.SDFG, expansion_type: str):
    for n, g in sdfg.all_nodes_recursive():
        if isinstance(n, CopyLibraryNode):
            n.implementation = _COPY_IMPL_FROM_EXPANSION_TYPE.get(expansion_type, expansion_type)
        elif isinstance(n, MemsetLibraryNode):
            n.implementation = expansion_type


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


def temporarily_disable_autoopt_and_serialization(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        orig_autoopt = dace.config.Config.get("optimizer", "autooptimize")
        orig_serialization = dace.config.Config.get("testing", "serialization")
        try:
            dace.config.Config.set("optimizer", "autooptimize", value=False)
            dace.config.Config.set("testing", "serialization", value=False)
            return func(*args, **kwargs)
        finally:
            dace.config.Config.set("optimizer", "autooptimize", value=orig_autoopt)
            dace.config.Config.set("testing", "serialization", value=orig_serialization)

    return wrapper


def _sdfg_from_program(program) -> dace.SDFG:
    # simplify: nested-SDFG simplifications affect pass applicability
    sdfg = program.to_sdfg()
    sdfg.simplify()
    return sdfg


def _prepare_sdfg(sdfg: dace.SDFG, expansion_type: str, name_suffix: str = "") -> dace.SDFG:
    suffix = f"_{name_suffix}" if name_suffix else ""
    sdfg.name = sdfg.name + suffix + f"_expansion_type_{expansion_type}"
    set_dtype_to_gpu_if_expansion_type_is_cuda(sdfg, expansion_type)
    return sdfg


def _expand_and_validate(sdfg: dace.SDFG, expansion_type: str) -> None:
    _set_lib_node_type(sdfg, expansion_type)
    sdfg.expand_library_nodes(recursive=True)
    sdfg.validate()


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


@pytest.mark.parametrize("expansion_type", EXPANSION_TYPES)
@temporarily_disable_autoopt_and_serialization
def test_nested_memcpy_maps_with_dimension_change(expansion_type, xp):
    sdfg = _prepare_sdfg(_sdfg_from_program(nested_memcpy_maps_with_dimension_change), expansion_type)
    AssignmentAndCopyKernelToMemsetAndMemcpy(overapproximate_first_dimensions=True).apply_pass(sdfg, {})
    assert _get_num_memcpy_library_nodes(sdfg) == 1
    assert _get_num_memset_library_nodes(sdfg) == 0

    A_IN = xp.random.rand(DIM_SIZE)
    B_IN = xp.random.rand(DIM_SIZE, DIM_SIZE)
    _expand_and_validate(sdfg, expansion_type)
    sdfg(zcovptot=A_IN, pcovptot=B_IN, kidia=0, kfdia=DIM_SIZE, D=DIM_SIZE)
    assert xp.allclose(A_IN, B_IN)


@pytest.mark.parametrize("expansion_type", EXPANSION_TYPES)
@temporarily_disable_autoopt_and_serialization
def test_nested_memset_maps_with_dimension_change(expansion_type, xp):
    sdfg = _prepare_sdfg(_sdfg_from_program(nested_memset_maps_with_dimension_change), expansion_type)
    AssignmentAndCopyKernelToMemsetAndMemcpy(overapproximate_first_dimensions=True).apply_pass(sdfg, {})
    assert _get_num_memset_library_nodes(sdfg) == 1
    assert _get_num_memcpy_library_nodes(sdfg) == 0

    B_IN = xp.random.rand(DIM_SIZE, DIM_SIZE)
    _expand_and_validate(sdfg, expansion_type)
    sdfg(pcovptot=B_IN, kidia=0, kfdia=DIM_SIZE, D=DIM_SIZE)
    assert xp.allclose(B_IN, 0.0)


@pytest.mark.parametrize("expansion_type", EXPANSION_TYPES_CPU_ONLY)
@temporarily_disable_autoopt_and_serialization
def test_nested_memset_maps_with_dynamic_connectors(expansion_type, xp):
    sdfg = _prepare_sdfg(_sdfg_from_program(nested_memset_maps_with_dynamic_connectors), expansion_type)

    AssignmentAndCopyKernelToMemsetAndMemcpy(overapproximate_first_dimensions=False).apply_pass(sdfg, {})
    assert _get_num_memset_library_nodes(sdfg) == 1
    AssignmentAndCopyKernelToMemsetAndMemcpy(overapproximate_first_dimensions=True).apply_pass(sdfg, {})
    assert _get_num_memset_library_nodes(sdfg) == 2

    A_IN = xp.random.rand(5, 5, DIM_SIZE)
    B_IN = xp.random.rand(5, DIM_SIZE)

    _set_lib_node_type(sdfg, expansion_type)
    sdfg.expand_library_nodes(recursive=True)
    from dace.sdfg import infer_types
    infer_types.set_default_schedule_and_storage_types(sdfg, None)
    sdfg.validate()
    sdfg(llindex=A_IN, zsinksum=B_IN, kidia=0, kfdia=DIM_SIZE, D=DIM_SIZE)
    assert xp.allclose(A_IN, 0.0)
    assert xp.allclose(B_IN, 0.0)


@pytest.mark.parametrize("expansion_type", EXPANSION_TYPES_CPU_ONLY)
@temporarily_disable_autoopt_and_serialization
def test_nested_memcpy_maps_with_dynamic_connectors(expansion_type, xp):
    sdfg = _prepare_sdfg(_sdfg_from_program(nested_memcpy_maps_with_dynamic_connectors), expansion_type)

    AssignmentAndCopyKernelToMemsetAndMemcpy(overapproximate_first_dimensions=False).apply_pass(sdfg, {})
    assert _get_num_memcpy_library_nodes(sdfg) == 1
    AssignmentAndCopyKernelToMemsetAndMemcpy(overapproximate_first_dimensions=True).apply_pass(sdfg, {})
    assert _get_num_memcpy_library_nodes(sdfg) == 2

    A_IN = xp.random.rand(5, 5, DIM_SIZE)
    A_OUT = xp.random.rand(5, 5, DIM_SIZE)
    B_IN = xp.random.rand(5, DIM_SIZE)
    B_OUT = xp.random.rand(5, DIM_SIZE)
    _expand_and_validate(sdfg, expansion_type)
    sdfg(llindex_in=A_IN, zsinksum_in=B_IN, llindex_out=A_OUT, zsinksum_out=B_OUT, kidia=0, kfdia=DIM_SIZE, D=DIM_SIZE)
    assert xp.allclose(A_IN, A_OUT)
    assert xp.allclose(B_IN, B_OUT)


@pytest.mark.parametrize("expansion_type", EXPANSION_TYPES)
@temporarily_disable_autoopt_and_serialization
def test_double_memset_with_dynamic_connectors(expansion_type, xp):
    sdfg = _prepare_sdfg(_sdfg_from_program(double_memset_with_dynamic_connectors), expansion_type)

    A_IN = xp.random.rand(DIM_SIZE, DIM_SIZE)
    B_IN = xp.ones(DIM_SIZE)

    p = AssignmentAndCopyKernelToMemsetAndMemcpy()
    p.overapproximate_first_dimension = True
    p.apply_pass(sdfg, {})
    for n, g in sdfg.all_nodes_recursive():
        if isinstance(n, dace.nodes.NestedSDFG):
            p.apply_pass(n.sdfg, {})
    sdfg.validate()

    assert _get_num_memcpy_library_nodes(sdfg) == 0
    assert _get_num_memset_library_nodes(sdfg) == 2

    # Two-stage expansion: first with default impl, then force the chosen impl.
    sdfg.expand_library_nodes(recursive=True)
    sdfg.validate()
    _expand_and_validate(sdfg, expansion_type)
    sdfg(llindex3=A_IN, zsinksum=B_IN, D=DIM_SIZE, kfdia=1, kidia=DIM_SIZE)

    assert xp.all(B_IN == 0.0), f"zsinksum should be fully zeroed {B_IN}"
    assert xp.all(A_IN == 0.0), f"llindex3 should be fully zeroed {A_IN}"


@pytest.mark.parametrize("expansion_type", EXPANSION_TYPES)
@temporarily_disable_autoopt_and_serialization
def test_double_memcpy_with_dynamic_connectors(expansion_type, xp):
    sdfg = _prepare_sdfg(_sdfg_from_program(double_memcpy_with_dynamic_connectors), expansion_type)

    A_IN = xp.random.rand(DIM_SIZE, DIM_SIZE)
    B_IN = xp.random.rand(DIM_SIZE)
    A_OUT = xp.random.rand(DIM_SIZE, DIM_SIZE)
    B_OUT = xp.random.rand(DIM_SIZE)

    p = AssignmentAndCopyKernelToMemsetAndMemcpy(overapproximate_first_dimensions=True)
    p.overapproximate_first_dimension = True
    p.apply_pass(sdfg, {})
    for n, g in sdfg.all_nodes_recursive():
        if isinstance(n, dace.nodes.NestedSDFG):
            p.apply_pass(n.sdfg)
    sdfg.validate()
    assert _get_num_memcpy_library_nodes(sdfg) == 2
    assert _get_num_memset_library_nodes(sdfg) == 0

    # Two-stage expansion: first with default impl, then force the chosen impl.
    sdfg.expand_library_nodes(recursive=True)
    sdfg.validate()
    _expand_and_validate(sdfg, expansion_type)
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
def test_simple_memcpy(expansion_type, xp):
    sdfg = _prepare_sdfg(_get_sdfg(1, 0, False, False, False), expansion_type, "simple_memcpy")
    AssignmentAndCopyKernelToMemsetAndMemcpy().apply_pass(sdfg, {})
    sdfg.validate()
    assert _get_num_memcpy_library_nodes(sdfg) == 1
    assert _get_num_memset_library_nodes(sdfg) == 0

    A_IN = xp.random.rand(DIM_SIZE, DIM_SIZE)
    A_OUT = xp.zeros_like(A_IN)
    _expand_and_validate(sdfg, expansion_type)
    sdfg(A_IN=A_IN, A_OUT=A_OUT)

    assert xp.allclose(A_IN, A_OUT)


@pytest.mark.parametrize("expansion_type", EXPANSION_TYPES)
@temporarily_disable_autoopt_and_serialization
def test_simple_memset(expansion_type, xp):
    sdfg = _prepare_sdfg(_get_sdfg(0, 1, False, False, False), expansion_type, "simple_memset")
    AssignmentAndCopyKernelToMemsetAndMemcpy().apply_pass(sdfg, {})
    assert _get_num_memcpy_library_nodes(sdfg) == 0
    assert _get_num_memset_library_nodes(sdfg) == 1

    A_IN = xp.random.rand(DIM_SIZE, DIM_SIZE)
    A_OUT = xp.zeros_like(A_IN)
    _expand_and_validate(sdfg, expansion_type)
    sdfg(A_IN=A_IN, A_OUT=A_OUT)

    assert xp.allclose(A_OUT, 0.0)


@pytest.mark.parametrize("expansion_type", EXPANSION_TYPES)
@temporarily_disable_autoopt_and_serialization
def test_multi_memcpy(expansion_type, xp):
    sdfg = _prepare_sdfg(_get_sdfg(2, 0, False, False, False), expansion_type, "multi_memcpy")
    AssignmentAndCopyKernelToMemsetAndMemcpy().apply_pass(sdfg, {})
    assert _get_num_memcpy_library_nodes(sdfg) == 2
    assert _get_num_memset_library_nodes(sdfg) == 0

    A_IN = xp.random.rand(DIM_SIZE, DIM_SIZE)
    A_OUT = xp.zeros_like(A_IN)
    B_IN = xp.random.rand(DIM_SIZE, DIM_SIZE)
    B_OUT = xp.zeros_like(B_IN)
    _expand_and_validate(sdfg, expansion_type)
    sdfg(A_IN=A_IN, A_OUT=A_OUT, B_IN=B_IN, B_OUT=B_OUT)

    assert xp.allclose(A_IN, A_OUT)
    assert xp.allclose(B_IN, B_OUT)


@pytest.mark.parametrize("expansion_type", EXPANSION_TYPES)
@temporarily_disable_autoopt_and_serialization
def test_multi_memset(expansion_type, xp):
    sdfg = _prepare_sdfg(_get_sdfg(0, 2, False, False, False), expansion_type, "multi_memset")
    AssignmentAndCopyKernelToMemsetAndMemcpy().apply_pass(sdfg, {})
    assert _get_num_memcpy_library_nodes(sdfg) == 0
    assert _get_num_memset_library_nodes(sdfg) == 2

    A_IN = xp.random.rand(DIM_SIZE, DIM_SIZE)
    A_OUT = xp.zeros_like(A_IN)
    B_IN = xp.random.rand(DIM_SIZE, DIM_SIZE)
    B_OUT = xp.zeros_like(B_IN)
    _expand_and_validate(sdfg, expansion_type)
    sdfg(A_IN=A_IN, A_OUT=A_OUT, B_IN=B_IN, B_OUT=B_OUT)

    assert xp.allclose(A_OUT, 0.0)
    assert xp.allclose(B_OUT, 0.0)


@pytest.mark.parametrize("expansion_type", EXPANSION_TYPES)
@temporarily_disable_autoopt_and_serialization
def test_multi_mixed(expansion_type, xp):
    sdfg = _prepare_sdfg(_get_sdfg(1, 1, False, False, False), expansion_type, "multi_mixed")
    AssignmentAndCopyKernelToMemsetAndMemcpy().apply_pass(sdfg, {})
    assert _get_num_memcpy_library_nodes(sdfg) == 1
    assert _get_num_memset_library_nodes(sdfg) == 1

    A_IN = xp.random.rand(DIM_SIZE, DIM_SIZE)
    A_OUT = xp.zeros_like(A_IN)
    B_IN = xp.random.rand(DIM_SIZE, DIM_SIZE)
    B_OUT = xp.zeros_like(B_IN)
    _expand_and_validate(sdfg, expansion_type)
    sdfg(A_IN=A_IN, A_OUT=A_OUT, B_IN=B_IN, B_OUT=B_OUT)

    assert xp.allclose(A_IN, A_OUT)
    assert xp.allclose(B_OUT, 0.0)


@pytest.mark.parametrize("expansion_type", EXPANSION_TYPES)
@temporarily_disable_autoopt_and_serialization
def test_simple_with_extra_computation(expansion_type, xp):
    sdfg = _prepare_sdfg(_get_sdfg(2, 2, True, False, False), expansion_type, "simple_with_extra_computation")
    AssignmentAndCopyKernelToMemsetAndMemcpy().apply_pass(sdfg, {})

    A_IN = xp.random.rand(DIM_SIZE, DIM_SIZE)
    A_OUT = xp.zeros_like(A_IN)
    B_IN = xp.random.rand(DIM_SIZE, DIM_SIZE)
    B_OUT = xp.zeros_like(B_IN)
    C_IN = xp.random.rand(DIM_SIZE, DIM_SIZE)
    C_OUT = xp.zeros_like(C_IN)
    D_IN = xp.random.rand(DIM_SIZE, DIM_SIZE)
    D_OUT = xp.zeros_like(D_IN)
    _expand_and_validate(sdfg, expansion_type)
    sdfg(A_IN=A_IN, A_OUT=A_OUT, B_IN=B_IN, B_OUT=B_OUT, C_IN=C_IN, C_OUT=C_OUT, D_IN=D_IN, D_OUT=D_OUT)

    assert xp.allclose(A_OUT, 2 * A_IN)
    assert xp.allclose(B_OUT, B_IN)
    assert xp.allclose(C_OUT, 0.0)
    assert xp.allclose(D_OUT, 0.0)


@pytest.mark.parametrize("expansion_type", EXPANSION_TYPES)
@temporarily_disable_autoopt_and_serialization
def test_simple_non_zero(expansion_type, xp):
    sdfg = _prepare_sdfg(_get_sdfg(0, 1, False, True, False), expansion_type, "simple_nonzero")
    AssignmentAndCopyKernelToMemsetAndMemcpy().apply_pass(sdfg, {})

    A_IN = xp.random.rand(DIM_SIZE, DIM_SIZE)
    A_OUT = xp.zeros_like(A_IN)
    _expand_and_validate(sdfg, expansion_type)
    sdfg(A_IN=A_OUT, A_OUT=A_OUT)

    assert xp.allclose(A_OUT, 1.0)


@pytest.mark.parametrize("expansion_type", EXPANSION_TYPES)
@temporarily_disable_autoopt_and_serialization
def test_mixed_overapprox(expansion_type, xp):
    sdfg = _prepare_sdfg(_get_sdfg(2, 2, False, False, True), expansion_type, "mixed_overapprox")
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

    _expand_and_validate(sdfg, expansion_type)
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
    return sdfg


# expected_memcpy is 1 only with fortran_strides=True — C-strides can't be
# collapsed into a single memcpy because of the dimension change.
@pytest.mark.parametrize("expansion_type", EXPANSION_TYPES)
@pytest.mark.parametrize(
    "full_inner_range,fortran_strides,expected_memcpy",
    [(True, True, 1), (False, True, 1), (True, False, 0), (False, False, 0)],
)
@temporarily_disable_autoopt_and_serialization
def test_nested_memcpy_with_dimension_change_and_strides(expansion_type, xp, full_inner_range, fortran_strides,
                                                         expected_memcpy):
    sdfg = _get_nested_memcpy_with_dimension_change_and_fortran_strides(full_inner_range=full_inner_range,
                                                                        fortran_strides=fortran_strides)
    _prepare_sdfg(sdfg, expansion_type, f"full_inner_range_{full_inner_range}_fortran_strides_{fortran_strides}")

    AssignmentAndCopyKernelToMemsetAndMemcpy(overapproximate_first_dimensions=True).apply_pass(sdfg, {})
    assert _get_num_memcpy_library_nodes(sdfg) == expected_memcpy
    assert _get_num_memset_library_nodes(sdfg) == 0

    A_IN = xp.fromfunction(lambda x: x, (DIM_SIZE, ), dtype=xp.float64).copy()
    B_IN = xp.fromfunction(lambda x, y: x * DIM_SIZE + y, (DIM_SIZE, DIM_SIZE), dtype=xp.float64).copy()
    _expand_and_validate(sdfg, expansion_type)
    sdfg(zcovptot=A_IN, pcovptot=B_IN, kidia=0, kfdia=DIM_SIZE, D=DIM_SIZE)

    if fortran_strides:
        assert xp.allclose(A_IN, B_IN)
    else:
        for j in range(DIM_SIZE):
            assert xp.allclose(B_IN[0:DIM_SIZE, j], A_IN), f"{j}: {B_IN[0:DIM_SIZE, j] - A_IN}"


def test_transpose_map_is_not_lifted_to_memcpy():
    """Pin: a map whose tasklet body is `_out = _in` but whose in/out
    memlet subsets permute the map indices is a *transpose*, not a
    pure copy. The pass must leave it alone — lifting it to a
    ``CopyLibraryNode`` (which lowers to ``cudaMemcpyAsync``) would
    silently turn a transpose into a flat memcpy and produce wrong
    output. Regressed in ``test_persistent_gpu_transpose_regression``.
    """
    sdfg = dace.SDFG("transpose_pin")
    sdfg.add_array("A", [5, 3], dace.float64)
    sdfg.add_array("AT", [3, 5], dace.float64)
    state = sdfg.add_state("main")
    a = state.add_access("A")
    at = state.add_access("AT")
    me, mx = state.add_map("transpose_map", {"i": "0:5", "j": "0:3"})
    t = state.add_tasklet("tr", {"_in"}, {"_out"}, "_out = _in")
    state.add_memlet_path(a, me, t, dst_conn="_in", memlet=dace.Memlet("A[i, j]"))
    state.add_memlet_path(t, mx, at, src_conn="_out", memlet=dace.Memlet("AT[j, i]"))

    AssignmentAndCopyKernelToMemsetAndMemcpy().apply_pass(sdfg, {})
    assert _get_num_memcpy_library_nodes(sdfg) == 0, (
        "Transpose pattern (in subset [i, j], out subset [j, i]) was incorrectly "
        "lifted to a CopyLibraryNode — the pass treats permutation as pure copy.")


def test_inkernel_memset_is_not_lifted():
    """Pin: a memset (``scratch[j] = 0``) that sits *inside* a GPU_Device
    map must NOT be lifted to a ``MemsetLibraryNode``. The libnode expands
    to ``cudaMemsetAsync``, which is a host-only runtime entry point and
    cannot be issued from device code. Also, the expansion produces a
    ``GPU_Device``-scheduled mapped tasklet that, when nested inside
    another GPU map, fails ``AddThreadBlockMap.can_be_applied`` and
    breaks ``InferGPUGridAndBlockSize`` downstream. Regressed in the
    four ``nested_kernel_transient_test`` variants.

    Uses ``simplify=True`` so the inner Sequential map's ``MapEntry ->
    Tasklet -> MapExit -> AccessNode`` shape is actually present at the
    top level — with ``simplify=False`` the frontend wraps each map body
    in a NestedSDFG and the lift pattern never matches even when the
    precondition is off.
    """

    @dace.program
    def kernel_with_inner_memset(A: dace.float64[128, 64] @ dace.StorageType.GPU_Global):
        for i in dace.map[0:128] @ dace.ScheduleType.GPU_Device:
            scratch = dace.define_local([64], numpy.float64, storage=dace.StorageType.GPU_Global)
            for j in dace.map[0:64] @ dace.ScheduleType.Sequential:
                scratch[j] = 0
            A[i, :] = scratch

    sdfg = kernel_with_inner_memset.to_sdfg(simplify=True)
    AssignmentAndCopyKernelToMemsetAndMemcpy().apply_pass(sdfg, {})
    assert _get_num_memset_library_nodes(sdfg) == 0, (
        "An in-kernel memset (Sequential map inside GPU_Device) was lifted to a "
        "MemsetLibraryNode — but cudaMemsetAsync is host-only and cannot run from "
        "device code. The pass should skip maps nested in any GPU scope.")


def test_single_element_memset_is_not_lifted():
    """Pin: a memset over a single-element array must NOT be lifted.

    The pure expansion of ``MemsetLibraryNode`` collapses every singleton
    dim (``_make_memset_skeleton``'s ``keep`` filter); a 1-element memset
    therefore lifts to a mapped tasklet with an empty map, which downstream
    ``propagate_memlet`` rejects with ``TypeError: object of type 'NoneType'
    has no len()``. Skip the lift entirely.
    """

    @dace.program
    def single_element_zero(A: dace.float64[1]):
        for i in dace.map[0:1]:
            A[i] = 0

    sdfg = single_element_zero.to_sdfg(simplify=True)
    AssignmentAndCopyKernelToMemsetAndMemcpy().apply_pass(sdfg, {})
    assert _get_num_memset_library_nodes(sdfg) == 0, (
        "A single-element memset was lifted to a MemsetLibraryNode; the pure "
        "expansion would collapse to an empty map and crash propagation.")


def test_single_element_memcpy_is_not_lifted():
    """Pin: a memcpy over a single element must NOT be lifted (same family
    as ``test_single_element_memset_is_not_lifted`` — singleton-collapse in
    ``CopyLibraryNode``'s pure expansion produces a degenerate map).
    """

    @dace.program
    def single_element_copy(A: dace.float64[1], B: dace.float64[1]):
        for i in dace.map[0:1]:
            B[i] = A[i]

    sdfg = single_element_copy.to_sdfg(simplify=True)
    AssignmentAndCopyKernelToMemsetAndMemcpy().apply_pass(sdfg, {})
    assert _get_num_memcpy_library_nodes(sdfg) == 0, (
        "A single-element memcpy was lifted to a CopyLibraryNode; the pure "
        "expansion would collapse to an empty map and crash propagation.")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
