# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``MaskedCopyLibraryNode`` loads and stores a tile under a mask, and refuses what it cannot copy."""

import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes.copy.common import INPUT_CONNECTOR_NAME, OUTPUT_CONNECTOR_NAME
from dace.libraries.tileops import MaskedCopyLibraryNode, TileGather
from dace.libraries.tileops.dispatch import ISA_TO_IMPL, detect_host_isa

MASKS = {
    "all_active": [True] * 8,
    "tail": [True] * 5 + [False] * 3,
    "all_inactive": [False] * 8,
    "alternating": [True, False] * 4,
}
#: ``(shape of the array, window of it, shape of the tile, step of the window along the tile's last dim)``
WINDOWS = {
    "contiguous": ((24,), "5:13", (8,), 1),
    "strided": ((40,), "3:35:4", (8,), 4),
    "row_of_a_matrix": ((6, 20), "3, 4:12", (8,), 1),
    "matrix_window": ((6, 20), "1:5, 2:10", (4, 8), 1),
}
LANES = 8
IMPLEMENTATIONS = ["pure", "scalar", ISA_TO_IMPL[detect_host_isa()]]
#: The header backends move one lane dim, so a 2-D tile is the pure loop whichever implementation is asked for.
CASES = [
    (window, implementation)
    for window in WINDOWS
    for implementation in IMPLEMENTATIONS
    if implementation == "pure" or len(WINDOWS[window][2]) == 1
]


def tile_subset(shape):
    return ", ".join(f"0:{extent}" for extent in shape)


def lane_mask(mask, shape):
    """The mask over the lanes of a tile of ``shape``; the 2-D tile repeats the 1-D pattern along its rows."""
    # ``np.resize`` returns a reshaped view, and programs take arrays that own their memory
    return np.resize(np.array(mask, dtype=bool), shape).copy()


def window_slices(window):
    return tuple(
        slice(*(int(bound) if bound else None for bound in part.split(":"))) if ":" in part else int(part)
        for part in window.split(", ")
    )


def masked_copy(name, shape, implementation):
    node = MaskedCopyLibraryNode(name, widths=shape)
    node.implementation = implementation
    return node


def build_load(name, array_shape, window, shape, dtype, implementation):
    """``A[window]`` loaded into a register tile under ``M``, then copied out to ``OUT``."""
    sdfg = dace.SDFG(f"masked_load_{name}")
    sdfg.add_array("A", array_shape, dtype)
    sdfg.add_array("M", shape, dace.bool_)
    sdfg.add_array("OUT", shape, dtype)
    sdfg.add_array("TILE", shape, dtype, transient=True, storage=dace.StorageType.Register)
    sdfg.add_array("MASKT", shape, dace.bool_, transient=True, storage=dace.StorageType.Register)
    state = sdfg.add_state()
    node = masked_copy(name, shape, implementation)
    state.add_node(node)
    tile, mask = state.add_access("TILE"), state.add_access("MASKT")
    state.add_edge(state.add_read("M"), None, mask, None, dace.Memlet(f"M[{tile_subset(shape)}]"))
    state.add_edge(mask, None, node, "_mask", dace.Memlet(f"MASKT[{tile_subset(shape)}]"))
    state.add_edge(state.add_read("A"), None, node, INPUT_CONNECTOR_NAME, dace.Memlet(f"A[{window}]"))
    state.add_edge(node, OUTPUT_CONNECTOR_NAME, tile, None, dace.Memlet(f"TILE[{tile_subset(shape)}]"))
    state.add_edge(tile, None, state.add_write("OUT"), None, dace.Memlet(f"OUT[{tile_subset(shape)}]"))
    return sdfg


def build_store(name, array_shape, window, shape, dtype, implementation):
    """The tile ``IN`` stored into ``B[window]`` under ``M``."""
    sdfg = dace.SDFG(f"masked_store_{name}")
    sdfg.add_array("IN", shape, dtype)
    sdfg.add_array("M", shape, dace.bool_)
    sdfg.add_array("B", array_shape, dtype)
    sdfg.add_array("TILE", shape, dtype, transient=True, storage=dace.StorageType.Register)
    sdfg.add_array("MASKT", shape, dace.bool_, transient=True, storage=dace.StorageType.Register)
    state = sdfg.add_state()
    node = masked_copy(name, shape, implementation)
    state.add_node(node)
    tile, mask = state.add_access("TILE"), state.add_access("MASKT")
    state.add_edge(state.add_read("IN"), None, tile, None, dace.Memlet(f"IN[{tile_subset(shape)}]"))
    state.add_edge(state.add_read("M"), None, mask, None, dace.Memlet(f"M[{tile_subset(shape)}]"))
    state.add_edge(mask, None, node, "_mask", dace.Memlet(f"MASKT[{tile_subset(shape)}]"))
    state.add_edge(tile, None, node, INPUT_CONNECTOR_NAME, dace.Memlet(f"TILE[{tile_subset(shape)}]"))
    state.add_edge(node, OUTPUT_CONNECTOR_NAME, state.add_write("B"), None, dace.Memlet(f"B[{window}]"))
    return sdfg


@pytest.mark.parametrize("dtype", [dace.float64, dace.int32], ids=lambda dtype: dtype.to_string())
@pytest.mark.parametrize("window_name,implementation", CASES)
def test_a_masked_load_fills_the_active_lanes_and_zeroes_the_others(window_name, implementation, dtype):
    array_shape, window, shape = WINDOWS[window_name][:3]
    sdfg = build_load(
        f"{window_name}_{implementation}_{dtype.to_string()}", array_shape, window, shape, dtype, implementation
    )
    compiled = sdfg.compile()
    data = (np.arange(np.prod(array_shape)).reshape(array_shape) + 1).astype(dtype.as_numpy_dtype())
    for mask_name, mask in MASKS.items():
        out = np.full(shape, 99, dtype=dtype.as_numpy_dtype())
        compiled(A=data, M=lane_mask(mask, shape), OUT=out)
        active = lane_mask(mask, shape)
        expected = np.where(active, data[window_slices(window)], 0)
        np.testing.assert_array_equal(out, expected, err_msg=mask_name)


@pytest.mark.parametrize("dtype", [dace.float64, dace.int32], ids=lambda dtype: dtype.to_string())
@pytest.mark.parametrize("window_name,implementation", CASES)
def test_a_masked_store_writes_the_active_lanes_and_leaves_the_others(window_name, implementation, dtype):
    array_shape, window, shape = WINDOWS[window_name][:3]
    sdfg = build_store(
        f"{window_name}_{implementation}_{dtype.to_string()}", array_shape, window, shape, dtype, implementation
    )
    compiled = sdfg.compile()
    tile = (np.arange(np.prod(shape)).reshape(shape) + 1000).astype(dtype.as_numpy_dtype())
    for mask_name, mask in MASKS.items():
        array = np.full(array_shape, -7, dtype=dtype.as_numpy_dtype())
        compiled(IN=tile, M=lane_mask(mask, shape), B=array)
        expected = np.full(array_shape, -7, dtype=dtype.as_numpy_dtype())
        view = expected[window_slices(window)]
        view[...] = np.where(lane_mask(mask, shape), tile, view)
        np.testing.assert_array_equal(array, expected, err_msg=mask_name)


def tile_node_load(name, array_shape, window, shape, step, dtype):
    sdfg = dace.SDFG(f"tile_load_{name}")
    sdfg.add_array("A", array_shape, dtype)
    sdfg.add_array("M", shape, dace.bool_)
    sdfg.add_array("OUT", shape, dtype)
    sdfg.add_array("TILE", shape, dtype, transient=True, storage=dace.StorageType.Register)
    sdfg.add_array("MASKT", shape, dace.bool_, transient=True, storage=dace.StorageType.Register)
    state = sdfg.add_state()
    node = TileGather(name, widths=shape, has_mask=True, dim_strides=[step] * len(shape))
    state.add_node(node)
    tile, mask = state.add_access("TILE"), state.add_access("MASKT")
    state.add_edge(state.add_read("M"), None, mask, None, dace.Memlet(f"M[{tile_subset(shape)}]"))
    state.add_edge(mask, None, node, "_mask", dace.Memlet(f"MASKT[{tile_subset(shape)}]"))
    state.add_edge(state.add_read("A"), None, node, "_src", dace.Memlet(f"A[{window}]"))
    state.add_edge(node, "_dst", tile, None, dace.Memlet(f"TILE[{tile_subset(shape)}]"))
    state.add_edge(tile, None, state.add_write("OUT"), None, dace.Memlet(f"OUT[{tile_subset(shape)}]"))
    return sdfg


@pytest.mark.parametrize("window_name", ["contiguous", "strided"])
def test_a_masked_load_equals_the_tile_load_it_replaces(window_name):
    array_shape, window, shape, step = WINDOWS[window_name]
    data = np.random.default_rng(3).random(array_shape)
    copy = build_load(f"equals_{window_name}", array_shape, window, shape, dace.float64, "pure").compile()
    load = tile_node_load(f"equals_{window_name}", array_shape, window, shape, step, dace.float64).compile()
    for mask in MASKS.values():
        copied, loaded = np.full(shape, 5.0), np.full(shape, 5.0)
        copy(A=data, M=lane_mask(mask, shape), OUT=copied)
        load(A=data, M=lane_mask(mask, shape), OUT=loaded)
        np.testing.assert_array_equal(copied, loaded)


def test_a_masked_load_reads_a_lane_only_where_the_mask_is_on():
    """The window may reach past the end of the array in a tail, so the read of an inactive lane is guarded."""
    sdfg = build_load("guard", (24,), "5:13", (8,), dace.float64, "pure")
    sdfg.expand_library_nodes()
    code = next(
        node for node, state in sdfg.all_nodes_recursive() if isinstance(node, dace.nodes.Tasklet)
    ).code.as_string
    assert "_mask[__l0] ? _cpy_in[(__l0 * (1))] : double(0)" in code, code


def test_a_masked_load_selects_the_header_call_with_the_stride_of_its_window():
    sdfg = build_load("header", (40,), "3:35:4", (8,), dace.float64, "scalar")
    sdfg.expand_library_nodes()
    code = next(
        node for node, state in sdfg.all_nodes_recursive() if isinstance(node, dace.nodes.Tasklet)
    ).code.as_string
    assert "dace::tileops::tile_load<double, 8, true>(_cpy_out, _cpy_in, _mask, 4);" in code, code


@pytest.mark.parametrize("window,alignment", [("0:8", "16"), ("3:11", "4, 1"), ("4:12", "8")])
@pytest.mark.parametrize("masked", [False, True])
def test_a_half_precision_tile_is_loaded_with_the_alignment_its_window_proves(window, alignment, masked):
    """The CUDA backend widens an fp16 access it can prove aligned: a window at element 0 to 16 bytes, one at element 4
    to 8, and one at element 3 to the 4-byte word below it with a shift of one."""
    sdfg = dace.SDFG(f"masked_align_{window.replace(':', '_')}_{int(masked)}")
    sdfg.add_array("A", (1024,), dace.float16, storage=Storage.GPU_Global)
    sdfg.add_array("TILE", (8,), dace.float16, storage=Storage.Register, transient=True)
    sdfg.add_array("MASKT", (8,), dace.bool_, storage=Storage.Register, transient=True)
    state = sdfg.add_state()
    copy = MaskedCopyLibraryNode("copy", widths=(8,), has_mask=masked)
    state.add_node(copy)
    state.add_edge(state.add_access("A"), None, copy, INPUT_CONNECTOR_NAME, dace.Memlet(f"A[{window}]"))
    state.add_edge(copy, OUTPUT_CONNECTOR_NAME, state.add_access("TILE"), None, dace.Memlet("TILE[0:8]"))
    if masked:
        state.add_edge(state.add_access("MASKT"), None, copy, "_mask", dace.Memlet("MASKT[0:8]"))
    code = copy.isa_tasklet(state, sdfg, "cuda").code.as_string
    assert code.startswith(f"dace::tileops::tile_load<dace::float16, 8, {str(masked).lower()}, {alignment}>"), code


def wired_copy(
    source_storage, destination_storage, source_transient, destination_transient, shape=(8,), dtype=dace.float64
):
    sdfg = dace.SDFG("masked_copy_storages")
    sdfg.add_array("S", shape, dtype, storage=source_storage, transient=source_transient)
    sdfg.add_array("D", shape, dtype, storage=destination_storage, transient=destination_transient)
    sdfg.add_array("MASKT", shape, dace.bool_, storage=dace.StorageType.Register, transient=True)
    state = sdfg.add_state()
    node = MaskedCopyLibraryNode("c", widths=shape)
    state.add_node(node)
    full = tile_subset(shape)
    state.add_edge(state.add_access("S"), None, node, INPUT_CONNECTOR_NAME, dace.Memlet(f"S[{full}]"))
    state.add_edge(state.add_access("MASKT"), None, node, "_mask", dace.Memlet(f"MASKT[{full}]"))
    state.add_edge(node, OUTPUT_CONNECTOR_NAME, state.add_access("D"), None, dace.Memlet(f"D[{full}]"))
    return node, state, sdfg


Storage = dace.StorageType


@pytest.mark.parametrize(
    "source,destination",
    [
        (Storage.GPU_Global, Storage.Register),
        (Storage.GPU_Global, Storage.GPU_Shared),
        (Storage.CPU_Heap, Storage.Register),
        (Storage.Register, Storage.GPU_Global),
        (Storage.GPU_Shared, Storage.GPU_Global),
        (Storage.Register, Storage.CPU_Heap),
    ],
)
def test_the_storage_pairs_of_a_load_or_a_store_validate(source, destination):
    node, state, sdfg = wired_copy(source, destination, False, False)
    node.validate(sdfg, state)


@pytest.mark.parametrize(
    "source,destination",
    [
        (Storage.CPU_Heap, Storage.CPU_Heap),
        (Storage.GPU_Shared, Storage.Register),
        (Storage.CPU_Pinned, Storage.Register),
    ],
)
def test_any_other_storage_pair_is_refused(source, destination):
    node, state, sdfg = wired_copy(source, destination, False, False)
    with pytest.raises(NotImplementedError, match="no masked copy"):
        node.validate(sdfg, state)


@pytest.mark.parametrize("storage", [Storage.Default, Storage.Register])
@pytest.mark.parametrize("source_transient,destination_transient,loads", [(False, True, True), (True, False, False)])
def test_a_copy_within_one_storage_reads_its_direction_off_the_transient_tile(
    storage, source_transient, destination_transient, loads
):
    from dace.libraries.tileops.nodes.masked_copy import is_load

    node, state, sdfg = wired_copy(storage, storage, source_transient, destination_transient)
    node.validate(sdfg, state)
    assert is_load(sdfg.arrays["S"], sdfg.arrays["D"], "c") is loads


@pytest.mark.parametrize("storage", [Storage.Default, Storage.Register])
@pytest.mark.parametrize("source_transient,destination_transient", [(False, False), (True, True)])
def test_a_copy_within_one_storage_with_no_single_tile_is_refused_not_guessed(
    storage, source_transient, destination_transient
):
    node, state, sdfg = wired_copy(storage, storage, source_transient, destination_transient)
    with pytest.raises(NotImplementedError, match="does not say whether it loads or stores"):
        node.validate(sdfg, state)


def test_a_view_is_never_the_tile_of_a_copy():
    from dace.libraries.tileops.nodes.masked_copy import is_load

    sdfg = dace.SDFG("masked_copy_views")
    sdfg.add_view("V", (8,), dace.float64, storage=Storage.Register)
    sdfg.add_array("T", (8,), dace.float64, storage=Storage.Register, transient=True)
    assert is_load(sdfg.arrays["V"], sdfg.arrays["T"], "c")
    assert not is_load(sdfg.arrays["T"], sdfg.arrays["V"], "c")


def test_a_transposed_window_is_not_a_masked_copy():
    sdfg = build_store("transposed", (20, 6), "2:10, 1:5", (4, 8), dace.float64, "pure")
    with pytest.raises(ValueError, match="does not transpose or reshape"):
        sdfg.expand_library_nodes()


def test_a_masked_copy_does_not_convert_dtypes():
    node, state, sdfg = wired_copy(Storage.CPU_Heap, Storage.Register, False, False)
    sdfg.arrays["D"].dtype = dace.float32
    with pytest.raises(ValueError, match="does not convert dtypes"):
        node.validate(sdfg, state)


def test_the_mask_must_be_a_register_tile_of_the_widths():
    node, state, sdfg = wired_copy(Storage.CPU_Heap, Storage.Register, False, False)
    sdfg.arrays["MASKT"].storage = Storage.CPU_Heap
    with pytest.raises(ValueError, match="must be Register"):
        node.validate(sdfg, state)


if __name__ == "__main__":
    for tile_dtype in (dace.float64, dace.int32):
        for tile_window, tile_implementation in CASES:
            test_a_masked_load_fills_the_active_lanes_and_zeroes_the_others(
                tile_window, tile_implementation, tile_dtype
            )
            test_a_masked_store_writes_the_active_lanes_and_leaves_the_others(
                tile_window, tile_implementation, tile_dtype
            )
    for tile_window in ("contiguous", "strided"):
        test_a_masked_load_equals_the_tile_load_it_replaces(tile_window)
    test_a_masked_load_reads_a_lane_only_where_the_mask_is_on()
    test_a_masked_load_selects_the_header_call_with_the_stride_of_its_window()
    for tile_masked in (False, True):
        test_a_half_precision_tile_is_loaded_with_the_alignment_its_window_proves("0:8", "16", tile_masked)
        test_a_half_precision_tile_is_loaded_with_the_alignment_its_window_proves("3:11", "4, 1", tile_masked)
        test_a_half_precision_tile_is_loaded_with_the_alignment_its_window_proves("4:12", "8", tile_masked)
    for tile_source, tile_destination in (
        (Storage.GPU_Global, Storage.Register),
        (Storage.GPU_Global, Storage.GPU_Shared),
        (Storage.CPU_Heap, Storage.Register),
        (Storage.Register, Storage.GPU_Global),
        (Storage.GPU_Shared, Storage.GPU_Global),
        (Storage.Register, Storage.CPU_Heap),
    ):
        test_the_storage_pairs_of_a_load_or_a_store_validate(tile_source, tile_destination)
    for tile_source, tile_destination in (
        (Storage.CPU_Heap, Storage.CPU_Heap),
        (Storage.GPU_Shared, Storage.Register),
        (Storage.CPU_Pinned, Storage.Register),
    ):
        test_any_other_storage_pair_is_refused(tile_source, tile_destination)
    for tile_storage in (Storage.Default, Storage.Register):
        test_a_copy_within_one_storage_reads_its_direction_off_the_transient_tile(tile_storage, False, True, True)
        test_a_copy_within_one_storage_reads_its_direction_off_the_transient_tile(tile_storage, True, False, False)
        test_a_copy_within_one_storage_with_no_single_tile_is_refused_not_guessed(tile_storage, False, False)
        test_a_copy_within_one_storage_with_no_single_tile_is_refused_not_guessed(tile_storage, True, True)
    test_a_view_is_never_the_tile_of_a_copy()
    test_a_transposed_window_is_not_a_masked_copy()
    test_a_masked_copy_does_not_convert_dtypes()
    test_the_mask_must_be_a_register_tile_of_the_widths()
